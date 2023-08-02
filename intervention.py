# make deterministic
import argparse
import os

import math
import time
import numpy as np
from copy import deepcopy
import pickle
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
from torch.utils.data import Subset
from tqdm import tqdm
# from matplotlib import pyplot as plt

from data import get_othello, plot_probs, plot_mentals
from data.othello import permit, start_hands, OthelloBoardState, permit_reverse
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig, GPTforProbeIA
from mingpt.utils import sample, intervene, print_board, set_seed
from mingpt.probe_model import BatteryProbeClassification, BatteryProbeClassificationTwoLayer

htd = {"lr": 1e-3, "steps": 1000, "reg_strg": 0.2}

device = torch.cuda.current_device()


def parseArguments(): 
    parser = argparse.ArgumentParser()

    # Necessary variables
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report_step", type=int, default=100)
    parser.add_argument("--seed", type=int, default=44)
    
    parser.add_argument("--layer_s", type=int, required=True)
    parser.add_argument("--layer_e", type=int, required=True)
    parser.add_argument("--mid_dim", type=int, required=True)
    parser.add_argument("--num_intervention", type=int, default=1000)
    parser.add_argument("--championship", action="store_true")
    parser.add_argument("--how_many_history_step_to_use", type=int, default=99)
    
    parser.add_argument("--repo_root", type=str, default="/users/tyun/data/tyun/vlm_rule_learning/othello_world")
    parser.add_argument("--output_path", type=str, default="/users/tyun/data/tyun/vlm_rule_learning/othello_world/outputs")
    args = parser.parse_args()
    return args


def load_intervention_data(args):
    data_path = os.path.join(args.repo_root, "intervention_benchmark.pkl")
    with open(data_path, "rb") as input_file:
        dataset = pickle.load(input_file)
    return dataset


def load_probes(args):
    exp = f"state_tl{args.mid_dim}"
    if args.championship:
        exp += "_championship"
       
    probes = {}
    for layer in range(args.layer_s, args.layer_e):
        p = BatteryProbeClassificationTwoLayer(torch.cuda.current_device(), probe_class=3, num_task=64, mid_dim=args.mid_dim)
        p_path = os.path.join(args.repo_root, f"./ckpts/battery_othello/{exp}/layer{layer}/checkpoint.ckpt")
        load_res = p.load_state_dict(torch.load(p_path))
        p.eval()
        probes[layer] = p
    return probes
      
    
def load_model_and_train_dataset(args):
    othello = get_othello(ood_perc=0., data_root=None, wthor=False, ood_num=1)
    train_dataset = CharDataset(othello)

    mconf = GPTConfig(61, 59, n_layer=8, n_head=8, n_embd=512)

    models = {}
    for layer in range(args.layer_s, args.layer_e):
        model = GPTforProbeIA(mconf, probe_layer=layer)
        # model = GPT(mconf)
        model_path = os.path.join(args.repo_root, "./ckpts/gpt_synthetic.ckpt" if not args.championship else "./ckpts/gpt_championship.ckpt")
        load_res = model.load_state_dict(torch.load(model_path))
        if torch.cuda.is_available():
            model = model.to(device)
        _ = model.eval()
        models[layer] = model
    return models, train_dataset


def get_wtd_list(case_id, dataset, args):
    wtd = {
        "intervention_position": permit_reverse(dataset[case_id]["pos_int"]), 
        "intervention_from": dataset[case_id]["ori_color"], 
        "intervention_to": 2 - dataset[case_id]["ori_color"], 
    }
    wtd_list = [wtd]
    return wtd_list


def get_post_intv_valids_labels(wtd_list, completion):
    ab = OthelloBoardState()
    ab.update(completion, prt=False)
    pre_intv_valids = [permit_reverse(_) for _ in ab.get_valid_moves()]
    padding = torch.zeros(2).cuda()
    for wtd in wtd_list:
        move = permit(wtd["intervention_position"])
        r, c = move // 8, move % 8
        ab.state[r, c] = wtd["intervention_to"] - 1
    post_intv_valids_labels = set([permit(permit_reverse(_)) for _ in ab.get_valid_moves()])
    return post_intv_valids_labels


def compute_error(preds, labels):
    correct = len(preds.intersection(labels))
    return len(labels) - correct


def get_top_n_preds(preds, n):
    preds = preds[0, -1, 1:]
    preds = torch.softmax(preds, dim=0)
    padding = torch.zeros(2).cuda()
    preds = torch.cat([preds[:27], padding, preds[27:33], padding, preds[33:]], dim=0)
    top_n_preds = preds.argsort(descending=True)[:n]
    top_n_preds = set(top_n_preds.detach().cpu().numpy())
    return top_n_preds


def main(args):
    set_seed(args.seed)
     
    print("Load data, probes, models.", flush=True)
    dataset = load_intervention_data(args)
    probes = load_probes(args)
    models, train_dataset = load_model_and_train_dataset(args)
    
    print("Start intervention", flush=True)
    pre_intv_error_list, post_intv_error_list = [],[]
    for case_id in tqdm(range(args.num_intervention)):
        # prepare data
        wtd_list = get_wtd_list(case_id, dataset, args)
        completion = dataset[case_id]["history"]
        partial_game = torch.tensor([train_dataset.stoi[s] for s in completion], dtype=torch.long).to(device)
        pre_intv_pred, _ = models[8](partial_game[None, :])  # [B, T, F=512]
        
        # intervene at the earliest interested layer
        p = probes[args.layer_s]
        whole_mid_act = models[args.layer_s].forward_1st_stage(partial_game[None, :])  # [B, T, F=512]
        mid_act = whole_mid_act[0, -1]
        pre_intv_logits = p(mid_act[None, :])[0].squeeze(0)  # [64, 3]
        labels_pre_intv = pre_intv_logits.detach().argmax(dim=-1)
        new_mid_act = mid_act.clone()
        for wtd in wtd_list:
            new_mid_act = intervene(p, new_mid_act, labels_pre_intv, wtd, htd, plot=True)
            pre_intv_logits = p(new_mid_act[None, :])[0].squeeze(0)  # [64, 3]
            labels_pre_intv = pre_intv_logits.detach().argmax(dim=-1)
        post_intv_logits = p(new_mid_act[None, :])[0].squeeze(0)  # [64, 3]

        # swap in 
        whole_mid_act[0, -1] = new_mid_act
        
        # intervene till the last interested layer
        for i, layer in enumerate(range(args.layer_s, args.layer_e - 1)):  # 4, 5, 6, 7, indices of the layers to be passed
            p = probes[layer+1]
            whole_mid_act = models[args.layer_s].forward_2nd_stage(whole_mid_act, layer, layer+1)[0]  # [1, T, F=512]

            # intervene the output of the features freshly out
            mid_act = whole_mid_act[0, -1]
            pre_intv_logits = p(mid_act[None, :])[0].squeeze(0)  # [64, 3]
            
            labels_pre_intv = pre_intv_logits.detach().argmax(dim=-1)
            new_mid_act = mid_act.clone()
            for wtd in wtd_list:
                new_mid_act = intervene(p, new_mid_act, labels_pre_intv, wtd, htd, plot=True)
                pre_intv_logits = p(new_mid_act[None, :])[0].squeeze(0)  # [64, 3]
                labels_pre_intv = pre_intv_logits.detach().argmax(dim=-1)
            post_intv_logits = p(new_mid_act[None, :])[0].squeeze(0)  # [64, 3]
            
            # swap in 
            whole_mid_act[0, -1] = new_mid_act
            
        # evaluate post-intervention error
        post_intv_valids_labels = get_post_intv_valids_labels(wtd_list, completion)
        
        tb_resumed = whole_mid_act
        post_intv_pred, _ = models[8].predict(tb_resumed)  # `8` predict at last layer
        n = len(post_intv_valids_labels)

        top_n_pre_intv_pred = get_top_n_preds(pre_intv_pred, n)
        top_n_post_intv_pred = get_top_n_preds(post_intv_pred, n)
        pre_intv_error = compute_error(top_n_pre_intv_pred, post_intv_valids_labels)
        post_intv_error = compute_error(top_n_post_intv_pred, post_intv_valids_labels)
        
        pre_intv_error_list.append(pre_intv_error)
        post_intv_error_list.append(post_intv_error)
    
#         print("labels:",post_intv_valids_labels)
#         print("pre:",pre_intv_error)
#         print(top_n_pre_intv_pred)
#         print("post:",post_intv_error)
#         print(top_n_post_intv_pred)
        
    print("Done!", flush=True)
    print(f"| championship {args.championship} | layer_s {args.layer_s} | layer_e {args.layer_e} | mid_dim {args.mid_dim} | num_intervention {args.num_intervention} |", flush=True)
    print(f"pre_intv_error: {np.mean(np.array(pre_intv_error_list)):8.6f} +- {np.std(np.array(pre_intv_error_list)):8.6f}", flush=True)
    print(f"post_intv_error: {np.mean(np.array(post_intv_error_list)):8.6f} +- {np.std(np.array(post_intv_error_list)):8.6f}", flush=True)

    
if __name__ == "__main__":
    args = parseArguments()
    main(args)


