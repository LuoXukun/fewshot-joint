#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.10.29

import torch
import random
import numpy as np
import argparse

from config import *

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_parameters():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--load_ckpt", default=None, type=str,
                        help="Directory of checkpoints for loading. Default: None.")
    parser.add_argument("--save_ckpt", default=best_model_path, type=str,
                        help="Directory of checkpoints for saving. Default: None.")
    parser.add_argument("--log_path", default=log_path, type=str,
                        help="Log path.")
    parser.add_argument("--pre_ckpt", default=best_model_path, type=str,
                        help="Directory of pre-train checkpoints for saving. Default: None.")

    # Model options.
    parser.add_argument("--model_type", 
        choices=["few-tplinker", "few-tplinker-plus"],
        default="few-tplinker",
        help="Few shot model type.")
    parser.add_argument("--plus_type", 
        choices=["dot-sigmoid", "top", "top-k-average"],
        default="dot-sigmoid",
        help="The plus type of few-tplinker-plus.")
    parser.add_argument("--shaking_type", 
        choices=["cat", "cat_plus", "cln", "cln_plus"],
        default="cat",
        help="The type of handshaking procedure.")
    parser.add_argument("--inner_enc_type", 
        choices=["lstm", "max_pooling", "mean_pooling", "mix_pooling"],
        default="lstm",
        help="The encoder type for generating inner context. There is no use if the shaking_type is not *_plus.")
    parser.add_argument("--dist_type", 
        choices=["dot", "euclidean"],
        default="euclidean",
        help="The type of calculating disance.")
    parser.add_argument("--train", type=int, default=1,
                        help="If train or test. 1->train, 0->test.")

    # Training options.
    parser.add_argument("--fewshot", type=int, default=1,
                        help="If build up the few-shot module.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch_size.")
    parser.add_argument("--trainN", type=int, default=2,
                        help="Number of classes for each batch when training.")
    parser.add_argument("--N", type=int, default=2,
                        help="Number of classes for each batch.")
    parser.add_argument("--K", type=int, default=2,
                        help="Number of instances of each class in support set.")
    parser.add_argument("--Q", type=int, default=3,
                        help="Number of instances of each class in query set.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--train_iter", type=int, default=600,
                        help="Num of iterations of training.")
    parser.add_argument("--val_iter", type=int, default=100,
                        help="Num of iterations of validating.")
    parser.add_argument("--val_step", type=int, default=20,
                        help="Validate every val_step steps.")
    parser.add_argument("--report_step", type=int, default=10,
                        help="Validate every train_step steps.")
    parser.add_argument("--grad_iter", type=int, default=1,
                        help="Iter of gradient descent. Default: 1.")
    parser.add_argument("--map_hidden_size", type=int, default=map_hidden_size,
                        help="The hidden size of mapping layer.")
    parser.add_argument('--seed', type=int, default=7,
                        help='random seed.')
    parser.add_argument("--warmup", type=int, default=0.1,
                        help="Warmup rate.")
    parser.add_argument("--use_fp16", type=int, default=1,
                        help="If use fp16.")

    # Pre-train options.
    parser.add_argument("--pretrain", type=int, default=0,
                        help="If build up the pre-train module.")
    parser.add_argument("--pre_batch_size", type=int, default=4,
                        help="Pre-train batch_size.")
    parser.add_argument("--pre_learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--pre_train_epoch", type=int, default=20,
                        help="Num of epochs of training.")
    parser.add_argument("--pre_val_epoch", type=int, default=2,
                        help="Validate every pre-train val_step epochs.")
    parser.add_argument("--pre_report_step", type=int, default=1000,
                        help="Validate every pre-train train_step steps.")
    parser.add_argument("--pre_warmup_rate", type=float, default=0.1,
                        help="Warming up rate of pre-training.")
    parser.add_argument("--pre_grad_iter", type=int, default=1,
                        help="Iter of gradient descent for pre-training. Default: 1.")
    
    # Data options.
    #parser.add_argument('--num_workers', type=int, default=4,
    #                    help='Number of thread workers for data loader.')
    parser.add_argument("--data_name", 
        choices=["NYT"],
        default="NYT",
        help="Name of dataset.")
    parser.add_argument("--data_type", 
        choices=["inter_data", "intra_data"],
        default="inter_data",
        help="Type of dataset.")
    parser.add_argument("--language", 
        choices=["en", "ch"],
        default="en",
        help="The language.")
    parser.add_argument('--group', type=int, default=0,
                        help='Group setting. Training set: 0 -> AB, 1 -> AC, 2 -> BC.')
    
    args = parser.parse_args()

    # Others.
    args.seq_max_length = seq_max_length
    args.label_max_length = label_max_length
    args.warmup_step = int(args.warmup * args.train_iter)
    args.split_size = split_size
    args.tag_seqs_num = tag_seqs_num[args.model_type]
    args.train = True if args.train else False
    args.use_fp16 = True if args.use_fp16 else False
    args.fewshot = True if args.fewshot else False
    args.pretrain = True if args.pretrain else False
    
    return args
