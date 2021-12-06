#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.11.30

""" Data loader for pre-training models. """

import os
import json
import copy
import torch
import loguru
import argparse

from tqdm import tqdm
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader

from models import PreDataMaker, TaggingScheme
from utils.datapreprocess import get_norm_data
from config import data_path_index, data_path_temp, model_type_dict, seq_max_length, label_max_length, samples_length, tag_seqs_num

class PretrainDataset(Dataset):
    def __init__(self, args, mode=0):
        """ 
            Pretraining Joint Extraction Dataset.
            Args:
                args:       The arguments from command.
                mode:       The mode of data. 0 -> train, 1 -> valid.
                group:      The group setting.
            Returns:
        """
        self.mode = mode
        self.model_type = args.model_type
        self.data_name = args.data_name         # Name of dataset. Such as "NYT".
        self.data_type = args.data_type         # Type of dataset. "inter_data" or "intra_data".
        self.language = args.language           # The language. "en" or "ch".
        self.logger = args.logger
        self.group = args.group
        self.data_index = data_path_index[self.group]
        self.data_path = []
        self.data_path.append(data_path_temp[self.data_index[0]].replace("<name_template>", self.data_name).replace("<type_template>", self.data_type))
        self.data_path.append(data_path_temp[self.data_index[1]].replace("<name_template>", self.data_name).replace("<type_template>", self.data_type))
        self.tokenizer = BertTokenizerFast.from_pretrained(model_type_dict[self.language])
        self.samples, self.tag2id, self.id2tag, self.label2array = self.__load_data__()
        self.samples = self.__preprocess__(self.samples)
        self.length = len(self.samples)
        self.tag_seqs_num = tag_seqs_num[self.model_type]
        self.seq_max_length = seq_max_length
        self.label_max_length = label_max_length
        self.tagger = args.tagger
        self.datamaker = PreDataMaker[self.model_type](self.tokenizer, self.tagger, self.tag2id, self.label2array)

    def __load_data__(self):
        train_datas, valid_datas, classes = [], [], set()
        self.logger.info("Loading data from {}".format(self.data_path))
        for data_file in self.data_path:
            tmp_datas, count = [], 0
            with open(data_file, "r", encoding="utf-8") as f:
                for index, line in enumerate(f):
                    tmp_datas.append(json.loads(line.strip()))
                    count += 1
            train_datas += tmp_datas[:int(count*0.8)]
            valid_datas += tmp_datas[int(count*0.8):]
        train_datas_len = len(train_datas)
        new_datas, label2array = get_norm_data(train_datas + valid_datas, self.tokenizer, self.data_name)
        train_datas = new_datas[:train_datas_len]
        valid_datas = new_datas[train_datas_len:]
        classes = sorted(list(label2array.keys()))
        tag2id = {classes[i]:i for i in range(len(classes))}
        id2tag = {i:classes[i] for i in range(len(classes))}
        samples = valid_datas if self.mode else train_datas
        self.logger.info("Samples length after loading: {}".format(len(samples)))
        return samples, tag2id, id2tag, label2array

    def __preprocess__(self, samples):
        new_samples = []
        self.logger.info("Preprocessing datas...")
        for sample in tqdm(samples, desc="Preprocessing", total=len(samples)):
            for rel, rel_id in self.tag2id.items():
                new_sample = {
                    "tokens": sample["tokens"],
                    "relations": [relation for relation in sample["relations"] if relation["label"] == rel],
                    "label": rel
                }
                if self.mode:
                    # Valid.
                    new_samples.append(new_sample)
                else:
                    # Train. Resampling.
                    sample_rel_len = len(new_sample["relations"])
                    if sample_rel_len:
                        new_samples.extend([new_sample for _ in range(len(self.tag2id) - 1)])
                    else:
                        new_samples.append(new_sample)
        self.logger.info("Samples length after preprocessing: {}".format(len(new_samples)))
        return new_samples
    
    def __getitem__(self, index):
        index_data = self.datamaker.get_indexed_data(self.samples[index], self.mode, self.seq_max_length, self.label_max_length)
        dataset = {
            "src_ids": torch.LongTensor(index_data[0]),
            "seg_ids": torch.LongTensor(index_data[1]),
            "mask_ids": torch.LongTensor(index_data[2]),
            "tags": [torch.LongTensor(index_data[3][i]) for i in range(self.tag_seqs_num)],
            "rel_id": index_data[4],
            "sample": index_data[5]
        }
        return dataset

    def __len__(self):
        return self.length

def collate_fn(batch, model_type):
    tags_size = tag_seqs_num[model_type]
    new_batch = {"src_ids": [], "seg_ids": [], "mask_ids": [], "tags": [[] for i in range(tags_size)], "rel_id": [], "sample": []}
    for i in range(len(batch)):
        for k in new_batch.keys():
            if k == "tags":
                for t in range(tags_size):
                    new_batch[k][t].append(batch[i][k][t])
            else:
                new_batch[k].append(batch[i][k])
    for k in new_batch.keys():
        if k != "tags" and k != "rel_id" and k!= "sample":
            new_batch[k] = torch.stack(new_batch[k], 0)
    
    return new_batch

def get_loader_pre(args, mode=0):
    """ 
        The data loader for Pre-train Joint Extration dataset.
        Args:
            args:       The arguments from command.
            mode:       The mode of data. 0 -> train, 1 -> valid.
        Returns:
            data_loader
    """
    dataset = PretrainDataset(args, mode)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.pre_batch_size,
        shuffle=True,
        #pin_memory=True,
        #num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, model_type=args.model_type)
    )
    return data_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()

    args.data_name = "NYT"                  # Name of dataset. Such as "NYT".
    args.data_type = "inter_data"           # Type of dataset. "inter_data" or "intra_data".
    args.group = 0
    args.language = "en"                    # The language. "en" or "ch".
    args.pre_batch_size = 16
    #args.num_workers = 4
    args.model_type = "few-tplinker-plus"
    args.seq_max_length = seq_max_length
    args.label_max_length = label_max_length
    args.logger = loguru.logger
    args.tagger = TaggingScheme[args.model_type](args.seq_max_length - args.label_max_length - 2)
    
    data_loader = get_loader_pre(args, mode=0)
    
    for index, batch in enumerate(data_loader):
        if index < 1:
            print(batch)
        else:
            break