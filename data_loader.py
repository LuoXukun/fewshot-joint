#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.10.21

import os
import json
import torch
import argparse

from tqdm import tqdm
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader

from models import DataMaker, TaggingScheme
from utils.datapreprocess import get_norm_data
from utils.fewshotsampler import FewshotSampleBase, FewshotSampler
from config import data_path_index, data_path_temp, model_type_dict, seq_max_length, label_max_length, samples_length, tag_seqs_num

class Sample(FewshotSampleBase):
    def __init__(self, sample_json):
        """ 
            Build up the sample from sample json. 
            Args:
                sample_json:        A sample including text and relations.
                                    {
                                        "tokens": ["a", "b", "c", ...],
                                        "relations": [
                                            {
                                                "label": "",
                                                "label_array": ["", "", "", ...],
                                                "subject": ["a"],
                                                "object": ["c"],
                                                "sub_span": (0, 1),
                                                "obj_span": (2, 3)
                                            }, ...
                                        ]
                                    }
            Returns:
        """
        self.sample_json = sample_json
        self.relation_tags_list = []
        for relation in sample_json["relations"]:
            if relation["label"] != "None":
                self.relation_tags_list.append(relation["label"])
        self.relation_tags_set = set(self.relation_tags_list)
        self.class_count = {}
    
    def __count_relations__(self):
        """ Count the number of classes in the sample. """
        for tag in self.relation_tags_list:
            if tag in self.class_count.keys():
                self.class_count[tag] += 1
            else:
                self.class_count[tag] = 1
    
    def get_class_count(self):
        """ Get the class_count dictionary. """
        if not self.class_count:
            self.__count_relations__()
        return self.class_count
    
    def get_tag_classes(self):
        """ Get the tag classes set. """
        return self.relation_tags_set
    
    def valid(self, target_classes):
        #return self.relation_tags_set.intersection(set(target_classes)) \
        #    and not self.relation_tags_set.difference(set(target_classes))
        return self.relation_tags_set.intersection(set(target_classes))
    
    def get_data(self):
        return self.sample_json
    
    def __str__(self):
        return str(self.sample_json)

class FewshotJointDataset(Dataset):
    def __init__(self, args, mode=0):
        """ 
            Fewshot Joint Extraction Dataset.
            Args:
                args:       The arguments from command.
                mode:       The mode of data. 0 -> train, 1 -> valid.
                group:      The group setting.
            Returns:
        """
        self.class2sampleid = {}
        self.N = args.N
        self.K = args.K
        self.Q = args.Q
        self.mode = mode
        self.model_type = args.model_type
        self.data_name = args.data_name         # Name of dataset. Such as "NYT".
        self.data_type = args.data_type         # Type of dataset. "inter_data" or "intra_data".
        self.language = args.language           # The language. "en" or "ch".
        self.logger = args.logger
        self.group = args.group
        self.data_index = data_path_index[self.group]
        self.data_path = []
        if self.mode == 0:
            self.data_path.append(data_path_temp[self.data_index[0]].replace("<name_template>", self.data_name).replace("<type_template>", self.data_type))
            self.data_path.append(data_path_temp[self.data_index[1]].replace("<name_template>", self.data_name).replace("<type_template>", self.data_type))
        else:
            self.data_path.append(data_path_temp[self.data_index[2]].replace("<name_template>", self.data_name).replace("<type_template>", self.data_type))
        self.tokenizer = BertTokenizerFast.from_pretrained(model_type_dict[self.language])
        self.samples, self.classes, self.label2array = self.__load_data__()
        self.tag_seqs_num = tag_seqs_num[self.model_type]
        self.seq_max_length = seq_max_length
        self.label_max_length = label_max_length
        self.sampler = FewshotSampler(self.N, self.K, self.Q, self.samples, classes=self.classes)
        self.tagger = args.tagger               # Tagging scheme.

    def __insert_sample__(self, index, sample_classes):
        for item in sample_classes:
            if item in self.class2sampleid:
                self.class2sampleid[item].append(index)
            else:
                self.class2sampleid[item] = [index]

    def __load_data__(self):
        """ Load data from the json file. """
        datas, samples, classes = [], [], set()
        self.logger.info("Loading data from {}".format(self.data_path))
        for data_file in self.data_path:
            with open(data_file, "r", encoding="utf-8") as f:
                for index, line in enumerate(f):
                    datas.append(json.loads(line.strip()))
        new_datas, label2array = get_norm_data(datas, self.tokenizer, self.data_name)
        for data in tqdm(new_datas, desc="Generating fewshot samples", total=len(new_datas)):
            sample = Sample(data)
            samples.append(sample)
            sample_classes = sample.get_tag_classes()
            self.__insert_sample__(index, list(sample_classes))
            classes = classes.union(sample_classes)
        classes = list(classes)
        return samples, classes, label2array

    def __additem__(self, index, data, src_ids, seg_ids, mask_ids, tags, rel_id, sample):
        data["index"].append(index)
        data["src_ids"].append(src_ids)
        data["seg_ids"].append(seg_ids)
        data["mask_ids"].append(mask_ids)
        for i in range(len(tags)):
            data["tags"][i].append(tags[i])
        data["rel_id"].append(rel_id)
        data["sample"].append(sample)
    
    def __populate__(self, indexs, savelabeldic=False):
        """ Populate samples into data dict. """
        #print("Populate: {}".format(indexs))
        dataset = {"index": [], "src_ids": [], "seg_ids": [], "mask_ids": [], "tags": [[] for i in range(self.tag_seqs_num)], "rel_id": [], "samples_num": [], "sample": []}
        for index in indexs:
            indexed_data = self.datamaker.get_indexed_data(self.samples[index].get_data(), self.mode, self.seq_max_length, self.label_max_length)
            for (src_ids, seg_ids, mask_ids, tags, rel_id, sample) in indexed_data:
                src_ids = torch.LongTensor(src_ids)
                seg_ids = torch.LongTensor(seg_ids)
                mask_ids = torch.LongTensor(mask_ids)
                tags = [torch.LongTensor(tags[i]) for i in range(self.tag_seqs_num)]
                self.__additem__(index, dataset, src_ids, seg_ids, mask_ids, tags, rel_id, sample)
        dataset["samples_num"] = [len(dataset["src_ids"])]
        if savelabeldic:
            dataset["id2tag"] = [self.id2tag]
        return dataset
    
    def __getitem__(self, index):
        #print("Getting item...")
        target_classes, support_idx, query_idx = self.sampler.__next__()
        self.tag2id = {tag: idx for idx, tag in enumerate(target_classes)}
        self.id2tag = {idx: tag for idx, tag in enumerate(target_classes)}
        self.datamaker = DataMaker[self.model_type](self.tokenizer, self.tagger, self.tag2id, self.label2array)
        support_set = self.__populate__(support_idx)
        query_set = self.__populate__(query_idx, savelabeldic=True)
        return support_set, query_set
    
    def __len__(self):
        return samples_length

def collate_fn(batch, model_type):
    #print("Collate_fn...")
    tags_size = tag_seqs_num[model_type]
    batch_support = {"src_ids": [], "seg_ids": [], "mask_ids": [], "tags": [[] for i in range(tags_size)], "rel_id": [], "samples_num": [], "sample": []}
    batch_query = {"src_ids": [], "seg_ids": [], "mask_ids": [], "tags": [[] for i in range(tags_size)], "rel_id": [], "samples_num": [], "id2tag": [], "sample": []}
    support_sets, query_sets = zip(*batch)
    for i in range(len(support_sets)):
        for k in batch_support.keys():
            if k == "tags":
                for t in range(tags_size):
                    batch_support[k][t] += support_sets[i][k][t]
            else:
                batch_support[k] += support_sets[i][k]
        for k in batch_query.keys():
            if k == "tags":
                for t in range(tags_size):
                    batch_query[k][t] += query_sets[i][k][t]
            else:
                batch_query[k] += query_sets[i][k]
    for k in batch_support.keys():
        if k != "tags" and k != "samples_num" and k != "rel_id" and k!= "sample":
            batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query.keys():
        if k != "tags" and k != "samples_num" and k!= "id2tag" and k != "rel_id" and k != "sample":
            batch_query[k] = torch.stack(batch_query[k], 0)
    #batch_support['tags'] = [torch.tensor(tag_list).long() for tag_list in batch_support['tags']]
    #batch_query['tags'] = [torch.tensor(tag_list).long() for tag_list in batch_query['tags']]
    return batch_support, batch_query

def get_loader(args, mode=0):
    """ 
        The data loader for Fewshot Joint Extration dataset.
        Args:
            args:       The arguments from command.
            mode:       The mode of data. 0 -> train, 1 -> valid.
        Returns:
            data_loader
    """
    dataset = FewshotJointDataset(args, mode)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        #pin_memory=True,
        #num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, model_type=args.model_type)
    )
    return iter(data_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()

    args.N = 5
    args.K = 1
    args.Q = 1
    args.data_name = "NYT"                  # Name of dataset. Such as "NYT".
    args.data_type = "inter_data"           # Type of dataset. "inter_data" or "intra_data".
    args.group = 0
    args.language = "en"                    # The language. "en" or "ch".
    args.batch_size = 1
    #args.num_workers = 4
    args.model_type = "few-tplinker"
    args.seq_max_length = seq_max_length
    args.label_max_length = label_max_length
    args.tagger = TaggingScheme[args.model_type](args.seq_max_length - args.label_max_length - 2)
    
    data_loader = get_loader(args, mode=0)
    
    for it in range(0, 100):
        print(it)
        batch_support, batch_query = next(data_loader)
        #print("Support: ", batch_support)
        #print("Query: ", batch_query)
        #break