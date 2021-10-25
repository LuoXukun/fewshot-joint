#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.10.21

import os
import json
import torch
import argparse

from transformers import BertTokenizer

from torch.utils.data import Dataset, DataLoader
from config import data_path_temp, model_type_dict, max_length, samples_length
from utils.fewshotsampler import FewshotSampleBase, FewshotSampler

class Sample(FewshotSampleBase):
    def __init__(self, sample_json):
        """ 
            Build up the sample from sample json. 
            Args:
                sample_json:        A sample including text and relations.
                                    {
                                        "sentText": "Just as it was 20 years ago , Nicaragua now finds itself smack in the middle of the conflict with the election this week of Daniel Ortega , the former Marxist rebel leader , as president .", 
                                        "intraRelationMentions": [{"label": "/business/person/company", "em1Text": "Ortega", "em2Text": "Nicaragua"}]
                                    }
            Returns:
        """
        self.sample_json = sample_json
        self.relation_tags_list = []
        relation_key = "intraRelationMentions" if "intraRelationMentions" in sample_json.keys() else "interRelationMentions"
        for relation in sample_json[relation_key]:
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
        return self.relation_tags_set.intersection(set(target_classes)) \
            and not self.relation_tags_set.difference(set(target_classes))
    
    def __str__(self):
        return str(self.sample_json)

class FewshotJointDataset(Dataset):
    def __init__(self, args, mode=0):
        """ 
            Fewshot Joint Extraction Dataset.
            Args:
                args:       The arguments from command.
                mode:       The mode of data. 0 -> train, 1 -> valid, 2 -> test.
            Returns:
        """
        self.class2sampleid = {}
        self.N = args.N
        self.K = args.K
        self.Q = args.Q
        self.data_name = args.data_name         # Name of dataset. Such as "NYT".
        self.data_type = args.data_type         # Type of dataset. "inter_data" or "intra_data".
        self.language = args.language           # The language. "en" or "ch".
        self.data_path = data_path_temp[mode].replace("<name_template>", self.data_name).replace("<type_template>", self.data_type)
        self.tokenizer = BertTokenizer.from_pretrained(model_type_dict[self.language])
        self.samples, self.classes = self.__load_data__()
        self.max_length = max_length
        self.sampler = FewshotSampler(self.N, self.K, self.Q, self.samples, classes=self.classes)

    def __insert_sample__(self, index, sample_classes):
        for item in sample_classes:
            if item in self.class2sampleid:
                self.class2sampleid[item].append(index)
            else:
                self.class2sampleid[item] = [index]

    def __load_data__(self):
        """ Load data from the json file. """
        samples, classes = [], set()
        with open(self.data_path, "r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                sample_json = json.loads(line.strip())
                sample = Sample(sample_json)
                samples.append(sample)
                sample_classes = sample.get_tag_classes()
                self.__insert_sample__(index, list(sample_classes))
                classes = classes.union(sample_classes)
        classes = list(classes)
        return samples, classes

    def __get_data__(self, sample):
        pass

    def __additem__(self, index, data, src_ids, seg_ids, mask_ids, tags):
        data["index"].append(index)
        data["src_ids"] += src_ids
        data["seg_ids"] += seg_ids
        data["mask_ids"] += mask_ids
        # data["tags"] 
    
    def __populate__(self, indexs, savelabeldic=False):
        """ Populate samples into data dict. """
        dataset = {"index": [], "src_ids": [], "seg_ids": [], "mask_ids": [], "tags": [], "samples_num": []}
        for index in indexs:
            src_ids, seg_ids, mask_ids, tags = self.__get_data__(self.samples[index])
            src_ids = torch.LongTensor(src_ids)
            seg_ids = torch.LongTensor(seg_ids)
            mask_ids = torch.LongTensor(mask_ids)
            self.__additem__(index, dataset, src_ids, seg_ids, mask_ids, tags)
        dataset["samples_num"] = [len(dataset["src_ids"])]
        if savelabeldic:
            dataset["id2tag"] = [self.id2tag]
        return dataset
    
    def __getitem__(self, index):
        target_classes, support_idx, query_idx = self.sampler.__next__()
        # add 'O' and make sure 'O' is labeled 0
        distinct_tags = ["O"] + target_classes
        self.tag2id = {tag: idx for idx, tag in enumerate(distinct_tags)}
        self.id2tag = {idx: tag for idx, tag in enumerate(distinct_tags)}
        support_set = self.__populate__(support_idx)
        query_set = self.__populate__(query_idx, savelabeldic=True)
        return support_set, query_set
    
    def __len__(self):
        return samples_length

def collate_fn(batch):
    batch_support = {"src_ids": [], "seg_ids": [], "mask_ids": [], "tags": [], "samples_num": []}
    batch_query = {"src_ids": [], "seg_ids": [], "mask_ids": [], "tags": [], "samples_num": [], "id2tag": []}
    support_sets, query_sets = zip(*batch)
    for i in range(len(support_sets)):
        for k in batch_support.keys():
            batch_support[k] += support_sets[i][k]
        for k in batch_query.keys():
            batch_query[k] += query_sets[i][k]
    for k in batch_support.keys():
        if k != "tags" and k != "samples_num":
            batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query.keys():
        if k != "tags" and k != "samples_num" and k!= "id2tag":
            batch_query[k] = torch.stack(batch_query[k], 0)
    #batch_support['tags'] = [torch.tensor(tag_list).long() for tag_list in batch_support['tags']]
    #batch_query['tags'] = [torch.tensor(tag_list).long() for tag_list in batch_query['tags']]
    return batch_support, batch_query

def get_loader(args, mode=0):
    """ 
        The data loader for Fewshot Joint Extration dataset.
        Args:
            args:       The arguments from command.
            mode:       The mode of data. 0 -> train, 1 -> valid, 2 -> test.
        Returns:
            data_loader
    """
    dataset = FewshotJointDataset(args, mode)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    return iter(data_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()

    args.N = 2
    args.K = 2
    args.Q = 2
    args.data_name = "NYT"                  # Name of dataset. Such as "NYT".
    args.data_type = "inter_data"           # Type of dataset. "inter_data" or "intra_data".
    args.language = "en"                    # The language. "en" or "ch".
    args.batch_size = 2
    args.num_workers = 4
    
    
    for batch_support, batch_query in get_loader(args, mode=0):
        print(batch_support)
        print(batch_query)
        break