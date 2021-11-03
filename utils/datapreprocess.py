#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.10.26

import unicodedata
from tqdm import tqdm

def get_norm_data(data, tokenizer, data_name):
    """ 
        Get normal data based on tokenizer and data type.
        Args:
            tokenizer:          The tokenizer.
            data_name:          Datasets' name, including NYT, WebNLG and DuIE.
        Returns:
            new_data:           Data format:
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
    """
    new_data, label2array = [], {}
    
    for index, sample in tqdm(enumerate(data), desc = "Getting normalized data", total=len(data)):
        if data_name == "NYT" or data_name == "WebNLG":
            text = normalize_text(sample["sentText"]).strip()
            rel_list = sample["relationMentions"]
            sub_key, label_key, obj_key = "em1Text", "label", "em2Text"
        elif data_name == "DuIE":
            text = sample["text"].strip()
            rel_list = sample["spo_list"]
            sub_key, label_key, obj_key = "subject", "predicate", "object"
        
        tokens = tokenizer.tokenize(text)
        normal_sample = {"tokens": tokens, "relations": []}
        for rel in rel_list:
            label = rel[label_key]
            if data_name == "NYT":
                sub = tokenizer.tokenize(normalize_text(rel[sub_key]).strip())
                obj = tokenizer.tokenize(normalize_text(rel[obj_key]).strip())
                label_array = label.replace("_", "/").split("/")[1:]
            elif data_name == "WebNLG":
                sub = tokenizer.tokenize(normalize_text(rel[sub_key]).strip())
                obj = tokenizer.tokenize(normalize_text(rel[obj_key]).strip())
                label_array = label.split(" ")
            elif data_name == "DuIE":
                sub = tokenizer.tokenize(rel[sub_key].strip())
                obj = tokenizer.tokenize(rel[obj_key].strip())
                label_array = list(label)
            if label not in label2array.keys():
                label2array[label] = label_array
            sub_span = find_index(tokens, sub)
            obj_span = find_index(tokens, obj)
            relation = {"label": label, "label_array": label_array, "subject": sub, "object": obj, "sub_span": sub_span, "obj_span": obj_span}
            normal_sample["relations"].append(relation)
        
        new_data.append(normal_sample)
    
    return new_data, label2array

    
def normalize_text(text):
    """ 
        Normalize the unicode string.
        Args:
            text:           unicode string 
        Return:
    """
    return unicodedata.normalize('NFKD', text).encode('ascii','ignore').decode('utf-8')

def find_index(sen_split, word_split):
    """ 
        Find the loaction of entity in sentence.
        Args:
            sen_split:      the sentence array.
            word_split:     the entity array.
        Return:
            index1:         start index
            index2:         end index
    """
    index1 = -1
    index2 = -1
    for i in range(len(sen_split)):
        if str(sen_split[i]) == str(word_split[0]):
            flag = True
            k = i
            for j in range(len(word_split)):
                if word_split[j] != sen_split[k]:
                    flag = False
                if k < len(sen_split) - 1:
                    k += 1
            if flag:
                index1 = i
                index2 = i + len(word_split)
                break
    return index1, index2
        