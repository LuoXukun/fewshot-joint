#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.12.09

""" 
    Pretrain model for FewBiTT.
"""

import torch
import torch.nn as nn

from models.FewBiTT import FewBiTT, BidirectionalTreeTaggingScheme

class PreBiTT(FewBiTT):
    def __init__(self, args):
        super(PreBiTT, self).__init__(args)

        self.classify_fc = nn.ModuleList([
            nn.Linear(self.map_hidden_size, self.tags_num[i % self.parts_num]) for i in range(self.parts_num * 2)
        ])
    
    def forward(self, batch):
        # Embedding. (batch_size, seq_len, hidden_size)
        embedding = self.encoder(batch["src_ids"], batch["mask_ids"], batch["seg_ids"])[:, self.label_max_length + 2:, :]

        # Hidden. (parts_num * 2, batch_size, seq_len, map_hidden_size)
        hidden = self.dropout(embedding)
        hidden = [self.mapping_fc[i](embedding) for i in range(self.parts_num * 2)]

        # Feats. (parts_num * 2, batch_size, seq_len, tag_dim)
        feats = [self.classify_fc[i](hidden[i]) for i in range(self.parts_num * 2)]

        # Preds. (parts_num * 2, batch_size, seq_len)
        preds = [torch.max(feats[i], -1)[1] for i in range(self.parts_num * 2)]

        return feats, preds

class PreBiTTDataMaker():
    def __init__(self, tokenizer, tagger: BidirectionalTreeTaggingScheme, tag2id, label2array):
        """ DataMaker for Bidirectional Tree Tagging model. """
        self.tokenizer = tokenizer
        self.tagger = tagger
        self.rel2id = tag2id
        self.rel2array = label2array
    
    def get_indexed_data(self, data, mode, seq_max_len, label_max_len):
        """ 
            Get the indexed data.
            Args:
                data:               {
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
                                        ],
                                        "label": ""
                                    }
                mode:               The mode of data. 0 -> train, 1 -> valid.
                seq_max_len:        Max length of the src_ids.
                label_max_len:      Max length of label array.
            Returns:
                [(src_ids, seg_ids, mask_ids, tags, rel_id, sample)]
        """
        #print("Get indexed data...")
        indexed_data = []
        tokens, relations, label = data["tokens"], data["relations"], data["label"]

        rel_tokens = self.rel2array[label]
        rel_tokens_len = len(rel_tokens)

        if rel_tokens_len > label_max_len:
            rel_tokens = rel_tokens[0:label_max_len]
            rel_mask_index = label_max_len + 1
        else:
            rel_tokens = rel_tokens + ["[PAD]"] * (label_max_len - rel_tokens_len)
            rel_mask_index = rel_tokens_len + 1
    
        src_res = self.tokenizer(rel_tokens, tokens, is_split_into_words=True, truncation="only_second", max_length=seq_max_len, padding="max_length")
        src_ids, seg_ids, mask_ids = src_res["input_ids"], src_res["token_type_ids"], src_res["attention_mask"]
        mask_ids[rel_mask_index:label_max_len+1] = [0 for i in range(label_max_len - rel_mask_index + 1)]

        # Tagging.
        tags = self.tagger.encode_rel_to_bitt_tag(data)
        
        return src_ids, seg_ids, mask_ids, tags, self.rel2id[label], data
