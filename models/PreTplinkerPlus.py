#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.11.29

""" 
    Pretrain model for FewTplinkerPlus.
"""

import torch
import torch.nn as nn
import numpy as np

from torch.nn.parameter import Parameter

from models.FewTplinkerPlus import FewTPLinkerPlus
from models.FewTplinker import HandshakingTaggingScheme

class PreTPlinkerPlus(FewTPLinkerPlus):
    def __init__(self, args):
        super(PreTPlinkerPlus, self).__init__(args)
        
        self.ent_fc = nn.Linear(self.map_hidden_size * 2, 2)
        self.head_rel_fc = nn.Linear(self.map_hidden_size * 2, 2)
        self.tail_rel_fc = nn.Linear(self.map_hidden_size * 2, 2)

        self.cost_weight = torch.Tensor([1.0, 10.0])
        self.pre_cost = nn.CrossEntropyLoss(weight=self.cost_weight)

    def forward(self, batch):
        # Embedding. (batch_size, seq_len, hidden_size)
        embedding = self.encoder(batch["src_ids"], batch["mask_ids"], batch["seg_ids"])[:, self.label_max_length + 2:, :]

        # Hidden. (2, batch_size, seq_len, map_hidden_size)
        ent_hidden = torch.stack([self.ent_map_fc[i](self.dropout(embedding)) for i in range(2)])               # The head and tail ent matrix for handshaking.
        head_rel_hidden = torch.stack([self.head_rel_map_fc[i](self.dropout(embedding)) for i in range(2)])     # The head rel matrix for handshaking.
        tail_rel_hidden = torch.stack([self.tail_rel_map_fc[i](self.dropout(embedding)) for i in range(2)])     # The head rel matrix for handshaking.

        # Handshaking. (batch_size, seq_len * seq_len, map_hidden_size * 2)
        batch_size, seq_len = ent_hidden.size(1), ent_hidden.size(2)

        ent_hidden_0 = ent_hidden[0].view(batch_size, seq_len, 1, -1).repeat(1, 1, seq_len, 1)
        ent_hidden_1 = ent_hidden[1].view(batch_size, 1, seq_len, -1).repeat(1, seq_len, 1, 1)
        ent_hidden_cat = torch.cat([ent_hidden_0, ent_hidden_1], -1).view(batch_size, seq_len * seq_len, -1)
        #ent_hidden_cat = torch.tanh(torch.cat([ent_hidden_0, ent_hidden_1], -1).view(batch_size, seq_len * seq_len, -1))

        head_rel_hidden_0 = head_rel_hidden[0].view(batch_size, seq_len, 1, -1).repeat(1, 1, seq_len, 1)
        head_rel_hidden_1 = head_rel_hidden[1].view(batch_size, 1, seq_len, -1).repeat(1, seq_len, 1, 1)
        head_rel_hidden_cat = torch.cat([head_rel_hidden_0, head_rel_hidden_1], -1).view(batch_size, seq_len * seq_len, -1)
        #head_rel_hidden_cat = torch.tanh(torch.cat([head_rel_hidden_0, head_rel_hidden_1], -1).view(batch_size, seq_len * seq_len, -1))

        tail_rel_hidden_0 = tail_rel_hidden[0].view(batch_size, seq_len, 1, -1).repeat(1, 1, seq_len, 1)
        tail_rel_hidden_1 = tail_rel_hidden[1].view(batch_size, 1, seq_len, -1).repeat(1, seq_len, 1, 1)
        tail_rel_hidden_cat = torch.cat([tail_rel_hidden_0, tail_rel_hidden_1], -1).view(batch_size, seq_len * seq_len, -1)
        #tail_rel_hidden_cat = torch.tanh(torch.cat([tail_rel_hidden_0, tail_rel_hidden_1], -1).view(batch_size, seq_len * seq_len, -1))
        
        # Feats. (batch_size, seq_len * seq_len, 2)
        ent_feats = self.ent_fc(ent_hidden_cat)
        head_rel_feats = self.head_rel_fc(head_rel_hidden_cat)
        tail_rel_feats = self.tail_rel_fc(head_rel_hidden_cat)

        # preds. (batch_size, seq_len * seq_len)
        _, ent_preds = torch.max(ent_feats, -1)
        _, head_rel_preds = torch.max(head_rel_feats, -1)
        _, tail_rel_preds = torch.max(tail_rel_feats, -1)

        # Result. (3, batch_size, seq_len * seq_len, 2), (3, batch_size, seq_len * seq_len)
        return [ent_feats, head_rel_feats, tail_rel_feats], [ent_preds, head_rel_preds, tail_rel_preds]
    
    def loss(self, logits, label, quiet=True):
        '''
            Args:

                logits:     Logits with the size (2, batch_size, seq_len x seq_len, class_num)
                label:      Label with the size (2, batch_size, seq_len x seq_len).

            Returns: 

                [Loss] (A single value)
        '''
        loss = []
        for i in range(3):
            N = logits[i].size(-1)
            loss.append(self.pre_cost(logits[i].view(-1, N), label[i].view(-1)))
        if not quiet:
            print("ent_loss: {}, head_rel_loss: {}, tail_rel_loss: {}".format(loss[0], loss[1], loss[2]))
        loss = loss[0] + loss[1] + loss[2]
        return loss

class PreTPLinkerDataMaker():
    def __init__(self, tokenizer, tagger: HandshakingTaggingScheme, tag2id, label2array):
        """ DataMaker for TPLinker. """
        self.tokenizer = tokenizer
        self.handshaking_tagger = tagger
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
                mode:               The mode of data. 0 -> train, 1 -> valid, 2 -> test.
                seq_max_len:        Max length of the src_ids.
                label_max_len:      Max length of label array.
            Returns:
                [(src_ids, seg_ids, mask_ids, tags, rel_id, sample)]
        """
        #print("Get indexed data...")
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
        mask_ids[rel_mask_index:label_max_len+1] = [0 for i in range(label_max_len - rel_mask_index + 1)]        # rel padding should not be calculated when producing attention.

        # Tagging.
        ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = self.handshaking_tagger.get_spots(data)
        ent_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag(ent_matrix_spots)
        head_rel_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag(head_rel_matrix_spots)
        tail_rel_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag(tail_rel_matrix_spots)
        tags = [ent_shaking_tag, head_rel_shaking_tag, tail_rel_shaking_tag]
        #tags = [ent_shaking_tag, head_rel_shaking_tag]

        return src_ids, seg_ids, mask_ids, tags, self.rel2id[label], data