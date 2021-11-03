#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.10.26

""" 
    Few-shot model based on the TpLinker scheme. 
    Reference:  TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking
    Link:       https://github.com/131250208/TPlinker-joint-extraction
"""

import re
import copy
import json
import math
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.nn.parameter import Parameter


""" Data Preprocess. """

class HandshakingTaggingScheme(object):
    """docstring for HandshakingTaggingScheme"""
    def __init__(self, max_seq_len):
        super(HandshakingTaggingScheme, self).__init__()

        self.tag2id_ent = {
            "O": 0,
            "ENT-H2T": 1, # entity head to entity tail
        }
        self.id2tag_ent = {id_:tag for tag, id_ in self.tag2id_ent.items()}

        self.tag2id_head_rel = {
            "O": 0,
            "REL-SH2OH": 1, # subject head to object head
            "REL-OH2SH": 2, # object head to subject head
        }
        self.id2tag_head_rel = {id_:tag for tag, id_ in self.tag2id_head_rel.items()}

        self.tag2id_tail_rel = {
            "O": 0,    
            "REL-ST2OT": 1, # subject tail to object tail
            "REL-OT2ST": 2, # object tail to subject tail
        }
        self.id2tag_tail_rel = {id_:tag for tag, id_ in self.tag2id_tail_rel.items()}

        # mapping shaking sequence and matrix
        self.matrix_size = max_seq_len
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.shaking_ind2matrix_ind = [(ind, end_ind) for ind in range(self.matrix_size) for end_ind in list(range(self.matrix_size))[ind:]]
        # We do not cut the tagging sequence. e.g. [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        #self.shaking_ind2matrix_ind = [(ind, end_ind) for ind in range(self.matrix_size) for end_ind in range(self.matrix_size)]

        self.matrix_ind2shaking_ind = [[0 for i in range(self.matrix_size)] for j in range(self.matrix_size)]
        for shaking_ind, matrix_ind in enumerate(self.shaking_ind2matrix_ind):
            self.matrix_ind2shaking_ind[matrix_ind[0]][matrix_ind[1]] = shaking_ind

    def get_spots(self, sample):
        '''
        sample:                     {
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
        entity spot and tail_rel spot: (span_pos1, span_pos2, tag_id)
        head_rel spot: (rel_id, span_pos1, span_pos2, tag_id)
        '''
        ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = [], [], [] 

        for rel in sample["relations"]:
            subj_tok_span = rel["sub_span"]
            obj_tok_span = rel["obj_span"]
            ent_matrix_spots.append((subj_tok_span[0], subj_tok_span[1] - 1, self.tag2id_ent["ENT-H2T"]))
            ent_matrix_spots.append((obj_tok_span[0], obj_tok_span[1] - 1, self.tag2id_ent["ENT-H2T"]))

            if  subj_tok_span[0] <= obj_tok_span[0]:
                head_rel_matrix_spots.append((subj_tok_span[0], obj_tok_span[0], self.tag2id_head_rel["REL-SH2OH"]))
            else:
                head_rel_matrix_spots.append((obj_tok_span[0], subj_tok_span[0], self.tag2id_head_rel["REL-OH2SH"]))
                
            if subj_tok_span[1] <= obj_tok_span[1]:
                tail_rel_matrix_spots.append((subj_tok_span[1] - 1, obj_tok_span[1] - 1, self.tag2id_tail_rel["REL-ST2OT"]))
            else:
                tail_rel_matrix_spots.append((obj_tok_span[1] - 1, subj_tok_span[1] - 1, self.tag2id_tail_rel["REL-OT2ST"]))
                
        return ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots

    def sharing_spots2shaking_tag(self, spots):
        '''
        convert spots to shaking seq tag
        spots: [(start_ind, end_ind, tag_id), ], for entity and rel
        return: 
            shake_seq_tag: (shaking_seq_len, )
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_seq_tag = torch.zeros(shaking_seq_len).long()
        for sp in spots:
            if sp[1] < self.matrix_size:
                shaking_ind = self.matrix_ind2shaking_ind[sp[0]][sp[1]]
                shaking_seq_tag[shaking_ind] = sp[2]
        return shaking_seq_tag

    def get_sharing_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, )
        spots: [(start_ind, end_ind, tag_id), ]
        '''
        spots = []

        for shaking_ind in shaking_tag.nonzero():
            shaking_ind_ = shaking_ind[0].item()
            tag_id = shaking_tag[shaking_ind_]
            matrix_inds = self.shaking_ind2matrix_ind[shaking_ind_]
            spot = (matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots


    def decode_rel_fr_shaking_tag(self,
                      tokens, 
                      ent_shaking_tag, 
                      head_rel_shaking_tag, 
                      tail_rel_shaking_tag):
        '''
        ent shaking tag: (shaking_seq_len, )
        head rel and tail rel shaking_tag: size = (shaking_seq_len, )
        '''
        rel_list = []
        
        ent_matrix_spots = self.get_sharing_spots_fr_shaking_tag(ent_shaking_tag)
        head_rel_matrix_spots = self.get_sharing_spots_fr_shaking_tag(head_rel_shaking_tag)
        tail_rel_matrix_spots = self.get_sharing_spots_fr_shaking_tag(tail_rel_shaking_tag)

        # entity
        head_ind2entities = {}
        for sp in ent_matrix_spots:
            tag_id = sp[2]
            if tag_id != self.tag2id_ent["ENT-H2T"] or sp[1] >= len(tokens):
                continue

            ent_tokens = tokens[sp[0]:sp[1] + 1]
            
            head_key = sp[0] # take head as the key to entity list start with the head token
            if head_key not in head_ind2entities:
                head_ind2entities[head_key] = []
            head_ind2entities[head_key].append({
                "tokens": ent_tokens,
                "tok_span": [sp[0], sp[1] + 1]
            })
            
        # tail relation
        tail_rel_memory_set = set()
        for sp in tail_rel_matrix_spots:
            tag_id = sp[2]
            if tag_id == self.tag2id_tail_rel["REL-ST2OT"]:
                tail_rel_memory = "{}-{}".format(sp[0], sp[1])
                tail_rel_memory_set.add(tail_rel_memory)
            elif tag_id == self.tag2id_tail_rel["REL-OT2ST"]:
                tail_rel_memory = "{}-{}".format(sp[1], sp[0])
                tail_rel_memory_set.add(tail_rel_memory)

        # head relation
        for sp in head_rel_matrix_spots:
            tag_id = sp[2]
            
            if tag_id == self.tag2id_head_rel["REL-SH2OH"]:
                subj_head_key, obj_head_key = sp[0], sp[1]
            elif tag_id == self.tag2id_head_rel["REL-OH2SH"]:
                subj_head_key, obj_head_key = sp[1], sp[0]
                
            if subj_head_key not in head_ind2entities or obj_head_key not in head_ind2entities:
                # no entity start with subj_head_key and obj_head_key
                continue
            subj_list = head_ind2entities[subj_head_key] # all entities start with this subject head
            obj_list = head_ind2entities[obj_head_key] # all entities start with this object head

            # go over all subj-obj pair to check whether the relation exists
            for subj in subj_list:
                for obj in obj_list:
                    tail_rel_memory = "{}-{}".format(subj["tok_span"][1] - 1, obj["tok_span"][1] - 1)
                    if tail_rel_memory not in tail_rel_memory_set:
                        # no such relation 
                        continue
                    
                    rel_list.append({
                        "subject": subj["tokens"],
                        "object": obj["tokens"],
                        "sub_span": (subj["tok_span"][0], subj["tok_span"][1]),
                        "obj_span": (obj["tok_span"][0], obj["tok_span"][1])
                    })
        return rel_list

class TPLinkerDataMaker():
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
                                        ]
                                    }
                mode:               The mode of data. 0 -> train, 1 -> valid, 2 -> test.
                seq_max_len:        Max length of the src_ids.
                label_max_len:      Max length of label array.
            Returns:
                [(src_ids, seg_ids, mask_ids, tags, rel_id, sample)]
        """
        #print("Get indexed data...")
        indexed_data = []
        tokens, relations = data["tokens"], data["relations"]

        for rel, rel_id in self.rel2id.items():
            rel_tokens = self.rel2array[rel]
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
            sample = {
                "tokens": tokens,
                "relations": [relation for relation in relations if relation["label"] == rel]
            }
            ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = self.handshaking_tagger.get_spots(sample)
            ent_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag(ent_matrix_spots)
            head_rel_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag(head_rel_matrix_spots)
            tail_rel_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag(tail_rel_matrix_spots)
            tags = [ent_shaking_tag, head_rel_shaking_tag, tail_rel_shaking_tag]

            indexed_data.append((src_ids, seg_ids, mask_ids, tags, rel_id, sample))
        
        return indexed_data


""" Model. """

class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim = 0, center = True, scale = True, epsilon = None, conditional = False,
                 hidden_units = None, hidden_activation = 'linear', hidden_initializer = 'xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        # self.hidden_activation = activations.get(hidden_activation) keras中activation为linear时，返回原tensor,unchanged
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features = self.cond_dim, out_features = self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features = self.cond_dim, out_features = input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features = self.cond_dim, out_features = input_dim, bias=False)

        self.initialize_weights()


    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)
            # 下面这两个为什么都初始化为0呢? 
            # 为了防止扰乱原来的预训练权重，两个变换矩阵可以全零初始化（单层神经网络可以用全零初始化，连续的多层神经网络才不应当用全零初始化），这样在初始状态，模型依然保持跟原来的预训练模型一致。
            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)


    def forward(self, inputs, cond=None):
        """
            如果是条件Layer Norm，则cond不是None
        """
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            # for _ in range(K.ndim(inputs) - K.ndim(cond)): # K.ndim: 以整数形式返回张量中的轴数。
            # TODO: 这两个为什么有轴数差呢？ 为什么在 dim=1 上增加维度??
            # 为了保持维度一致，cond可以是（batch_size, cond_dim）
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)
            
            # cond在加入beta和gamma之前做一次线性变换，以保证与input维度一致
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs**2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) **2
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

class HandshakingKernel(nn.Module):
    def __init__(self, hidden_size, shaking_type, inner_enc_type):
        super().__init__()
        self.shaking_type = shaking_type
        if shaking_type == "cat":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "cat_plus":
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)
        elif shaking_type == "cln":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional = True)
        elif shaking_type == "cln_plus":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional = True)
            self.inner_context_cln = LayerNorm(hidden_size, hidden_size, conditional = True)
            
        self.inner_enc_type = inner_enc_type
        if inner_enc_type == "mix_pooling":
            self.lamtha = Parameter(torch.rand(hidden_size))
        elif inner_enc_type == "lstm":
            self.inner_context_lstm = nn.LSTM(hidden_size, 
                           hidden_size, 
                           num_layers = 1, 
                           bidirectional = False, 
                           batch_first = True)
     
    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type = "lstm"):
        # seq_hiddens: (batch_size, seq_len, hidden_size)
        def pool(seqence, pooling_type):
            if pooling_type == "mean_pooling":
                pooling = torch.mean(seqence, dim = -2)
            elif pooling_type == "max_pooling":
                pooling, _ = torch.max(seqence, dim = -2)
            elif pooling_type == "mix_pooling":
                pooling = self.lamtha * torch.mean(seqence, dim = -2) + (1 - self.lamtha) * torch.max(seqence, dim = -2)[0]
            return pooling
        if "pooling" in inner_enc_type:
            inner_context = torch.stack([pool(seq_hiddens[:, :i+1, :], inner_enc_type) for i in range(seq_hiddens.size()[1])], dim = 1)
        elif inner_enc_type == "lstm":
            inner_context, _ = self.inner_context_lstm(seq_hiddens)
            
        return inner_context
    
    def forward(self, seq_hiddens):
        '''
        seq_hiddens: (batch_size, seq_len, hidden_size)
        return:
            shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        '''
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]
            visible_hiddens = seq_hiddens[:, ind:, :] # ind: only look back
            repeat_hiddens = hidden_each_step[:, None, :].repeat(1, seq_len - ind, 1)  
            
            if self.shaking_type == "cat":
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim = -1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cat_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens, inner_context], dim = -1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cln":
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
            elif self.shaking_type == "cln_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
                shaking_hiddens = self.inner_context_cln(shaking_hiddens, inner_context)

            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim = 1)
        return long_shaking_hiddens

class FewTPLinker(nn.Module):
    def __init__(self, args):
        """ 
            Fewshot version for TPLinker model.

            Args:
                encoder:            A Bert Encoder.
                shaking_type:       The type of handshaking procedure. ["cat", "cat_plus", "cln", "cln_plus"]
                inner_enc_type:     The encoder type for generating inner context. ["lstm", "max_pooling", "mean_pooling", "mix_pooling"]
                map_hidden_size:    The hidden size of mapping layer.
                dist_type:          The type of calculating disance. ["dot", "euclidean"]
                label_max_length:   The max length of label inference.
        """
        super(FewTPLinker, self).__init__()
        self.encoder = args.encoder
        self.hidden_size = self.encoder.config.hidden_size
        self.shaking_type = args.shaking_type
        self.inner_enc_type = args.inner_enc_type
        self.map_hidden_size = args.map_hidden_size
        self.dist_type = args.dist_type
        self.label_max_length = args.label_max_length
        self.split_size = args.split_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Handshaking Kernel
        self.handshaking_kernel = HandshakingKernel(self.hidden_size, self.shaking_type, self.inner_enc_type)

        # Mapping Layer
        self.ent_map_fc = nn.Linear(self.hidden_size, self.map_hidden_size)
        self.head_rel_map_fc = nn.Linear(self.hidden_size, self.map_hidden_size)
        self.tail_rel_map_fc = nn.Linear(self.hidden_size, self.map_hidden_size)

        # Drop out
        self.dropout = nn.Dropout()

        self.cost = nn.CrossEntropyLoss()
    
    def forward(self, support, query):
        # Support
        #print("src_ids: ", support["src_ids"].size())
        support_emb = self.encoder(support["src_ids"], support["mask_ids"], support["seg_ids"])[:, self.label_max_length + 2:, :]
        #print("support_emb: ", support_emb.size())
        support_hidden = self.dropout(self.handshaking_kernel(support_emb))
        #print("support_hidden: ", support_hidden.size())
        ent_support_hidden = self.dropout(torch.tanh(self.ent_map_fc(support_hidden)))
        head_rel_support_hidden = self.dropout(torch.tanh(self.head_rel_map_fc(support_hidden)))
        tail_rel_support_hidden = self.dropout(torch.tanh(self.tail_rel_map_fc(support_hidden)))

        # Query
        query_emb = self.encoder(query["src_ids"], query["mask_ids"], query["seg_ids"])[:, self.label_max_length + 2:, :]
        query_hidden = self.dropout(self.handshaking_kernel(query_emb))
        ent_query_hidden = self.dropout(torch.tanh(self.ent_map_fc(query_hidden)))
        head_rel_query_hidden = self.dropout(torch.tanh(self.head_rel_map_fc(query_hidden)))
        tail_rel_query_hidden = self.dropout(torch.tanh(self.tail_rel_map_fc(query_hidden)))

        logits = [[] for i in range(3)] # ent, head_rel, tail_rel
        pred = [[] for i in range(3)]
        current_support_num = 0
        current_query_num = 0
        
        for index, support_samples_num in enumerate(support["samples_num"]):
            query_samples_num = query["samples_num"][index]
            #print("query_samples_num: ", query_samples_num)
            # Calculate nearest distance to each tokens pair in each class in support set.
            #print("ent")
            logits[0].append(   # ent
                self.__get_nearest_dist__(
                    ent_support_hidden[current_support_num:current_support_num+support_samples_num],
                    support["tags"][0][current_support_num:current_support_num+support_samples_num],
                    ent_query_hidden[current_query_num:current_query_num+query_samples_num],
                    max_tag = 1
                )
            )
            #print("head_rel")
            logits[1].append(   # head_rel
                self.__get_nearest_dist__(
                    head_rel_support_hidden[current_support_num:current_support_num+support_samples_num],
                    support["tags"][1][current_support_num:current_support_num+support_samples_num],
                    head_rel_query_hidden[current_query_num:current_query_num+query_samples_num],
                    max_tag = 2
                )
            )
            #print("tail_rel")
            logits[2].append(   # tail_rel
                self.__get_nearest_dist__(
                    tail_rel_support_hidden[current_support_num:current_support_num+support_samples_num],
                    support["tags"][2][current_support_num:current_support_num+support_samples_num],
                    tail_rel_query_hidden[current_query_num:current_query_num+query_samples_num],
                    max_tag = 2
                )
            )
            current_query_num += query_samples_num
            current_support_num += support_samples_num
        
        #print("logits: {}, {}, {}".format(logits[0][0].size(), logits[1][0].size(), logits[2][0].size()))
        for i in range(3):
            logits[i] = torch.cat(logits[i], 0)
            _, pred[i] = torch.max(logits[i], -1)
        
        return logits, pred

    def __dist__(self, x, y, dim):
        if self.dist_type == "dot":
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)
    
    def __batch_dist__(self, S, Q):
        # S [class, embed_dim], Q [num_of_sent, num_of_tokens, embed_dim]
        Q = Q.view(-1, Q.size(-1))
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)
    
    def __get_nearest_dist__(self, embedding, tag, query, max_tag):
        nearest_dist = []
        S = embedding.view(-1, embedding.size(-1))
        tag = tag.view(-1)
        #print("tag_size: {}, S_size: {}, Q_size: {}".format(tag.size(), S.size(), query.size()))
        assert tag.size(0) == S.size(0)
        support_token_len, query_num, query_len = embedding.shape[0], query.shape[0], query.shape[1]
        #dist = self.__batch_dist__(S, query) # [num_of_query_tokens, num_of_support_tokens]
        #for label in range(torch.max(tag)+1):
        #    nearest_dist.append(torch.max(dist[:,tag==label], 1)[0])
        #nearest_dist = torch.stack(nearest_dist, dim=1) # [num_of_query_tokens, class_num]
        """ Enumerate query and support to avoid error 'CUDA out of memory'. """
        """ for i in range(query_num):
            nearest_dist_q = []
            for j in range(support_num):
                dist = self.__batch_dist__(S[j], query[i])
                temp = []
                for label in range(max_tag + 1):
                    dist_valid = dist[:,tag[j]==label]
                    #print("dist_valid:", dist_valid.size())
                    if dist_valid.size(1) == 0:
                        temp.append(torch.ones(dist_valid.size(0)).to(self.device) * -10000.0)
                    else:
                        temp.append(torch.max(dist_valid, dim=1)[0])
                #temp = [torch.max(dist[:,tag[j]==label], dim=1)[0] for label in range(max_tag + 1)]
                nearest_dist_q.append(torch.stack(temp, dim=1))
                #del temp; del dist
            nearest_dist_q = torch.max(torch.stack(nearest_dist_q, dim=0), dim=0)[0]
            nearest_dist.append(nearest_dist_q)
            #print("nearest_dist_q: ", nearest_dist_q.size())
        nearest_dist = torch.stack(nearest_dist, dim=0)
        #print("nearest_dist: ", nearest_dist.size())
        return nearest_dist.view(query_num, -1, max_tag + 1) """
        for i in range(query_num):
            nearest_dist_q = [[torch.ones(query_len).to(self.device) * -10000.0] for label in range(max_tag + 1)]
            support_split_index = 0
            for start_index in range(0, support_token_len, self.split_size):
                end_index = min(start_index + self.split_size, support_token_len)
                dist = self.__batch_dist__(S[start_index:end_index, :], query[i])
                #print("dist: ", dist.size())
                for label in range(max_tag + 1):
                    dist_valid = dist[:,tag[start_index:end_index]==label]
                    #print("dist_valid: ", dist_valid.size())
                    if dist_valid.size(1) == 0:
                        nearest_dist_q[label].append(torch.ones(query_len).to(self.device) * -10000.0)
                    else:
                        nearest_dist_q[label].append(torch.max(dist_valid, dim=1)[0])
                    nearest_dist_q[label] = [torch.max(torch.stack(nearest_dist_q[label], dim=0), dim=0)[0]]
            for label in range(max_tag + 1):
                assert len(nearest_dist_q[label]) == 1
                nearest_dist_q[label] = nearest_dist_q[label][0]
            #print("nearest_dist_q: ", nearest_dist_q[0].size())
            nearest_dist.append(torch.stack(nearest_dist_q, dim=1))
        nearest_dist = torch.stack(nearest_dist, dim=0)
        #print("nearest_dist: ", nearest_dist.size())
        return nearest_dist.view(query_num, -1, max_tag + 1)
    
    def loss(self, logits, label):
        '''
            Args:

                logits:     Logits with the size (3, query_num, seq_len, class_num)
                label:      Label with the size (3, query_num, seq_len).

            Returns: 

                [Loss] (A single value)
        '''
        # 此处无法使用mask_ids来去掉[PAD]得到的logits，因为logits形状变了，
        # 但由于attention计算使用了mask，会使那一部分的参数不更新
        loss = []
        for i in range(3):
            N = logits[i].size(-1)
            loss.append(self.cost(logits[i].view(-1, N), label[i].view(-1)))
        loss = loss[0] + loss[1] + loss[2]
        return loss

class TPLinkerMetricsCalculator():
    def __init__(self, handshaking_tagger: HandshakingTaggingScheme):
        self.handshaking_tagger = handshaking_tagger
        self.accs_keys = ["ent_sample_acc", "head_rel_sample_acc", "tail_rel_sample_acc"]
    
    def get_accs(self, preds, labels):
        '''
            The accuracy of all pred labels of a sample are right.

            Args:

                preds:      Preds with the size (3, query_num, seq_len).
                label:      Label with the size (3, query_num, seq_len).

            Returns: 

                [accs] (A dict)
        '''
        accs = {}

        for index, key in enumerate(self.accs_keys):
            pred_ids, label_ids = preds[index], labels[index]

            correct_tag_num = torch.sum(torch.eq(label_ids, pred_ids).float(), dim=1)
            
            # correct_tag_num == seq_len.
            sample_acc_ = torch.eq(correct_tag_num, torch.ones_like(correct_tag_num) * label_ids.size()[-1]).float()
            sample_acc = torch.mean(sample_acc_)
            
            accs[self.accs_keys[index]] = sample_acc
        
        return accs
    
    def get_rel_pgc(self, samples, preds):
        '''
            Get pred, gold and correct.

            Args:

                samples:    Sample dict.
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
                preds:      Preds with the size (3, query_num, seq_len).

            Returns: 

                (pred, gold, correct)
        '''
        correct_num, pred_num, gold_num = 0, 0, 0

        for index in range(len(samples)):
            sample = samples[index]
            tokens = sample["tokens"]
            pred_ent_shaking_tag = preds[0][index]
            pred_head_rel_shaking_tag = preds[1][index]
            pred_tail_rel_shaking_tag = preds[2][index]

            pred_rel_list = self.handshaking_tagger.decode_rel_fr_shaking_tag(
                tokens, pred_ent_shaking_tag, pred_head_rel_shaking_tag, pred_tail_rel_shaking_tag
            )

            gold_rel_list = sample["relations"]

            pred_rel_set = set([
                "{}, {}, {}, {}".format(
                    rel["sub_span"][0], rel["sub_span"][1], rel["obj_span"][0], rel["obj_span"][1]
                ) for rel in pred_rel_list
            ])
            gold_rel_set = set([
                "{}, {}, {}, {}".format(
                    rel["sub_span"][0], rel["sub_span"][1], rel["obj_span"][0], rel["obj_span"][1]
                ) for rel in gold_rel_list
            ])

            correct_num += len(pred_rel_set.intersection(gold_rel_set))
            pred_num += len(pred_rel_set)
            gold_num += len(gold_rel_set)
        
        return pred_num, gold_num, correct_num