#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.11.22

""" 
    Some variant models of Few-Tplinker, which reduce memory consumption and speed up training.
    We simply cat the hidden state of a token-pair, thus there is no need of HandshakingKernel when calculating the distance.
"""

import torch
import torch.nn as nn
import numpy as np

from torch.nn.parameter import Parameter

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x-mean) / (std+self.eps) + self.beta

class FewTPLinkerPlus(nn.Module):
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
                plus_type:          The type of variant models. ["dot-sigmoid", "top-k", "negative-sampling"]
        """
        super(FewTPLinkerPlus, self).__init__()
        self.encoder = args.encoder
        self.hidden_size = self.encoder.config.hidden_size
        self.shaking_type = args.shaking_type
        self.inner_enc_type = args.inner_enc_type
        self.map_hidden_size = args.map_hidden_size
        self.dist_type = args.dist_type
        self.label_max_length = args.label_max_length
        self.seq_max_length = args.seq_max_length
        self.split_size = args.split_size
        self.plus_type = args.plus_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.plus_type == "dot-sigmoid" and self.dist_type != "dot":
            args.logger.warning("The plus_type of Few-TPlinker is dot-sogmoid, thus the dist_type should be dot, we will change for you!")
            self.dist_type = "dot"

        # Mapping Layer
        self.ent_map_fc = nn.ModuleList([nn.Linear(self.hidden_size, self.map_hidden_size) for _ in range(2)])
        self.head_rel_map_fc = nn.ModuleList([nn.Linear(self.hidden_size, self.map_hidden_size) for _ in range(2)])

        # Drop out
        self.dropout = nn.Dropout()

        # Layer normalization.
        self.norm_head = LayerNorm(self.seq_max_length  - self.label_max_length - 2)
        self.norm_tail = LayerNorm(self.seq_max_length  - self.label_max_length - 2)

        self.cost = nn.CrossEntropyLoss() if self.plus_type != "dot-sigmoid" else nn.BCELoss()

        self.__get_nearest_dist__ = {
            "dot-sigmoid": self.__get_nearest_dist_dot_sigmoid__,
            "top-k": self.__get_nearest_dist_top_k__,
            "negative-sampling": self.__get_nearest_dist_negative_sampling__
        }
    
    def forward(self, support, query):
        # Support.
        support_emb = self.encoder(support["src_ids"], support["mask_ids"], support["seg_ids"])[:, self.label_max_length + 2:, :]
        ent_support_hidden = torch.stack([self.ent_map_fc[i](self.dropout(support_emb)) for i in range(2)])             # The head and tail ent matrix for handshaking.
        head_rel_support_hidden = torch.stack([self.head_rel_map_fc[i](self.dropout(support_emb)) for i in range(2)])   # The head and tail rel matrix for handshaking.

        # Query.
        query_emb = self.encoder(query["src_ids"], query["mask_ids"], query["seg_ids"])[:, self.label_max_length + 2:, :]
        ent_query_hidden = torch.stack([self.ent_map_fc[i](self.dropout(query_emb)) for i in range(2)])                 # The head and tail ent matrix for handshaking.
        head_rel_query_hidden = torch.stack([self.head_rel_map_fc[i](self.dropout(query_emb)) for i in range(2)])       # The head and tail rel matrix for handshaking.

        logits, pred = [[] for i in range(2)], [[] for i in range(2)] # ent, head_rel
        current_support_num, current_query_num = 0, 0
        
        for index, support_samples_num in enumerate(support["samples_num"]):
            query_samples_num = query["samples_num"][index]
            # Calculate nearest distance to each tokens pair in each class in support set.
            #print("ent")
            logits[0].append(   # ent
                self.__get_nearest_dist__[self.plus_type](
                    ent_support_hidden[:, current_support_num:current_support_num+support_samples_num],
                    support["tags"][0][current_support_num:current_support_num+support_samples_num],
                    ent_query_hidden[:, current_query_num:current_query_num+query_samples_num]
                )
            )
            #print("head_rel")
            logits[1].append(   # head_rel
                self.__get_nearest_dist__[self.plus_type](
                    head_rel_support_hidden[:, current_support_num:current_support_num+support_samples_num],
                    support["tags"][1][current_support_num:current_support_num+support_samples_num],
                    head_rel_query_hidden[:, current_query_num:current_query_num+query_samples_num]
                )
            )
            current_query_num += query_samples_num
            current_support_num += support_samples_num

        for i in range(2):
            logits[i] = torch.cat(logits[i], 0)
            _, pred[i] = torch.max(logits[i], -1)
        
        return logits, pred
    
    def inference(self, support, query):
        # Support.
        support_emb = self.encoder(support["src_ids"], support["mask_ids"], support["seg_ids"])[:, self.label_max_length + 2:, :]
        ent_support_hidden = torch.stack([self.ent_map_fc[i](self.dropout(support_emb)) for i in range(2)])             # The head and tail ent matrix for handshaking.
        head_rel_support_hidden = torch.stack([self.head_rel_map_fc[i](self.dropout(support_emb)) for i in range(2)])   # The head and tail rel matrix for handshaking.

        # Query.
        query_emb = self.encoder(query["src_ids"], query["mask_ids"], query["seg_ids"])[:, self.label_max_length + 2:, :]
        ent_query_hidden = torch.stack([self.ent_map_fc[i](self.dropout(query_emb)) for i in range(2)])                 # The head and tail ent matrix for handshaking.
        head_rel_query_hidden = torch.stack([self.head_rel_map_fc[i](self.dropout(query_emb)) for i in range(2)])       # The head and tail rel matrix for handshaking.

        pred = [[] for i in range(2)] # ent, head_rel
        current_support_num, current_query_num = 0, 0
        
        for index, support_samples_num in enumerate(support["samples_num"]):
            query_samples_num = query["samples_num"][index]
            # Calculate nearest distance to each tokens pair in each class in support set.
            #print("ent")
            pred[0].append(   # ent
                self.__get_inference_preds__(
                    ent_support_hidden[:, current_support_num:current_support_num+support_samples_num],
                    support["tags"][0][current_support_num:current_support_num+support_samples_num],
                    ent_query_hidden[:, current_query_num:current_query_num+query_samples_num]
                )
            )
            #print("head_rel")
            pred[1].append(   # head_rel
                self.__get_inference_preds__(
                    head_rel_support_hidden[:, current_support_num:current_support_num+support_samples_num],
                    support["tags"][1][current_support_num:current_support_num+support_samples_num],
                    head_rel_query_hidden[:, current_query_num:current_query_num+query_samples_num]
                )
            )
            current_query_num += query_samples_num
            current_support_num += support_samples_num

        for i in range(2):
            pred[i] = torch.cat(pred[i], 0)
        
        return None, pred

    def __get_nearest_dist_dot_sigmoid__(self, support_hidden, tag, query_hidden):
        """ 
            Get the nearest distance. Dot sigmoid.
            Args:
                support_hidden:         (2, support_num, seq_len_support -> seq_S, hidden_size)
                tag:                    (support_num, seq_S x seq_S)
                query_hidden:           (2, query_num, seq_len_query -> seq_Q, hidden_size)
            Returns:
                logits:                 (query, seq_Q x seq_Q, 2)
        """
        seq_len, hidden_size = support_hidden.size(2), support_hidden.size(3)
        support_num, query_num = support_hidden.size(1), query_hidden.size(1)

        #print("seq_len: {}, hidden_size: {}, support_num: {}, query_num: {}".format(seq_len, hidden_size, support_num, query_num))

        # Get the head and tail distance matrix.
        S_head, S_tail = support_hidden[0].unsqueeze(0), support_hidden[1].unsqueeze(0)                             # (1, support_num, seq_S, hidden_size)
        Q_head, Q_tail = query_hidden[0].view(-1, 1, 1, hidden_size), query_hidden[1].view(-1, 1, 1, hidden_size)   # (query_num x seq_Q -> tokens_Q, 1, 1, hidden_size)
        #print("S_head: {}, S_tail: {}, Q_head: {}, Q_tail: {}".format(S_head.size(), S_tail.size(), Q_head.size(), Q_tail.size()))
        dist_head = self.__dist__(S_head, Q_head, dim=-1)   # (tokens_Q, support_num, seq_S)
        dist_tail = self.__dist__(S_tail, Q_tail, dim=-1)   # (tokens_Q, support_num, seq_S)
        #print("dist_head:", dist_head)
        #print("dist_tail: ", dist_tail)

        # Nomalize the distance.
        dist_head = self.norm_head(dist_head)               # (tokens_Q, support_num, seq_S)
        dist_tail = self.norm_tail(dist_tail)               # (tokens_Q, support_num, seq_S)
        #print("dist_head_norm:", dist_head)
        #print("dist_tail_norm: ", dist_tail)

        # Get the max distance and the index.
        dist_head_max, index_head_max = torch.max(dist_head, dim=-1)                                                # (tokens_Q, support_num)
        dist_tail_max, index_tail_max = torch.max(dist_tail, dim=-1)                                                # (tokens_Q, support_num)
        dist_max = dist_head_max.view(query_num, seq_len, 1, -1) + dist_tail_max.view(query_num, 1, seq_len, -1)    # (query_num, seq_Q, seq_Q, support_num)
        dist_max, index_max = torch.max(dist_max, dim=-1)                                                           # (query_num, seq_Q, seq_Q)
        #print("dist_max: ", dist_max)
        dist_max = torch.sigmoid(dist_max)                                                                          # (query_num, seq_Q, seq_Q)
        #print("dist_max_sigmoid: ", dist_max)

        # Get the dist_max mask.
        tag_mask = tag.view(support_num, seq_len, -1)                                                               # (support_num, seq_S, seq_S)
        index_max = index_max.view(-1)                                                                              # (query_num x seq_Q x seq_Q)
        iter_index = torch.linspace(0, query_num * seq_len - 1, query_num * seq_len).long().to(self.device)         # (query_num x seq_Q), [0, 1, 2, 3]
        iter_index_head = iter_index.view(-1, 1).expand(-1, seq_len).contiguous().view(-1)                          # (query_num x seq_Q x seq_Q), [00, 11, 22, 33]
        iter_index_tail = iter_index_head.view(query_num, seq_len, -1).transpose(1, 2).contiguous().view(-1)        # (query_num x seq_Q x seq_Q), [01, 01, 23, 23]
        index_head_max = index_head_max[iter_index_head, index_max]                                                 # (query_num x seq_Q x seq_Q)
        index_tail_max = index_tail_max[iter_index_tail, index_max]                                                 # (query_num x seq_Q x seq_Q)
        dist_max_mask = tag_mask[index_max, index_head_max, index_tail_max].view(query_num, seq_len, seq_len)       # (query_num, seq_Q, seq_Q)

        # Get the dist_max matrix for tag 0 and 1.
        # dist_max_0 = (1 - dist_max) * dist_max_mask + dist_max * (1 - dist_max_mask)
        dist_max_0 = dist_max + dist_max_mask - 2 * dist_max * dist_max_mask            # (query_num, seq_Q, seq_Q)
        dist_max_1 = 1 - dist_max_0                                                     # (query_num, seq_Q, seq_Q)
        logits = torch.stack([dist_max_0, dist_max_1], dim=3).view(query_num, -1, 2)    # (query_num, seq_Q x seq_Q, 2)

        return logits
    
    def __get_nearest_dist_top_k__(self, support_hidden, tag, query_hidden):
        """ 
            Get the nearest distance. Top k.
            Args:
                support_hidden:         (2, support_num, seq_len_support -> seq_S, hidden_size)
                tag:                    (support_num, seq_S x seq_S)
                query_hidden:           (2, query_num, seq_len_query -> seq_Q, hidden_size)
            Returns:
                logits:                 (query, seq_Q x seq_Q, 2)
        """
        seq_len, hidden_size = support_hidden.size(2), support_hidden.size(3)
        support_num, query_num = support_hidden.size(1), query_hidden.size(1)

        # Get the head and tail distance matrix.
        S_head, S_tail = support_hidden[0].unsqueeze(0), support_hidden[1].unsqueeze(0)                             # (1, support_num, seq_S, hidden_size)
        Q_head, Q_tail = query_hidden[0].view(-1, 1, 1, hidden_size), query_hidden[1].view(-1, 1, 1, hidden_size)   # (query_num x seq_Q -> tokens_Q, 1, 1, hidden_size)
        dist_head = self.__dist__(S_head, Q_head, dim=-1)   # (tokens_Q, support_num, seq_S)
        dist_tail = self.__dist__(S_tail, Q_tail, dim=-1)   # (tokens_Q, support_num, seq_S)

        # Get the top 2 distance and the index.
        dist_head_top2, index_head_top2 = torch.topk(dist_head, 2, dim=-1)                                                      # (tokens_Q, support_num, 2)
        dist_tail_top2, index_tail_top2 = torch.topk(dist_tail, 2, dim=-1)                                                      # (tokens_Q, support_num, 2)
        dist_top4 = dist_head_top2.view(query_num, seq_len, 1, -1, 2, 1) + dist_tail_top2.view(query_num, 1, seq_len, -1, 1, 2) # (query_num, seq_Q, seq_Q, support_num, 2, 2)
        dist_top2, index_top2 = torch.topk(dist_top4.view(query_num, seq_len, seq_len, -1), 2, dim=-1)                          # (query_num, seq_Q, seq_Q, 2)

        # Get the max distance for tag 1.
        index_tag1 = torch.nonzero(tag.view(support_num, seq_len, -1), as_tuple=True)
        dist_head_tag1 = dist_head[:, index_tag1[0], index_tag1[1]]                                                 # (tokens_Q, tag1_num)
        dist_tail_tag1 = dist_tail[:, index_tag1[0], index_tag1[2]]                                                 # (tokens_Q, tag1_num)
        dist_tag1 = dist_head_tag1.view(query_num, seq_len, 1, -1) + dist_tail_tag1.view(query_num, 1, seq_len, -1) # (query_num, seq_Q, seq_Q, tag1_num)
        dist_max_tag1, index_max_tag1 = torch.max(dist_tag1, dim=-1)                                                # (query_num, seq_Q, seq_Q)

        # Get the max distance for tag 0 approximately.
        mask_tag1 = (dist_top2[:, :, :, 0] == dist_max_tag1).long()                                                 # (query_num, seq_Q, seq_Q)
        dist_max_tag0 = dist_top2[:, :, :, 0] * (1 - mask_tag1) + dist_top2[:, :, :, 1] * mask_tag1                 # (query_num, seq_Q, seq_Q)

        # Stack the logits.
        logits = torch.stack([dist_max_tag0, dist_max_tag1], dim=3).view(query_num, -1, 2)                          # (query_num, seq_Q x seq_Q, 2)

        return logits

    def __get_nearest_dist_negative_sampling__(self, support_hidden, tag, query_hidden):
        pass

    def __get_inference_preds__(self, support_hidden, tag, query_hidden):
        """ 
            Get the inference preds.
            Args:
                support_hidden:         (2, support_num, seq_len_support -> seq_S, hidden_size)
                tag:                    (support_num, seq_S x seq_S)
                query_hidden:           (2, query_num, seq_len_query -> seq_Q, hidden_size)
            Returns:
                preds:                  (query_num, seq_Q x seq_Q)
        """
        seq_len, hidden_size = support_hidden.size(2), support_hidden.size(3)
        support_num, query_num = support_hidden.size(1), query_hidden.size(1)

        # Get the head and tail distance matrix.
        S_head, S_tail = support_hidden[0].unsqueeze(0), support_hidden[1].unsqueeze(0)                             # (1, support_num, seq_S, hidden_size)
        Q_head, Q_tail = query_hidden[0].view(-1, 1, 1, hidden_size), query_hidden[1].view(-1, 1, 1, hidden_size)   # (query_num x seq_Q -> tokens_Q, 1, 1, hidden_size)
        dist_head = self.__dist__(S_head, Q_head, dim=-1)   # (tokens_Q, support_num, seq_S)
        dist_tail = self.__dist__(S_tail, Q_tail, dim=-1)   # (tokens_Q, support_num, seq_S)

        # Nomalize the distance.
        if self.plus_type == "dot-sigmoid":
            dist_head = self.norm_head(dist_head)               # (tokens_Q, support_num, seq_S)
            dist_tail = self.norm_tail(dist_tail)               # (tokens_Q, support_num, seq_S)

        # Get the max distance and the index.
        dist_head_max, index_head_max = torch.max(dist_head, dim=-1)                                                # (tokens_Q, support_num)
        dist_tail_max, index_tail_max = torch.max(dist_tail, dim=-1)                                                # (tokens_Q, support_num)
        dist_max = dist_head_max.view(query_num, seq_len, 1, -1) + dist_tail_max.view(query_num, 1, seq_len, -1)    # (query_num, seq_Q, seq_Q, support_num)
        dist_max, index_max = torch.max(dist_max, dim=-1)                                                           # (query_num, seq_Q, seq_Q)

        # Get the max distance for tag 1.
        index_tag1 = torch.nonzero(tag.view(support_num, seq_len, -1), as_tuple=True)
        dist_head_tag1 = dist_head[:, index_tag1[0], index_tag1[1]]                                                 # (tokens_Q, tag1_num)
        dist_tail_tag1 = dist_tail[:, index_tag1[0], index_tag1[2]]                                                 # (tokens_Q, tag1_num)
        dist_tag1 = dist_head_tag1.view(query_num, seq_len, 1, -1) + dist_tail_tag1.view(query_num, 1, seq_len, -1) # (query_num, seq_Q, seq_Q, tag1_num)
        dist_max_tag1, index_max_tag1 = torch.max(dist_tag1, dim=-1)                                                # (query_num, seq_Q, seq_Q)

        # Get the preds.
        preds = (dist_max == dist_max_tag1).long().view(query_num, -1)

        """ # Get the preds.
        tag_mask = tag.view(support_num, seq_len, -1)                                                               # (support_num, seq_S, seq_S)
        index_max = index_max.view(-1)                                                                              # (query_num x seq_Q x seq_Q)
        iter_index = torch.linspace(0, query_num * seq_len - 1, query_num * seq_len).long().to(self.device)         # (query_num x seq_Q), [0, 1, 2, 3]
        iter_index_head = iter_index.view(-1, 1).expand(-1, seq_len).contiguous().view(-1)                          # (query_num x seq_Q x seq_Q), [00, 11, 22, 33]
        iter_index_tail = iter_index_head.view(query_num, seq_len, -1).transpose(1, 2).contiguous().view(-1)        # (query_num x seq_Q x seq_Q), [01, 01, 23, 23]
        index_head_max = index_head_max[iter_index_head, index_max]                                                 # (query_num x seq_Q x seq_Q)
        index_tail_max = index_tail_max[iter_index_tail, index_max]                                                 # (query_num x seq_Q x seq_Q)
        preds = tag_mask[index_max, index_head_max, index_tail_max].view(query_num, seq_len * seq_len)              # (query_num, seq_Q x seq_Q) """

        return preds

    def __dist__(self, x, y, dim):
        #torch.cuda.empty_cache()
        if self.dist_type == "dot":
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)
    
    def loss(self, logits, label):
        '''
            Args:

                logits:     Logits with the size (2, query_num, seq_len x seq_len, class_num)
                label:      Label with the size (2, query_num, seq_len x seq_len).

            Returns: 

                [Loss] (A single value)
        '''
        # 此处无法使用mask_ids来去掉[PAD]得到的logits，因为logits形状变了，
        # 但由于attention计算使用了mask，会使那一部分的参数不更新
        loss = []
        for i in range(2):
            if self.plus_type == "dot-sigmoid":
                loss.append(self.cost(logits[i][:, :, 1].view(-1), label[i].float().view(-1)))
            else:
                N = logits[i].size(-1)
                loss.append(self.cost(logits[i].view(-1, N), label[i].view(-1)))
        loss = loss[0] + loss[1]
        return loss