#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.10.21

import os
current_path = os.path.abspath(os.path.dirname(__file__))

data_path_temp = [
    os.path.join(current_path, "data/<name_template>/<type_template>/train.json"),
    os.path.join(current_path, "data/<name_template>/<type_template>/dev.json"),
    os.path.join(current_path, "data/<name_template>/<type_template>/test.json"),
]

best_model_path = os.path.join(current_path, "result_model/best_model.ckpt")
if not os.path.exists(os.path.join(current_path, "result_model")):
    os.makedirs(os.path.join(current_path, "result_model"))

log_path = os.path.join(current_path, "logs/default_log.log")
if not os.path.exists(os.path.join(current_path, "logs")):
    os.makedirs(os.path.join(current_path, "logs"))

model_type_dict = {
    "en": "bert-base-cased",
    "ch": "bert-base-chinese"
}

tag_seqs_num = {
    "few-tplinker": 2,
    "few-tplinker-plus": 2,
    #"few-bitt": 8
}

seq_max_length = 50
label_max_length = 8
samples_length = 1000000000
map_hidden_size = 20
split_size = 32