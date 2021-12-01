#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.10.29

import loguru

from config import *

from data_loader import get_loader
from data_loader_pre import get_loader_pre
from tools import set_seed, load_parameters
from models.encoder import MyBertEncoder
from models import TaggingScheme, FewshotModel, PreModel, MetricsCalculator
from utils.fewshotframework import FewshotJointFramework

def main():
    # Load parameters.
    args = load_parameters()

    # Log setting.
    if os.path.exists(args.log_path):
        os.remove(args.log_path)
    args.logger = loguru.logger
    args.logger.add(args.log_path)

    if args.fewshot:
        args.logger.info("{}-way-{}-shot Few-Shot Joint Extraction".format(args.N, args.K))
        args.logger.info(
            "model: {}, shaking_type: {}, inner_enc_type: {}, dist_type: {}, plus_type: {}, train: {}".format(
                args.model_type, args.shaking_type, args.inner_enc_type, args.dist_type, args.plus_type, args.train
            )
        )
    if args.pretrain:
        args.logger.info("Pretrain Joint Extraction for {}".format(args.model_type))
    args.logger.info("seq_max_length: {}, label_max_len: {}".format(args.seq_max_length, args.label_max_length))
    args.logger.info("data_name: {}, data_type: {}, language: {}".format(args.data_name, args.data_type, args.language))

    # Set seed.
    set_seed(args.seed)

    # Model.
    args.logger.info("Loading models...")
    args.tagger = TaggingScheme[args.model_type](args.seq_max_length - args.label_max_length - 2)
    args.encoder = MyBertEncoder(model_type_dict[args.language])
    args.metrics_calculator = MetricsCalculator[args.model_type](args.tagger)
    args.model = FewshotModel[args.model_type](args) if args.fewshot else None
    args.pre_model = PreModel[args.model_type](args) if args.pretrain else None

    # Dataset.
    args.logger.info("Loading fewshot train dataset...")
    args.train_data_loader = get_loader(args, mode=0) if args.fewshot else None
    args.logger.info("Loading fewshot valid dataset...")
    args.valid_data_loader = get_loader(args, mode=1) if args.fewshot else None

    args.logger.info("Loading pretrain train dataset...")
    args.pre_train_data_loader = get_loader_pre(args, mode=0) if args.pretrain else None
    args.logger.info("Loading pretrain valid dataset...")
    args.pre_valid_data_loader = get_loader_pre(args, mode=1) if args.pretrain else None

    # Something to assert.
    if args.pretrain:
        train_tag2id = args.pre_train_data_loader.dataset.tag2id
        valid_tag2id = args.pre_valid_data_loader.dataset.tag2id
        for key in train_tag2id:
            assert train_tag2id[key] == valid_tag2id[key]
    if args.pretrain and args.fewshot:
        if not args.pre_ckpt:
            assert args.pre_ckpt is not None
        if args.pre_ckpt != args.load_ckpt:
            args.logger.warning("The pre_ckpt and load_ckpt should be the same. We will change load_ckpt to pre_ckpt '{}'".format(args.pre_ckpt))

    # Framework
    args.logger.info("Building up few-shot Joint extraction framework...")
    args.framework = FewshotJointFramework(args)

    # Train, valid and test.
    if args.pretrain:
        args.framework.pretrain(args)
    if args.fewshot:
        if args.train:
            args.framework.train(args)
        else:
            args.framework.test(args)

if __name__ == "__main__":
    main()