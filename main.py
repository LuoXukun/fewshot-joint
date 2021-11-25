#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.10.29

import loguru

from config import *

from data_loader import get_loader
from tools import set_seed, load_parameters
from models.encoder import MyBertEncoder
from models import TaggingScheme, FewshotModel, MetricsCalculator
from utils.fewshotframework import FewshotJointFramework

def main():
    # Load parameters.
    args = load_parameters()

    # Log setting.
    if os.path.exists(args.log_path):
        os.remove(args.log_path)
    args.logger = loguru.logger
    args.logger.add(args.log_path)

    args.logger.info("{}-way-{}-shot Few-Shot Joint Extraction".format(args.N, args.K))
    args.logger.info(
        "model: {}, shaking_type: {}, inner_enc_type: {}, dist_type: {}, plus_type: {}, train: {}".format(
            args.model_type, args.shaking_type, args.inner_enc_type, args.dist_type, args.plus_type, args.train
        )
    )
    args.logger.info("seq_max_length: {}, label_max_len: {}".format(args.seq_max_length, args.label_max_length))
    args.logger.info("data_name: {}, data_type: {}, language: {}".format(args.data_name, args.data_type, args.language))

    # Set seed.
    set_seed(args.seed)

    # Model.
    args.logger.info("Loading models...")
    args.tagger = TaggingScheme[args.model_type](args.seq_max_length - args.label_max_length - 2)
    args.encoder = MyBertEncoder(model_type_dict[args.language])
    args.model = FewshotModel[args.model_type](args)
    args.metrics_calculator = MetricsCalculator[args.model_type](args.tagger)

    # Dataset.
    args.logger.info("Loading train dataset...")
    args.train_data_loader = get_loader(args, mode=0)
    args.logger.info("Loading valid dataset...")
    args.valid_data_loader = get_loader(args, mode=1)
    args.logger.info("Loading test dataset...")
    args.test_data_loader = get_loader(args, mode=2)

    # Framework
    args.logger.info("Building up few-shot Joint extraction framework...")
    args.framework = FewshotJointFramework(args)

    # Train, valid and test.
    if args.train:
        args.framework.train(args)
    else:
        args.framework.test(args, False)

if __name__ == "__main__":
    main()