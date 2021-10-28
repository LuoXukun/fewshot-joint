#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.10.28

from models.FewTplinker import TPLinkerDataMaker, HandshakingTaggingScheme, TPLinkerMetricsCalculator, FewTPLinker

TaggingScheme = {
    "few-tplinker": HandshakingTaggingScheme
}

DataMaker = {
    "few-tplinker": TPLinkerDataMaker
}

MetricsCalculator = {
    "few-tplinker": TPLinkerMetricsCalculator
}

FewshotModel = {
    "few-tplinker": FewTPLinker
}