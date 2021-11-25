#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.10.28

from models.FewTplinker import TPLinkerDataMaker, HandshakingTaggingScheme, TPLinkerMetricsCalculator, FewTPLinker
from models.FewTplinkerPlus import FewTPLinkerPlus
#from models.FewBiTT import BidirectionalTreeTaggingScheme

TaggingScheme = {
    "few-tplinker": HandshakingTaggingScheme,
    "few-tplinker-plus": HandshakingTaggingScheme
    #"few-bitt": BidirectionalTreeTaggingScheme
}

DataMaker = {
    "few-tplinker": TPLinkerDataMaker,
    "few-tplinker-plus": TPLinkerDataMaker
}

MetricsCalculator = {
    "few-tplinker": TPLinkerMetricsCalculator,
    "few-tplinker-plus": TPLinkerMetricsCalculator
}

FewshotModel = {
    "few-tplinker": FewTPLinker,
    "few-tplinker-plus": FewTPLinkerPlus
}