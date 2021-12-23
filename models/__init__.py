#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.10.28

from models.FewTplinker import TPLinkerDataMaker, HandshakingTaggingScheme, TPLinkerMetricsCalculator, FewTPLinker
from models.FewTplinkerPlus import FewTPLinkerPlus
from models.PreTplinkerPlus import PreTPlinkerPlus, PreTPLinkerDataMaker
from models.FewBiTT import BiTTDataMaker, BidirectionalTreeTaggingScheme, BiTTMetricsCalculator, FewBiTT
from models.PreBiTT import PreBiTTDataMaker, PreBiTT

TaggingScheme = {
    "few-tplinker": HandshakingTaggingScheme,
    "few-tplinker-plus": HandshakingTaggingScheme,
    "few-bitt": BidirectionalTreeTaggingScheme
}

DataMaker = {
    "few-tplinker": TPLinkerDataMaker,
    "few-tplinker-plus": TPLinkerDataMaker,
    "few-bitt": BiTTDataMaker
}

MetricsCalculator = {
    "few-tplinker": TPLinkerMetricsCalculator,
    "few-tplinker-plus": TPLinkerMetricsCalculator,
    "few-bitt": BiTTMetricsCalculator
}

FewshotModel = {
    "few-tplinker": FewTPLinker,
    "few-tplinker-plus": FewTPLinkerPlus,
    "few-bitt": FewBiTT
}

PreDataMaker = {
    "few-tplinker-plus": PreTPLinkerDataMaker,
    "few-bitt": PreBiTTDataMaker
}

PreModel = {
    "few-tplinker-plus": PreTPlinkerPlus,
    "few-bitt": PreBiTT
}