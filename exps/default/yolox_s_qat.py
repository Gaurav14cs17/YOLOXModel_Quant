#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
YOLOX-S Quantization Aware Training experiment.

Usage:
    python tools/train_qat.py -f exps/default/yolox_s_qat.py \
        -c /path/to/yolox_s.pth -d 1 -b 16

This starts from a pretrained YOLOX-S model and fine-tunes it with
fake-quantize nodes inserted for QAT.
"""

import os

from yolox.exp import QATExp


class Exp(QATExp):
    def __init__(self):
        super().__init__()

        # Student architecture (YOLOX-S)
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # QAT schedule
        self.qat_backend = "fbgemm"
        self.qat_start_epoch = 2
        self.observer_freeze_epoch = 25
        self.max_epoch = 30

        # Reduced LR for QAT fine-tuning
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1
        self.no_aug_epochs = 5
        self.min_lr_ratio = 0.01

        self.eval_interval = 5
