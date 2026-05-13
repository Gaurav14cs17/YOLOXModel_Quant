#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
YOLOX-S Knowledge Distillation experiment.

Teacher: YOLOX-L (depth=1.0, width=1.0)
Student: YOLOX-S (depth=0.33, width=0.5)

Usage:
    python tools/train_distill.py -f exps/default/yolox_s_distill.py \
        -d 1 -b 16 teacher_ckpt /path/to/yolox_l.pth

The student learns from both ground-truth labels and the teacher's
soft predictions + feature representations.
"""

import os

from yolox.exp import DistillExp


class Exp(DistillExp):
    def __init__(self):
        super().__init__()

        # Student architecture (YOLOX-S)
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Teacher architecture (YOLOX-L)
        self.teacher_depth = 1.0
        self.teacher_width = 1.0
        self.teacher_ckpt = None  # set via CLI: teacher_ckpt /path/to/yolox_l.pth

        # Distillation config
        self.distill_strategy = "both"
        self.distill_weight = 1.0
        self.feature_weight = 0.5
        self.logit_weight = 1.0
        self.cls_temperature = 3.0
        self.feature_loss_type = "l2"

        # Training schedule
        self.max_epoch = 100
        self.basic_lr_per_img = 0.005 / 64.0
        self.warmup_epochs = 3
        self.no_aug_epochs = 10

        self.eval_interval = 5
