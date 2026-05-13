#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
YOLOX-S Combined QAT + Knowledge Distillation experiment.

Teacher: YOLOX-L (depth=1.0, width=1.0, full precision)
Student: YOLOX-S (depth=0.33, width=0.5, QAT-prepared)

Usage:
    python tools/train_qat_distill.py -f exps/default/yolox_s_qat_distill.py \
        -d 1 -b 16 teacher_ckpt /path/to/yolox_l.pth

The student is prepared for QAT (fake-quantize nodes) while being guided
by a full-precision teacher through distillation. This achieves the best
quantized model quality.
"""

import os

from yolox.exp import QATDistillExp


class Exp(QATDistillExp):
    def __init__(self):
        super().__init__()

        # Student architecture (YOLOX-S, will be QAT-prepared)
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Teacher architecture (YOLOX-L, full precision)
        self.teacher_depth = 1.0
        self.teacher_width = 1.0
        self.teacher_ckpt = None  # set via CLI: teacher_ckpt /path/to/yolox_l.pth

        # QAT schedule
        self.qat_backend = "fbgemm"
        self.qat_start_epoch = 2
        self.observer_freeze_epoch = 25

        # Distillation config
        self.distill_strategy = "both"
        self.distill_weight = 1.0
        self.feature_weight = 0.5
        self.logit_weight = 1.0
        self.cls_temperature = 3.0

        # Training schedule
        self.max_epoch = 30
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1
        self.no_aug_epochs = 5
        self.min_lr_ratio = 0.01

        self.eval_interval = 5
