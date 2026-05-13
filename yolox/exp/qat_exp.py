#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# QAT Experiment base class for YOLOX

import torch.nn as nn

from .yolox_base import Exp

__all__ = ["QATExp"]


class QATExp(Exp):
    """
    Experiment class for Quantization Aware Training.

    Extends the base YOLOX experiment to:
      - Wrap the float model with QAT fake-quantize nodes
      - Use the QATTrainer instead of the base Trainer
      - Configure QAT schedule (warmup, freeze epochs)

    QAT-specific config attributes:
      - qat_backend: "fbgemm" (server) or "qnnpack" (mobile)
      - qat_start_epoch: epoch to enable fake-quant (0 = from start)
      - observer_freeze_epoch: epoch to freeze observers (-1 = never)
      - pretrained_ckpt: path to pretrained float model checkpoint
    """

    def __init__(self):
        super().__init__()

        # QAT config
        self.qat_backend = "fbgemm"
        self.qat_start_epoch = 0
        self.observer_freeze_epoch = -1
        self.pretrained_ckpt = None

        # QAT typically uses fewer epochs and lower LR
        self.max_epoch = 30
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1
        self.no_aug_epochs = 5
        self.min_lr_ratio = 0.01

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
        from yolox.models.qat import QATModel

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            float_model = YOLOX(backbone, head)
            float_model.apply(init_yolo)
            float_model.head.initialize_biases(1e-2)

            if self.pretrained_ckpt:
                import torch
                from yolox.utils import load_ckpt
                ckpt = torch.load(self.pretrained_ckpt, map_location="cpu")
                if "model" in ckpt:
                    ckpt = ckpt["model"]
                float_model = load_ckpt(float_model, ckpt)

            self.model = QATModel(
                float_model,
                backend=self.qat_backend,
                qat_start_epoch=self.qat_start_epoch,
                observer_freeze_epoch=self.observer_freeze_epoch,
            )

        self.model.train()
        return self.model

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    if hasattr(v, "weight") and v.weight is not None:
                        pg0.append(v.weight)
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)

            import torch
            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_trainer(self, args):
        from yolox.core import QATTrainer
        trainer = QATTrainer(self, args)
        return trainer
