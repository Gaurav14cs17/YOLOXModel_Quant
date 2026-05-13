#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Knowledge Distillation Experiment base class for YOLOX

import torch
import torch.nn as nn

from .yolox_base import Exp

__all__ = ["DistillExp", "QATDistillExp"]


class DistillExp(Exp):
    """
    Experiment class for Knowledge Distillation training.

    Builds a DistillationYOLOX model that wraps a student and teacher.
    The teacher is a larger/pretrained YOLOX variant, the student is
    the target model.

    Config attributes:
      - teacher_depth / teacher_width: architecture scale of teacher
      - teacher_ckpt: path to teacher's pretrained weights
      - distill_strategy: "feature", "logit", or "both"
      - distill_weight: global scaling factor for distill loss
      - feature_weight: weight for feature distillation term
      - logit_weight: weight for logit distillation term
      - cls_temperature: temperature for classification KD
      - feature_loss_type: "l2" or "l1"
    """

    def __init__(self):
        super().__init__()

        # Teacher config
        self.teacher_depth = 1.0
        self.teacher_width = 1.0
        self.teacher_ckpt = None

        # Distillation config
        self.distill_strategy = "both"
        self.distill_weight = 1.0
        self.feature_weight = 0.5
        self.logit_weight = 1.0
        self.cls_temperature = 3.0
        self.feature_loss_type = "l2"

        # Student often trains with smaller LR and fewer epochs
        self.max_epoch = 100
        self.basic_lr_per_img = 0.005 / 64.0

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
        from yolox.models.distillation import DistillationYOLOX, DistillationLoss

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]

            # Build student
            s_backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            s_head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            student = YOLOX(s_backbone, s_head)
            student.apply(init_yolo)
            student.head.initialize_biases(1e-2)

            # Build teacher
            t_backbone = YOLOPAFPN(
                self.teacher_depth, self.teacher_width,
                in_channels=in_channels, act=self.act
            )
            t_head = YOLOXHead(
                self.num_classes, self.teacher_width,
                in_channels=in_channels, act=self.act
            )
            teacher = YOLOX(t_backbone, t_head)
            teacher.apply(init_yolo)

            if self.teacher_ckpt:
                from yolox.utils import load_ckpt
                ckpt = torch.load(self.teacher_ckpt, map_location="cpu")
                if "model" in ckpt:
                    ckpt = ckpt["model"]
                teacher = load_ckpt(teacher, ckpt)
            teacher.eval()

            student_ch = [int(c * self.width) for c in in_channels]
            teacher_ch = [int(c * self.teacher_width) for c in in_channels]

            distill_loss = DistillationLoss(
                strategy=self.distill_strategy,
                feature_weight=self.feature_weight,
                logit_weight=self.logit_weight,
                cls_temperature=self.cls_temperature,
                feature_loss_type=self.feature_loss_type,
            )

            self.model = DistillationYOLOX(
                student=student,
                teacher=teacher,
                distill_loss=distill_loss,
                distill_weight=self.distill_weight,
                student_channels=student_ch,
                teacher_channels=teacher_ch,
            )

        self.model.train()
        return self.model

    def get_optimizer(self, batch_size):
        """Only optimize student parameters (teacher is frozen)."""
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []

            trainable_modules = list(self.model.student.named_modules())
            if self.model.adapt_layers is not None:
                trainable_modules += list(self.model.adapt_layers.named_modules())

            for k, v in trainable_modules:
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    if hasattr(v, "weight") and v.weight is not None:
                        pg0.append(v.weight)
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)

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
        from yolox.core import DistillTrainer
        trainer = DistillTrainer(self, args)
        return trainer

    def eval(self, model, evaluator, is_distributed, half=False, return_outputs=False):
        student = model.student if hasattr(model, "student") else model
        return evaluator.evaluate(student, is_distributed, half, return_outputs=return_outputs)


class QATDistillExp(DistillExp):
    """
    Combined QAT + Distillation experiment.

    The student model is prepared for QAT, while a full-precision teacher
    guides learning via distillation. Best of both worlds: the student
    learns from the teacher while being quantization-aware.
    """

    def __init__(self):
        super().__init__()

        # QAT config
        self.qat_backend = "fbgemm"
        self.qat_start_epoch = 0
        self.observer_freeze_epoch = -1

        # Tighter training schedule for QAT+Distill
        self.max_epoch = 30
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1
        self.no_aug_epochs = 5
        self.min_lr_ratio = 0.01

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
        from yolox.models.distillation import DistillationYOLOX, DistillationLoss
        from yolox.models.qat import prepare_qat_model

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]

            # Build student (will be QAT-prepared)
            s_backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            s_head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            student = YOLOX(s_backbone, s_head)
            student.apply(init_yolo)
            student.head.initialize_biases(1e-2)

            student = prepare_qat_model(student, backend=self.qat_backend, inplace=True)

            # Build teacher (full precision)
            t_backbone = YOLOPAFPN(
                self.teacher_depth, self.teacher_width,
                in_channels=in_channels, act=self.act
            )
            t_head = YOLOXHead(
                self.num_classes, self.teacher_width,
                in_channels=in_channels, act=self.act
            )
            teacher = YOLOX(t_backbone, t_head)
            teacher.apply(init_yolo)

            if self.teacher_ckpt:
                from yolox.utils import load_ckpt
                ckpt = torch.load(self.teacher_ckpt, map_location="cpu")
                if "model" in ckpt:
                    ckpt = ckpt["model"]
                teacher = load_ckpt(teacher, ckpt)
            teacher.eval()

            student_ch = [int(c * self.width) for c in in_channels]
            teacher_ch = [int(c * self.teacher_width) for c in in_channels]

            distill_loss = DistillationLoss(
                strategy=self.distill_strategy,
                feature_weight=self.feature_weight,
                logit_weight=self.logit_weight,
                cls_temperature=self.cls_temperature,
                feature_loss_type=self.feature_loss_type,
            )

            self.model = DistillationYOLOX(
                student=student,
                teacher=teacher,
                distill_loss=distill_loss,
                distill_weight=self.distill_weight,
                student_channels=student_ch,
                teacher_channels=teacher_ch,
            )

        self.model.train()
        return self.model

    def get_trainer(self, args):
        from yolox.core import QATDistillTrainer
        trainer = QATDistillTrainer(self, args)
        return trainer
