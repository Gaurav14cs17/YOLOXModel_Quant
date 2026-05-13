#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Knowledge Distillation support for YOLOX

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAdaptation(nn.Module):
    """
    1x1 conv to adapt student feature channels to teacher feature channels
    when they differ in width.
    """

    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        if student_channels != teacher_channels:
            self.adapt = nn.Conv2d(student_channels, teacher_channels, 1, bias=False)
            nn.init.kaiming_normal_(self.adapt.weight)
        else:
            self.adapt = nn.Identity()

    def forward(self, x):
        return self.adapt(x)


class DistillationLoss(nn.Module):
    """
    Combined distillation loss for YOLOX:
      1. Feature-level distillation (FPN feature maps)
      2. Logit-level distillation (cls/obj/reg head outputs)

    Supports multiple distillation strategies:
      - "feature": only FPN feature distillation
      - "logit": only head logit distillation
      - "both": feature + logit distillation
    """

    def __init__(
        self,
        strategy: str = "both",
        feature_weight: float = 0.5,
        logit_weight: float = 1.0,
        cls_temperature: float = 3.0,
        feature_loss_type: str = "l2",
    ):
        """
        Args:
            strategy: "feature", "logit", or "both"
            feature_weight: weight for feature distillation loss
            logit_weight: weight for logit distillation loss
            cls_temperature: temperature for softening classification logits
            feature_loss_type: "l2" (MSE) or "l1" for feature distillation
        """
        super().__init__()
        self.strategy = strategy
        self.feature_weight = feature_weight
        self.logit_weight = logit_weight
        self.cls_temperature = cls_temperature
        self.feature_loss_type = feature_loss_type

    def feature_distillation_loss(self, student_feats, teacher_feats, adapt_layers=None):
        """
        L2 or L1 loss between normalized student and teacher FPN features.
        Both are tuples of 3 tensors: (P3/8, P4/16, P5/32).
        """
        loss = 0.0
        for i, (s_feat, t_feat) in enumerate(zip(student_feats, teacher_feats)):
            if adapt_layers is not None:
                s_feat = adapt_layers[i](s_feat)

            if s_feat.shape[2:] != t_feat.shape[2:]:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode="bilinear", align_corners=False)

            s_norm = F.normalize(s_feat.flatten(2), dim=-1)
            t_norm = F.normalize(t_feat.flatten(2).detach(), dim=-1)

            if self.feature_loss_type == "l2":
                loss += F.mse_loss(s_norm, t_norm)
            else:
                loss += F.l1_loss(s_norm, t_norm)
        return loss / len(student_feats)

    def logit_distillation_loss(self, student_logits, teacher_logits):
        """
        KD loss on head logits (cls + obj) using temperature-scaled KL divergence.
        student_logits / teacher_logits: dict with keys "cls", "obj", "reg"
        each shaped [B, N_anchors, C].
        """
        T = self.cls_temperature
        loss = 0.0

        # Classification KD: KL divergence with temperature
        s_cls = student_logits["cls"] / T
        t_cls = teacher_logits["cls"].detach() / T
        kd_cls = F.kl_div(
            F.log_softmax(s_cls, dim=-1),
            F.softmax(t_cls, dim=-1),
            reduction="batchmean",
        ) * (T * T)
        loss += kd_cls

        # Objectness KD: BCE between student and teacher sigmoid outputs
        s_obj = student_logits["obj"]
        t_obj = teacher_logits["obj"].detach()
        kd_obj = F.binary_cross_entropy_with_logits(
            s_obj, t_obj.sigmoid(), reduction="mean"
        )
        loss += kd_obj

        # Regression KD: L1 between student and teacher decoded boxes
        s_reg = student_logits["reg"]
        t_reg = teacher_logits["reg"].detach()
        kd_reg = F.l1_loss(s_reg, t_reg, reduction="mean")
        loss += kd_reg

        return loss

    def forward(
        self,
        student_feats=None,
        teacher_feats=None,
        student_logits=None,
        teacher_logits=None,
        adapt_layers=None,
    ):
        total_loss = 0.0
        feat_loss = torch.tensor(0.0)
        logit_loss = torch.tensor(0.0)

        if self.strategy in ("feature", "both") and student_feats is not None:
            feat_loss = self.feature_distillation_loss(
                student_feats, teacher_feats, adapt_layers
            )
            total_loss += self.feature_weight * feat_loss

        if self.strategy in ("logit", "both") and student_logits is not None:
            logit_loss = self.logit_distillation_loss(student_logits, teacher_logits)
            total_loss += self.logit_weight * logit_loss

        return total_loss, feat_loss, logit_loss


class DistillationYOLOX(nn.Module):
    """
    Teacher-Student wrapper for YOLOX distillation training.

    The teacher model is frozen (no gradients). During the forward pass,
    both teacher and student process the same input; the distillation loss
    is computed from their FPN features and/or head logits.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        distill_loss: DistillationLoss,
        distill_weight: float = 1.0,
        student_channels=None,
        teacher_channels=None,
    ):
        """
        Args:
            student: student YOLOX model (trainable)
            teacher: teacher YOLOX model (frozen)
            distill_loss: DistillationLoss instance
            distill_weight: global scaling factor for the distillation loss
            student_channels: list of FPN output channels for student [P3, P4, P5]
            teacher_channels: list of FPN output channels for teacher [P3, P4, P5]
        """
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.distill_loss = distill_loss
        self.distill_weight = distill_weight

        for p in self.teacher.parameters():
            p.requires_grad = False

        if student_channels and teacher_channels:
            self.adapt_layers = nn.ModuleList([
                FeatureAdaptation(sc, tc)
                for sc, tc in zip(student_channels, teacher_channels)
            ])
        else:
            self.adapt_layers = None

    def forward(self, x, targets=None):
        student_fpn = self.student.backbone(x)
        with torch.no_grad():
            teacher_fpn = self.teacher.backbone(x)

        if self.training:
            assert targets is not None

            student_logits = self._extract_head_logits(self.student.head, student_fpn)
            with torch.no_grad():
                teacher_logits = self._extract_head_logits(self.teacher.head, teacher_fpn)

            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.student.head(
                student_fpn, targets, x
            )

            distill_total, feat_loss, logit_loss = self.distill_loss(
                student_feats=student_fpn,
                teacher_feats=teacher_fpn,
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                adapt_layers=self.adapt_layers,
            )

            total_loss = loss + self.distill_weight * distill_total

            outputs = {
                "total_loss": total_loss,
                "det_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "distill_loss": distill_total,
                "feat_distill_loss": feat_loss,
                "logit_distill_loss": logit_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.student.head(student_fpn)

        return outputs

    def _extract_head_logits(self, head, fpn_outs):
        """
        Run the head conv layers to extract raw cls/obj/reg logits
        for distillation (without computing detection loss).
        """
        cls_list, obj_list, reg_list = [], [], []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(head.cls_convs, head.reg_convs, head.strides, fpn_outs)
        ):
            x = head.stems[k](x)

            cls_feat = cls_conv(x)
            cls_output = head.cls_preds[k](cls_feat)

            reg_feat = reg_conv(x)
            reg_output = head.reg_preds[k](reg_feat)
            obj_output = head.obj_preds[k](reg_feat)

            b, _, h, w = cls_output.shape
            cls_list.append(cls_output.view(b, -1, h * w).permute(0, 2, 1))
            obj_list.append(obj_output.view(b, -1, h * w).permute(0, 2, 1))
            reg_list.append(reg_output.view(b, -1, h * w).permute(0, 2, 1))

        return {
            "cls": torch.cat(cls_list, dim=1),
            "obj": torch.cat(obj_list, dim=1),
            "reg": torch.cat(reg_list, dim=1),
        }

    @property
    def backbone(self):
        return self.student.backbone

    @property
    def head(self):
        return self.student.head
