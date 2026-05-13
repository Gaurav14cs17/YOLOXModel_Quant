#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Knowledge Distillation Trainer for YOLOX

import os
import time
import datetime
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.exp import Exp
from yolox.utils import (
    is_parallel,
    get_model_info,
)

from .trainer import Trainer


class DistillTrainer(Trainer):
    """
    Extends the base YOLOX Trainer for knowledge distillation.

    The experiment's `get_model()` returns a DistillationYOLOX wrapper
    that contains both student and teacher. This trainer handles:
      - Logging distillation-specific losses
      - Proper EMA on the student branch only
      - Evaluation using the student model
    """

    def __init__(self, exp: Exp, args):
        super().__init__(exp, args)

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_iter(self):
        """Extended logging with distillation loss info."""
        if (self.iter + 1) % self.exp.print_interval == 0:
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            from yolox.utils import gpu_mem_usage, mem_usage
            mem_str = "gpu mem: {:.0f}Mb, mem: {:.1f}Gb".format(gpu_mem_usage(), mem_usage())

            logger.info(
                "{}, {}, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    mem_str,
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )

            if self.rank == 0:
                if self.args.logger == "tensorboard":
                    from torch.utils.tensorboard import SummaryWriter
                    self.tblogger.add_scalar(
                        "train/lr", self.meter["lr"].latest, self.progress_in_iter)
                    for k, v in loss_meter.items():
                        self.tblogger.add_scalar(
                            f"train/{k}", v.latest, self.progress_in_iter)
                if self.args.logger == "wandb":
                    metrics = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    metrics.update({"train/lr": self.meter["lr"].latest})
                    self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)
                if self.args.logger == "mlflow":
                    logs = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    logs.update({"train/lr": self.meter["lr"].latest})
                    self.mlflow_logger.on_log(self.args, self.exp, self.epoch + 1, logs)

            self.meter.clear_meters()

        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )


class QATDistillTrainer(DistillTrainer):
    """
    Combined QAT + Distillation trainer.
    Manages both the QAT schedule and distillation training.
    """

    def __init__(self, exp: Exp, args):
        super().__init__(exp, args)
        self.qat_start_epoch = getattr(exp, "qat_start_epoch", 0)
        self.observer_freeze_epoch = getattr(exp, "observer_freeze_epoch", -1)

    def before_epoch(self):
        super().before_epoch()

        model = self.model.module if is_parallel(self.model) else self.model
        student = model.student if hasattr(model, "student") else model

        from yolox.models.qat import (
            enable_fake_quant, disable_fake_quant,
            enable_observer, disable_observer,
        )

        if self.epoch >= self.qat_start_epoch:
            enable_fake_quant(student)
            logger.info(f"Epoch {self.epoch + 1}: Student FakeQuant ENABLED")
        else:
            disable_fake_quant(student)
            logger.info(f"Epoch {self.epoch + 1}: Student FakeQuant DISABLED (calibration)")

        if self.observer_freeze_epoch > 0 and self.epoch >= self.observer_freeze_epoch:
            disable_observer(student)
            logger.info(f"Epoch {self.epoch + 1}: Student Observers FROZEN")
        else:
            enable_observer(student)

    def after_train(self):
        super().after_train()
        if self.rank == 0:
            self.export_quantized_model()

    def export_quantized_model(self):
        """Convert student to quantized int8."""
        logger.info("Converting student QAT model to quantized int8...")
        eval_model = self.ema_model.ema if self.use_model_ema else self.model
        if is_parallel(eval_model):
            eval_model = eval_model.module

        student = eval_model.student if hasattr(eval_model, "student") else eval_model
        from yolox.models.qat import convert_qat_to_quantized
        quantized = convert_qat_to_quantized(student, inplace=False)

        save_path = os.path.join(self.file_name, "quantized_student_model.pth")
        torch.save({"model": quantized.state_dict()}, save_path)
        logger.info(f"Quantized student model saved to {save_path}")
