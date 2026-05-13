#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# QAT Trainer for YOLOX

import os
import time
import datetime
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.data import DataPrefetcher
from yolox.exp import Exp
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    WandbLogger,
    MlflowLogger,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    mem_usage,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize,
)
from yolox.models.qat import (
    enable_fake_quant,
    disable_fake_quant,
    enable_observer,
    disable_observer,
    convert_qat_to_quantized,
)

from .trainer import Trainer


class QATTrainer(Trainer):
    """
    Extends the base YOLOX Trainer with QAT lifecycle management.

    QAT schedule (epoch-based):
      [0, qat_start_epoch):           warmup — fake-quant OFF, observers ON (calibration)
      [qat_start_epoch, freeze_epoch): QAT   — fake-quant ON,  observers ON
      [freeze_epoch, max_epoch):       freeze — fake-quant ON,  observers OFF

    After training, call `export_quantized_model()` to convert to int8.
    """

    def __init__(self, exp: Exp, args):
        super().__init__(exp, args)
        self.qat_start_epoch = getattr(exp, "qat_start_epoch", 0)
        self.observer_freeze_epoch = getattr(exp, "observer_freeze_epoch", -1)

    def before_epoch(self):
        super().before_epoch()

        model = self.model.module if is_parallel(self.model) else self.model
        qat_model = model.model if hasattr(model, "model") else model

        if self.epoch >= self.qat_start_epoch:
            enable_fake_quant(qat_model)
            logger.info(f"Epoch {self.epoch + 1}: FakeQuant ENABLED")
        else:
            disable_fake_quant(qat_model)
            logger.info(f"Epoch {self.epoch + 1}: FakeQuant DISABLED (calibration)")

        if self.observer_freeze_epoch > 0 and self.epoch >= self.observer_freeze_epoch:
            disable_observer(qat_model)
            logger.info(f"Epoch {self.epoch + 1}: Observers FROZEN")
        else:
            enable_observer(qat_model)

    def after_train(self):
        super().after_train()
        if self.rank == 0:
            self.export_quantized_model()

    def export_quantized_model(self):
        """Convert QAT model to fully quantized int8 and save."""
        logger.info("Converting QAT model to quantized int8...")
        eval_model = self.ema_model.ema if self.use_model_ema else self.model
        if is_parallel(eval_model):
            eval_model = eval_model.module

        float_model = eval_model.model if hasattr(eval_model, "model") else eval_model
        quantized = convert_qat_to_quantized(float_model, inplace=False)

        save_path = os.path.join(self.file_name, "quantized_model.pth")
        torch.save({"model": quantized.state_dict()}, save_path)
        logger.info(f"Quantized model saved to {save_path}")

        return quantized
