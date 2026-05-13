#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Quantization Aware Training support for YOLOX

import copy

import torch
import torch.nn as nn
import torch.quantization as tq
from torch.quantization import (
    QConfig,
    default_qat_qconfig,
    get_default_qat_qconfig,
)

from .yolox import YOLOX
from .yolo_pafpn import YOLOPAFPN
from .yolo_head import YOLOXHead
from .network_blocks import BaseConv, DWConv


class QuantizableBaseConv(nn.Module):
    """BaseConv with explicit quantization stubs for residual-add fusion."""

    def __init__(self, base_conv: BaseConv):
        super().__init__()
        self.conv = base_conv.conv
        self.bn = base_conv.bn
        self.act = base_conv.act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def _fuse_baseconv_modules(model: nn.Module):
    """
    Fuse Conv2d + BatchNorm2d pairs inside every BaseConv / DWConv for QAT.
    Returns a list of module-path lists suitable for torch.quantization.fuse_modules.
    """
    fuse_list = []
    for name, module in model.named_modules():
        if isinstance(module, BaseConv):
            fuse_list.append([f"{name}.conv", f"{name}.bn"])
    return fuse_list


def prepare_qat_model(
    model: YOLOX,
    backend: str = "fbgemm",
    custom_qconfig: QConfig = None,
    inplace: bool = False,
):
    """
    Convert a standard YOLOX model into a QAT-ready model.

    Steps:
        1. Set qconfig on the model
        2. Fuse Conv-BN pairs
        3. Insert fake-quantize observers via prepare_qat

    Args:
        model: a YOLOX instance (already loaded with pretrained weights)
        backend: quantization backend ("fbgemm" for server, "qnnpack" for mobile)
        custom_qconfig: optional override; defaults to per-backend QAT qconfig
        inplace: modify model in-place or return a copy
    Returns:
        The QAT-prepared model (still float, with FakeQuantize modules)
    """
    if not inplace:
        model = copy.deepcopy(model)

    qconfig = custom_qconfig or get_default_qat_qconfig(backend)
    model.qconfig = qconfig

    # Fusion requires eval mode
    model.eval()
    fuse_list = _fuse_baseconv_modules(model)
    if fuse_list:
        torch.quantization.fuse_modules(model, fuse_list, inplace=True)

    # prepare_qat requires train mode
    model.train()
    torch.quantization.prepare_qat(model, inplace=True)

    return model


def convert_qat_to_quantized(model: nn.Module, inplace: bool = False):
    """
    Convert a QAT-trained model to a fully quantized model for inference.
    """
    if not inplace:
        model = copy.deepcopy(model)

    model.eval()
    torch.quantization.convert(model, inplace=True)
    return model


def disable_fake_quant(model: nn.Module):
    """Disable all FakeQuantize modules (useful during warmup epochs)."""
    for mod in model.modules():
        if isinstance(mod, (torch.quantization.FakeQuantize, torch.quantization.FakeQuantizeBase)):
            mod.disable_fake_quant()


def enable_fake_quant(model: nn.Module):
    """Re-enable all FakeQuantize modules after warmup."""
    for mod in model.modules():
        if isinstance(mod, (torch.quantization.FakeQuantize, torch.quantization.FakeQuantizeBase)):
            mod.enable_fake_quant()


def disable_observer(model: nn.Module):
    """Freeze observers (stop updating quantization statistics)."""
    for mod in model.modules():
        if isinstance(mod, (torch.quantization.FakeQuantize, torch.quantization.FakeQuantizeBase)):
            mod.disable_observer()


def enable_observer(model: nn.Module):
    """Enable observers (resume updating quantization statistics)."""
    for mod in model.modules():
        if isinstance(mod, (torch.quantization.FakeQuantize, torch.quantization.FakeQuantizeBase)):
            mod.enable_observer()


class QATModel(nn.Module):
    """
    Wrapper around YOLOX that manages QAT lifecycle.

    Provides a unified interface to:
      - build the float model
      - prepare for QAT
      - control observer / fake-quant enable/disable per epoch
      - convert to fully quantized model after training

    Typical QAT schedule:
      epoch 0..W-1   : warmup   — fake-quant OFF, observers ON  (calibration)
      epoch W..F-1   : fine-tune — fake-quant ON,  observers ON
      epoch F..end   : freeze   — fake-quant ON,  observers OFF
    """

    def __init__(
        self,
        float_model: YOLOX,
        backend: str = "fbgemm",
        qat_start_epoch: int = 0,
        observer_freeze_epoch: int = -1,
    ):
        super().__init__()
        self.backend = backend
        self.qat_start_epoch = qat_start_epoch
        self.observer_freeze_epoch = observer_freeze_epoch

        self.model = prepare_qat_model(float_model, backend=backend, inplace=False)

        if qat_start_epoch > 0:
            disable_fake_quant(self.model)

    def forward(self, x, targets=None):
        return self.model(x, targets)

    def update_qat_schedule(self, current_epoch: int):
        """Call at the start of each epoch to update QAT state."""
        if current_epoch >= self.qat_start_epoch:
            enable_fake_quant(self.model)
        else:
            disable_fake_quant(self.model)

        if self.observer_freeze_epoch > 0 and current_epoch >= self.observer_freeze_epoch:
            disable_observer(self.model)
        else:
            enable_observer(self.model)

    def convert(self):
        """Convert to fully quantized model after training."""
        return convert_qat_to_quantized(self.model, inplace=False)

    @property
    def backbone(self):
        return self.model.backbone

    @property
    def head(self):
        return self.model.head
