#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Export a QAT-trained YOLOX model to a fully quantized int8 model.

Usage:
    python tools/export_quantized.py \
        -f exps/default/yolox_s_qat.py \
        -c /path/to/qat_trained_ckpt.pth \
        --output quantized_yolox_s.pth

    # Export to TorchScript for deployment:
    python tools/export_quantized.py \
        -f exps/default/yolox_s_qat.py \
        -c /path/to/qat_trained_ckpt.pth \
        --output quantized_yolox_s.pt \
        --torchscript
"""

import argparse
from loguru import logger

import torch

from yolox.exp import get_exp
from yolox.utils import configure_module


def make_parser():
    parser = argparse.ArgumentParser("Export Quantized YOLOX")
    parser.add_argument("-f", "--exp_file", type=str, required=True)
    parser.add_argument("-c", "--ckpt", type=str, required=True,
                        help="QAT-trained checkpoint")
    parser.add_argument("--output", type=str, default="quantized_model.pth")
    parser.add_argument("--torchscript", action="store_true",
                        help="Also export as TorchScript")
    parser.add_argument("-n", "--name", type=str, default=None)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser


def main():
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    if args.opts:
        exp.merge(args.opts)

    model = exp.get_model()

    logger.info(f"Loading QAT checkpoint from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)

    from yolox.models.qat import convert_qat_to_quantized

    inner = model.model if hasattr(model, "model") else model
    quantized = convert_qat_to_quantized(inner, inplace=False)

    torch.save({"model": quantized.state_dict()}, args.output)
    logger.info(f"Quantized model weights saved to {args.output}")

    if args.torchscript:
        quantized.eval()
        quantized.head.decode_in_inference = True

        ts_path = args.output.replace(".pth", ".pt")
        dummy_input = torch.randn(1, 3, *exp.test_size)
        traced = torch.jit.trace(quantized, dummy_input)
        traced.save(ts_path)
        logger.info(f"TorchScript quantized model saved to {ts_path}")


if __name__ == "__main__":
    configure_module()
    main()
