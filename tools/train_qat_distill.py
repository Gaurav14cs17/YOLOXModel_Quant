#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
YOLOX Combined QAT + Knowledge Distillation training entry point.

Usage:
    python tools/train_qat_distill.py \
        -f exps/default/yolox_s_qat_distill.py \
        -d 1 -b 16 --fp16 \
        teacher_ckpt /path/to/yolox_l.pth

The student is prepared for QAT while being guided by a full-precision
teacher through feature + logit distillation.
"""

import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import launch
from yolox.exp import Exp, check_exp_value, get_exp
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices


def make_parser():
    parser = argparse.ArgumentParser("YOLOX QAT + Distillation Training")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument("--dist-backend", default="nccl", type=str)
    parser.add_argument("--dist-url", default=None, type=str)
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-d", "--devices", default=None, type=int)
    parser.add_argument("-f", "--exp_file", default=None, type=str,
                        help="experiment description file")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("-c", "--ckpt", default=None, type=str,
                        help="student checkpoint (for resuming)")
    parser.add_argument("-e", "--start_epoch", default=None, type=int)
    parser.add_argument("--num_machines", default=1, type=int)
    parser.add_argument("--machine_rank", default=0, type=int)
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true")
    parser.add_argument("--cache", type=str, nargs="?", const="ram")
    parser.add_argument("-o", "--occupy", dest="occupy", default=False, action="store_true")
    parser.add_argument("-l", "--logger", type=str, default="tensorboard",
                        help="Logger: tensorboard, mlflow, wandb")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    return parser


@logger.catch
def main(exp: Exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting."
        )

    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = exp.get_trainer(args)
    trainer.train()


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    check_exp_value(exp)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.cache is not None:
        exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
