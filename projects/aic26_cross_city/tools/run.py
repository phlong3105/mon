#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Running Script.

This script provides a CLI for running a model on a given dataset.
"""

from __future__ import annotations

__all__ = []

import argparse
import sys

import ecdetseg
from mon.core import Path, resolve_project_root, RunMode, Task

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region FUNCTIONS
# ==============================================================================

def train(args: argparse.Namespace):
    trainer = ecdetseg.ECDetSeg_Trainer.from_cli(
        root=resolve_project_root(current_dir),
        config_file=args.config,
        task=Task.DETECT,
        mode=RunMode.TRAIN,
        device="auto",
        save=True,
        save_debug=True,
        exist_ok=True,
        verbose=True,
        prompt=args.prompt,
    )
    trainer.train()


def predict(args: argparse.Namespace):
    pass


def metric(args: argparse.Namespace):
    pass


def benchmark(args: argparse.Namespace):
    pass

# endregion


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    """A hub for running models."""
    if args.train:
        # Train
        train(args)
    elif args.predict:
        # Predict
        predict(args)
    elif args.metric:
        # Metric
        metric(args)
    elif args.benchmark:
        # Benchmark
        benchmark(args)
    else:
        # Error
        raise NotImplementedError("Run mode hasn't been implemented.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("main")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--test", action="store_true", help="Test the model.")
    parser.add_argument("--predict", action="store_true", help="Predict using the model.")
    parser.add_argument("--metric", action="store_true", help="Evaluate the model.")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark the model.")
    parser.add_argument("--prompt", action="store_true", help="Prompt for additional inputs.")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    main(parse_args())

# endregion
