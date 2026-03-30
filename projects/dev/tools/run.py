#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Running Script.

This script provides a CLI for running a model on a given dataset.
"""

from __future__ import annotations

__all__ = []

import argparse
import sys

from mon.core import (
    ConfigContext,
    Path,
    PREDICTORS,
    resolve_project_root,
    RunMode,
    Task,
    TRAINERS,
)

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    if args.train:
        config_ctx = ConfigContext.from_cli(
            root=resolve_project_root(current_dir),
            config_file=args.config,
            task=Task.LLE,
            mode=RunMode.TRAIN,
            device="auto",
            save=True,
            save_debug=True,
            exist_ok=True,
            verbose=True,
        )
        config = config_ctx.config_for(RunMode.TRAIN, prompt=args.prompt)
        trainer = TRAINERS.build(name=config.model_name, config=config)
        trainer.train()
    elif args.predict:
        config_ctx = ConfigContext.from_cli(
            root=resolve_project_root(current_dir),
            config_file=args.config,
            task=Task.LLE,
            mode=RunMode.PREDICT,
            device="auto",
            save=True,
            save_debug=True,
            exist_ok=True,
            verbose=True,
        )
        config = config_ctx.config_for(RunMode.PREDICT, prompt=args.prompt)
        predictor = PREDICTORS.build(name=config.model_name, config=config)
        predictor.predict()
    else:
        raise NotImplementedError("Run mode hasn't been implemented.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("main")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--test", action="store_true", help="Test the model.")
    parser.add_argument("--predict", action="store_true", help="Predict using the model.")
    parser.add_argument("--prompt", action="store_true", default="True", help="Prompt for additional inputs.")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    main(parse_args())

# endregion
