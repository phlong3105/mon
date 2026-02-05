#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model runner.

This module provides a runner for training and predicting with various machine
learning models.
"""

from __future__ import annotations

__all__ = [
    "ModelRunner",
]

import copy
import os
import subprocess

import box

import mon
from mon.core import (
    is_valid_str,
    MODELS,
    parse_cli_args,
    parse_default_args,
    parse_device,
    Path,
    resolve_config_file,
    resolve_model_dir,
    to_list,
    to_str,
)

mon.preload()

current_file = Path(__file__).normalize(exist=True)
current_dir  = current_file.parents[0]


# ==============================================================================
# region CONTROL
# ==============================================================================

class ModelRunner:
    """A runner for extracting box masks from images."""

    # --- Lifecycle & Initialization ---
    def __init__(self, cfg: box.Box):
        self._cfg    = cfg
        self.verbose = cfg.verbose

    # --- Callable & Context Manager ---
    def run(self):
        """Run according to the specified mode."""
        mode = self._cfg.mode

        if mode in ["train"]:
            self._run_train()
        elif mode in ["predict", "speed"]:
            self._run_predict()
        else:
            raise ValueError(f"Unknown mode: {mode}.")

    def _run_train(self):
        """Run training."""
        # Parse arguments
        cfg         = copy.deepcopy(self._cfg)
        cfg.root    = Path(cfg.root).normalize()
        model_root  = resolve_model_dir(cfg.arch, cfg.model)
        cfg.config  = resolve_config_file(cfg.config, cfg.root, model_root=model_root)
        cfg.weights = to_str(cfg.weights, ",")

        if is_valid_str(cfg.fullname):
            cfg.fullname = Path(cfg.config).stem

        # Prepare kwargs and flags
        kwargs, flags = {}, []
        kwargs |= {"--root"           : str(cfg.root)}
        kwargs |= {"--task"           : str(cfg.task)}
        kwargs |= {"--mode"           : cfg.mode}
        kwargs |= {"--arch"           : cfg.arch}
        kwargs |= {"--model"          : cfg.model}
        kwargs |= {"--config"         : cfg.config}
        # kwargs |= {"--data"           : cfg.data}
        kwargs |= {"--fullname"       : cfg.fullname}
        kwargs |= {"--save-dir"       : str(cfg.save_dir)}
        kwargs |= {"--weights"        : cfg.weights}
        kwargs |= {"--device"         : cfg.device}
        kwargs |= {"--seed"           : cfg.seed}
        # kwargs |= {"--imgsz"          : cfg.imgsz}
        kwargs |= {"--epochs"         : cfg.epochs}
        kwargs |= {"--batch-size"     : cfg.batch_size}
        flags  += ["--torchrun"]     if cfg.torchrun     else []
        flags  += ["--save-result"]  if cfg.save_result  else []
        flags  += ["--save-image"]   if cfg.save_image   else []
        flags  += ["--save-debug"]   if cfg.save_debug   else []
        flags  += ["--use-fullname"] if cfg.use_fullname else []
        flags  += ["--keep-subdirs"] if cfg.keep_subdirs else []
        flags  += ["--save-nearby"]  if cfg.save_nearby  else []
        flags  += ["--exist-ok"]     if cfg.exist_ok     else []
        flags  += ["--verbose"]      if cfg.verbose      else []

        # Parse script file
        python_call = ["python"]
        env         = {**os.environ}
        script_file = MODELS[cfg.arch][cfg.model].model_dir / "train.py"
        if cfg.torchrun:
            device_     = parse_device(cfg.device)
            python_call = [
                "python", "-m", "torch.distributed.run",
                f"--nproc_per_node={len(device_)}",
                f"--master_port={cfg.master_port}",
                f"--master_addr={cfg.master_addr}",
            ]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(device_), **env}

        # Parse arguments
        args_call: list[str] = []
        for k, v in kwargs.items():
            if v is None:
                continue
            elif isinstance(v, list | tuple):
                args_call_ = [f"{k}={v_}" for v_ in v]
            else:
                args_call_ = [f"{k}={v}"]
            args_call += args_call_

        # Run training
        if script_file.is_py_file():
            print("\n")
            command = (
                python_call +
                [script_file] +
                args_call +
                flags
            )
            subprocess.run(command, cwd=current_dir, env=env)
        else:
            raise FileNotFoundError(f"Cannot find Python training script file at: {script_file}.")

    def _run_predict(self):
        """Run prediction."""
        # Parse arguments
        cfg         = copy.deepcopy(self._cfg)
        cfg.root    = Path(cfg.root).normalize()
        model_root  = resolve_model_dir(cfg.arch, cfg.model)
        cfg.data    = to_list(cfg.data)
        cfg.config  = resolve_config_file(cfg.config, cfg.root, model_root=model_root)
        cfg.config  = cfg.config or ""
        cfg.weights = to_str(cfg.weights, ",")

        if is_valid_str(cfg.fullname):
            cfg.fullname = cfg.model

        # Prepare kwargs and flags
        for d in cfg.data:
            kwargs, flags = {}, []
            kwargs |= {"--root"           : str(cfg.root)}
            kwargs |= {"--task"           : str(cfg.task)}
            kwargs |= {"--mode"           : cfg.mode}
            kwargs |= {"--arch"           : cfg.arch}
            kwargs |= {"--model"          : cfg.model}
            kwargs |= {"--config"         : cfg.config}
            kwargs |= {"--data"           : d}
            kwargs |= {"--fullname"       : cfg.fullname}
            kwargs |= {"--save-dir"       : str(cfg.save_dir)}
            kwargs |= {"--weights"        : cfg.weights}
            kwargs |= {"--device"         : cfg.device}
            kwargs |= {"--seed"           : cfg.seed}
            kwargs |= {"--imgsz"          : cfg.imgsz}
            flags  += ["--resize"]       if cfg.resize       else []
            flags  += ["--benchmark"]    if cfg.benchmark    else []
            flags  += ["--save-result"]  if cfg.save_result  else []
            flags  += ["--save-image"]   if cfg.save_image   else []
            flags  += ["--save-debug"]   if cfg.save_debug   else []
            flags  += ["--use-fullname"] if cfg.use_fullname else []
            flags  += ["--keep-subdirs"] if cfg.keep_subdirs else []
            flags  += ["--save-nearby"]  if cfg.save_nearby  else []
            flags  += ["--exist-ok"]     if cfg.exist_ok     else []
            flags  += ["--verbose"]      if cfg.verbose      else []

            # Parse script file
            script_file = MODELS[cfg.arch][cfg.model].model_dir / "predict.py"
            python_call = ["python"]

            # Parse arguments
            args_call: list[str] = []
            for k, v in kwargs.items():
                if v is None:
                    continue
                elif isinstance(v, list | tuple):
                    args_call_ = [f"{k}={v_}" for v_ in v]
                else:
                    args_call_ = [f"{k}={v}"]
                args_call += args_call_

            # Run prediction
            if script_file.is_py_file():
                print("\n")
                command = (
                    python_call +
                    [script_file] +
                    args_call +
                    flags
                )
                subprocess.run(command, cwd=current_dir)
            else:
                raise FileNotFoundError(f"Cannot find Python predicting script file at: {script_file}.")

    # --- CLI ---
    @staticmethod
    def parse_args() -> box.Box:
        """Parse command line arguments.

        Returns:
            Parsed arguments.
        """
        cli   = parse_default_args()
        cli.p = True  # With prompt
        return parse_cli_args(cli=cli, name="run_model")

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    args   = ModelRunner.parse_args()
    runner = ModelRunner(args)
    runner.run()

# endregion
