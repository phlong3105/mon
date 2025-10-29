#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements main running pipeline."""

import os
import subprocess

import box

import mon
from mon import Path

mon.dev()

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----
def run_train(args: dict | box.Box):
    # Parse arguments
    args.root    = Path(args.root)
    model_root   = mon.rt.parse_model_dir(args.arch, args.model)
    args.config  = mon.rt.parse_config_file(args.config, args.root, model_root=model_root)
    args.weights = mon.utils.to_str(args.weights, ",")
    
    if args.fullname in [None, "None", ""]:
        args.fullname = Path(args.config).stem
    
    # Prepare kwargs and flags
    kwargs, flags = {}, []
    kwargs |= {"--root"           : str(args.root)}
    kwargs |= {"--task"           : str(args.task)}
    kwargs |= {"--mode"           : args.mode}
    kwargs |= {"--arch"           : args.arch}
    kwargs |= {"--model"          : args.model}
    kwargs |= {"--config"         : args.config}
    # kwargs |= {"--data"           : args.data}
    kwargs |= {"--fullname"       : args.fullname}
    kwargs |= {"--save-dir"       : str(args.save_dir)}
    kwargs |= {"--weights"        : args.weights}
    kwargs |= {"--device"         : args.device}
    kwargs |= {"--seed"           : args.seed}
    # kwargs |= {"--imgsz"          : args.imgsz}
    kwargs |= {"--epochs"         : args.epochs}
    kwargs |= {"--batch-size"     : args.batch_size}
    flags  += ["--torchrun"]     if args.torchrun     else []
    flags  += ["--save-result"]  if args.save_result  else []
    flags  += ["--save-image"]   if args.save_image   else []
    flags  += ["--save-debug"]   if args.save_debug   else []
    flags  += ["--use-fullname"] if args.use_fullname else []
    flags  += ["--keep-subdirs"] if args.keep_subdirs else []
    flags  += ["--save-nearby"]  if args.save_nearby  else []
    flags  += ["--exist-ok"]     if args.exist_ok     else []
    flags  += ["--verbose"]      if args.verbose      else []

    # Parse script file
    python_call = ["python"]
    env         = {**os.environ}
    script_file = mon.MODELS[args.arch][args.model].model_dir / "train.py"
    if args.torchrun:
        device_     = mon.parse_device(args.device)
        python_call = [
            "python", "-m", "torch.distributed.run",
            f"--nproc_per_node={len(device_)}",
            f"--master_port={args.master_port}",
            f"--master_addr={args.master_addr}",
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
        result = subprocess.run(command, cwd=current_dir, env=env)
        print(result)
    else:
        raise ValueError(f"Cannot find Python training script file at: {script_file}.")


# ----- Predict -----
def run_predict(args: dict | box.Box):
    # Parse arguments
    args.root    = Path(args.root)
    model_root   = mon.rt.parse_model_dir(args.arch, args.model)
    args.data    = mon.utils.to_list(args.data)
    args.config  = mon.rt.parse_config_file(args.config, args.root, model_root=model_root)
    args.config  = args.config or ""
    args.weights = mon.utils.to_str(args.weights, ",")
    
    if args.fullname in [None, "None", ""]:
        args.fullname = args.model
    
    # Prepare kwargs and flags
    for d in args.data:
        kwargs, flags = {}, []
        kwargs |= {"--root"           : str(args.root)}
        kwargs |= {"--task"           : str(args.task)}
        kwargs |= {"--mode"           : args.mode}
        kwargs |= {"--arch"           : args.arch}
        kwargs |= {"--model"          : args.model}
        kwargs |= {"--config"         : args.config}
        kwargs |= {"--data"           : d}
        kwargs |= {"--fullname"       : args.fullname}
        kwargs |= {"--save-dir"       : str(args.save_dir)}
        kwargs |= {"--weights"        : args.weights}
        kwargs |= {"--device"         : args.device}
        kwargs |= {"--seed"           : args.seed}
        kwargs |= {"--imgsz"          : args.imgsz}
        flags  += ["--resize"]       if args.resize       else []
        flags  += ["--benchmark"]    if args.benchmark    else []
        flags  += ["--save-result"]  if args.save_result  else []
        flags  += ["--save-image"]   if args.save_image   else []
        flags  += ["--save-debug"]   if args.save_debug   else []
        flags  += ["--use-fullname"] if args.use_fullname else []
        flags  += ["--keep-subdirs"] if args.keep_subdirs else []
        flags  += ["--save-nearby"]  if args.save_nearby  else []
        flags  += ["--exist-ok"]     if args.exist_ok     else []
        flags  += ["--verbose"]      if args.verbose      else []

        # Parse script file
        script_file = mon.MODELS[args.arch][args.model].model_dir / "predict.py"
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
            result = subprocess.run(command, cwd=current_dir)
            print(result)
        else:
            raise ValueError(f"Cannot find Python predicting script file at: {script_file}.")


# ----- Main -----
def main():
    cli   = mon.rt.parse_default_args()
    cli.p = True  # With prompt
    args  = mon.rt.parse_cli_args(cli=cli, name="main")
 
    # Run
    if args.mode in ["train"]:
        run_train(args=args)
    elif args.mode in ["predict", "speed"]:
        run_predict(args=args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}.")


if __name__ == "__main__":
    main()
