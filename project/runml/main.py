#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from __future__ import annotations

import subprocess

import click

import mon
import utils

_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]
modes_ 	      = ["train", "predict", "online", "instance", "metric", "plot"]


# region Train

def run_train(args: dict):
    # Get user input
    root     = mon.Path(args["root"])
    task     = args["task"]
    mode     = args["mode"]
    config   = args["config"]
    weights  = args["weights"]
    model    = args["model"]
    fullname = args["fullname"]
    save_dir = args["save_dir"]
    device   = args["device"]
    epochs   = args["epochs"]
    steps    = args["steps"]
    exist_ok = args["exist_ok"]
    verbose  = args["verbose"]
    
    assert root.exists()
    
    # Parse arguments
    config   = mon.parse_config_file(project_root=root, config=config)
    assert config not in [None, "None", ""]
    fullname = fullname if fullname not in [None, "None", ""] else config.stem
    save_dir = save_dir or root / "run" / "train" / fullname
    weights  = mon.to_str(weights, ",")
    
    kwargs   = {
        "--root"    : str(root),
        "--config"  : config,
        "--weights" : weights,
        "--model"   : model,
        "--fullname": fullname,
        "--save-dir": str(save_dir),
        "--device"  : device,
        "--epochs"  : epochs,
        "--steps"   : steps,
    }
    flags    = ["--exist-ok"] if exist_ok else []
    flags   += ["--verbose"]  if verbose  else []
    
    # Parse script file
    if model in mon.MODELS:
        script_file = _current_dir / "train.py"
        python_call = ["python"]
    elif model in mon.MODELS_EXTRA:
        torch_distributed_launch = mon.MODELS_EXTRA[model]["torch_distributed_launch"]
        script_file = mon.MODELS_EXTRA[model]["model_dir"] / "my_train.py"
        devices     = mon.parse_device(device)
        if isinstance(devices, list) and torch_distributed_launch:
            python_call = [
                f"python",
                f"-m",
                f"torch.distributed.launch",
                f"--nproc_per_node={str(len(devices))}",
                f"--master_port=9527"
            ]
        else:
            python_call = ["python"]
    else:
        raise ValueError(f"Cannot find Python training script file.")
    
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
        result = subprocess.run(command, cwd=_current_dir)
        print(result)
    else:
        raise ValueError(f"Cannot find Python training script file at: {script_file}.")
    
# endregion


# region Predict

def run_predict(args: dict):
    # Get user input
    root         = mon.Path(args["root"])
    task         = args["task"]
    mode         = args["mode"]
    model        = args["model"]
    config       = args["config"]
    weights	     = args["weights"]
    data         = args["data"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    device       = args["device"]
    imgsz        = args["imgsz"]
    resize       = args["resize"]
    benchmark    = args["benchmark"]
    save_image   = args["save_image"]
    use_data_dir = args["use_data_dir"]
    verbose      = args["verbose"]
    
    assert root.exists()
    
    # Parse arguments
    config   = mon.parse_config_file(project_root=root, config=config)
    config   = config or "default"
    # assert config not in [None, "None", ""]
    fullname = fullname if fullname not in [None, "None", ""] else model
    weights  = mon.to_str(weights, ",")
    
    for d in data:
        if use_data_dir:
            save_dir = save_dir or mon.DATA_DIR / task.value / "#predict" / model
        else:
            save_dir = save_dir or root / "run" / "predict" / model
        kwargs  = {
            "--root"    : str(root),
            "--config"  : config,
            "--weights" : weights,
            "--data"    : d,
            "--model"   : model,
            "--fullname": fullname,
            "--save-dir": str(save_dir),
            "--device"  : device,
            "--imgsz"   : imgsz,
        }
        flags   = ["--resize"]     if resize     else []
        flags  += ["--benchmark"]  if benchmark  else []
        flags  += ["--save-image"] if save_image else []
        flags  += ["--verbose"]    if verbose    else []
        
        # Parse script file
        if model in mon.MODELS:
            script_file = _current_dir / "predict.py"
            python_call = ["python"]
        elif model in mon.MODELS_EXTRA:
            torch_distributed_launch = mon.MODELS_EXTRA[model]["torch_distributed_launch"]
            script_file = mon.MODELS_EXTRA[model]["model_dir"] / "my_predict.py"
            python_call = ["python"]
        else:
            raise ValueError(f"Cannot find Python training script file.")
        
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
            result = subprocess.run(command, cwd=_current_dir)
            print(result)
        else:
            raise ValueError(f"Cannot find Python predicting script file at: {script_file}.")
        
# endregion


# region Online

def run_online(args: dict):
    # Get user input
    root         = mon.Path(args["root"])
    task         = args["task"]
    mode         = args["mode"]
    model        = args["model"]
    config       = args["config"]
    weights	     = args["weights"]
    data         = args["data"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    device       = args["device"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    imgsz        = args["imgsz"]
    resize       = args["resize"]
    benchmark    = args["benchmark"]
    save_image   = args["save_image"]
    use_data_dir = args["use_data_dir"]
    verbose      = args["verbose"]
    
    assert root.exists()
    
    # Parse arguments
    config   = mon.parse_config_file(project_root=root, config=config)
    assert config not in [None, "None", ""]
    fullname = fullname if fullname not in [None, "None", ""] else config.stem
    weights  = mon.to_str(weights, ",")
    
    for d in data:
        if use_data_dir:
            save_dir = save_dir or mon.DATA_DIR / task.value / "#predict" / model
        else:
            save_dir = save_dir or root / "run" / "predict" / model
        kwargs  = {
            "--root"    : str(root),
            "--config"  : config,
            "--weights" : weights,
            "--data"    : d,
            "--model"   : model,
            "--fullname": fullname,
            "--save-dir": str(save_dir),
            "--device"  : device,
            "--imgsz"   : imgsz,
        }
        flags   = ["--resize"]     if resize     else []
        flags  += ["--benchmark"]  if benchmark  else []
        flags  += ["--save-image"] if save_image else []
        flags  += ["--verbose"]    if verbose    else []
        
        # Parse script file
        if model in mon.MODELS:
            script_file = _current_dir / "online.py"
            python_call = ["python"]
        elif model in mon.MODELS_EXTRA:
            torch_distributed_launch = mon.MODELS_EXTRA[model]["torch_distributed_launch"]
            script_file = mon.MODELS_EXTRA[model]["model_dir"] / "my_online.py"
            python_call = ["python"]
        else:
            raise ValueError(f"Cannot find Python online learning script file.")
        
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
            result = subprocess.run(command, cwd=_current_dir)
            print(result)
        else:
            raise ValueError(f"Cannot find Python predicting script file at: {script_file}.")

# endregion


# region Main

@click.command(name="main", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--root",     type=click.Path(exists=True), help="Project root.")
@click.option("--task",     type=str, default=None,       help="Running task.")
@click.option("--mode",     type=str, default="predict",  help="Running mode.")
@click.option("--model",    type=str, default=None,       help="Running model.")
@click.option("--config",   type=str, default=None,   	  help="Running config.")
@click.option("--data",     type=str, default=None,       help="Predict dataset.")
@click.option("--save-dir", type=str, default=None,       help="Optional saving directory.")
@click.option("--weights",  type=str, default=None,       help="Weights paths.")
@click.option("--device",   type=str, default=None,       help="Running devices.")
@click.option("--epochs",   type=int, default=-1,   	  help="Training epochs.")
@click.option("--steps",    type=int, default=-1,   	  help="Training steps.")
@click.option("--imgsz",    type=int, default=512,        help="Image size.")
def main(
    root    : str,
    task    : str,
    mode    : str,
    model   : str,
    config  : str,
    data    : str,
    save_dir: str,
    weights : str,
    device  : int | list[int] | str,
    epochs  : int,
    steps   : int,
    imgsz   : int,
):
    click.echo(click.style(f"\nInput Prompt:", fg="white", bg="red", bold=True))
    
    # Task
    tasks_     = utils.list_tasks(project_root=root)
    tasks_str_ = utils.parse_menu_string(tasks_)
    task       = click.prompt(click.style(f"Task {tasks_str_}", fg="bright_green", bold=True), default=task)
    task       = tasks_[int(task)] if mon.is_int(task) else task
    # Mode
    mode       = click.prompt(click.style(f"Mode {utils.parse_menu_string(modes_)}", fg="bright_green", bold=True), default=mode)
    mode       = modes_[int(mode)] if mon.is_int(mode) else mode
    
    if mode in ["train", "predict", "online", "instance"]:
        # Model
        models_      = utils.list_models(project_root=root, mode=mode, task=task)
        models_str_  = utils.parse_menu_string(models_)
        model	     = click.prompt(click.style(f"Model {models_str_}", fg="bright_green", bold=True), type=str, default=model)
        model 	     = models_[int(model)] if mon.is_int(model) else model
        # Config     
        configs_     = utils.list_configs(project_root=root, model=model)
        configs_str_ = utils.parse_menu_string(configs_)
        config	     = click.prompt(click.style(f"Config {configs_str_}", fg="bright_green", bold=True), type=str, default="")
        config       = configs_[int(config)] if mon.is_int(config) else config
        # Weights    
        weights_     = utils.list_weights_files(project_root=root, model=model, config=config)
        weights_str_ = utils.parse_menu_string(weights_)
        weights      = click.prompt(click.style(f"Weights {weights_str_}", fg="bright_green", bold=True), type=str, default=weights or "")
        weights      = weights if weights not in [None, ""] else None
        if weights is not None:
            if isinstance(weights, str):
                weights = mon.to_list(weights)
            weights = [weights_[int(w)] if mon.is_int(w) else w for w in weights]
            weights = [w.replace("'", "") for w in weights]
        # Predict data
        if mode in ["predict", "online", "instance"]:
            data_     = utils.list_datasets(project_root=root, task=task, mode="predict")
            data_str_ = utils.parse_menu_string(data_)
            data      = data.replace(",", ",\n    ") if isinstance(data, str) else data
            data	  = click.prompt(click.style(f"Predict(s) {data_str_}", fg="bright_green", bold=True), type=str, default=data)
            data 	  = mon.to_list(data)
            data 	  = [data_[int(d)] if mon.is_int(d) else d for d in data]
        # Fullname
        fullname    = mon.Path(config).stem
        fullname    = click.prompt(click.style(f"Save name: {fullname}", fg="bright_green", bold=True), type=str, default=fullname)
        # Device
        devices_    = mon.list_devices()
        devices_str = utils.parse_menu_string(devices_)
        device      = "auto" if model in utils.list_mon_models(mode=mode, task=task) and mode == "train" else device
        device      = click.prompt(click.style(f"Device {devices_str}", fg="bright_green", bold=True), type=str, default=device or "cuda:0")
        device 	    = devices_[int(device)] if mon.is_int(device) else device
        # Training Flags
        if mode in ["train", "online", "instance"]:
            epochs = click.prompt(click.style(f"Epochs              ", fg="bright_yellow", bold=True), type=int, default=epochs)
            steps  = click.prompt(click.style(f"Steps               ", fg="bright_yellow", bold=True), type=int, default=steps)
        # Predict Flags
        if mode in ["predict", "online", "instance"]:
            # Image size
            imgsz_       = imgsz
            imgsz        = click.prompt(click.style(f"Image size          ", fg="bright_yellow", bold=True), type=str, default=imgsz)
            imgsz        = mon.to_int_list(imgsz)
            imgsz        = imgsz[0] if len(imgsz) == 1 else imgsz
            # Resize
            resize       = "yes" if imgsz != imgsz_ else "no"
            resize       = click.prompt(click.style(f"Resize?     [yes/no]", fg="bright_yellow", bold=True), type=str, default=resize)
            # Other Flags
            benchmark    = click.prompt(click.style(f"Benchmark?  [yes/no]", fg="bright_yellow", bold=True), type=str, default="no")
            save_image   = click.prompt(click.style(f"Save image? [yes/no]", fg="bright_yellow", bold=True), type=str, default="yes")
            use_data_dir = click.prompt(click.style(f"Data dir?   [yes/no]", fg="bright_yellow", bold=True), type=str, default="no")
            resize       = True if resize       == "yes" else False
            benchmark    = True if benchmark    == "yes" else False
            save_image   = True if save_image   == "yes" else False
            use_data_dir = True if use_data_dir == "yes" else False
        # Common Flags
        exist_ok = click.prompt(click.style(f"Exist OK?   [yes/no]", fg="bright_yellow", bold=True), type=str, default="yes")
        exist_ok = True if exist_ok == "yes" else False
        verbose  = click.prompt(click.style(f"Verbosity?  [yes/no]", fg="bright_yellow", bold=True), type=str, default="yes")
        verbose  = True if verbose  == "yes" else False
    
    # Run
    if mode in ["train"]:
        args = {
            "root"    : root,
            "task"    : task,
            "mode"    : mode,
            "config"  : config,
            "weights" : weights,
            "model"   : model,
            "fullname": fullname,
            "save_dir": save_dir,
            "device"  : device,
            "epochs"  : epochs,
            "steps"   : steps,
            "exist_ok": exist_ok,
            "verbose" : verbose,
        }
        run_train(args=args)
    elif mode in ["predict"]:
        args = {
            "root"        : root,
            "task"        : task,
            "mode"        : mode,
            "config"      : config,
            "weights"     : weights,
            "model"       : model,
            "data"        : data,
            "fullname"    : fullname,
            "save_dir"    : save_dir,
            "device"      : device,
            "imgsz"       : imgsz,
            "resize" 	  : resize,
            "benchmark"   : benchmark,
            "save_image"  : save_image,
            "use_data_dir": use_data_dir,
            "verbose"     : verbose,
        }
        run_predict(args=args)
    elif mode in ["online", "instance"]:
        args = {
            "root"        : root,
            "task"        : task,
            "mode"        : mode,
            "config"      : config,
            "weights"     : weights,
            "model"       : model,
            "data"        : data,
            "fullname"    : fullname,
            "save_dir"    : save_dir,
            "device"      : device,
            "epochs"      : epochs,
            "steps"       : steps,
            "imgsz"       : imgsz,
            "resize" 	  : resize,
            "benchmark"   : benchmark,
            "save_image"  : save_image,
            "use_data_dir": use_data_dir,
            "verbose"     : verbose,
        }
        run_online(args=args)
    else:
        raise ValueError(
            f":param:`mode` must be one of ``'train'``, ``'predict'``, "
            f"``'online'``, ``'instance'``, ``'metric'``, or ``'plot'``, "
            f"but got {mode}."
        )
        

if __name__ == "__main__":
    main()

# endregion
