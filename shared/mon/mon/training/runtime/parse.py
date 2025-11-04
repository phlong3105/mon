#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements interactive CLI."""

__all__ = [
    "parse_cli_args",
    "parse_default_args",
    "parse_predict_args",
    "parse_train_args",
]

import argparse
import socket

import box

from mon.core.device import parse_device
from mon.core.dtypes import image as I
from mon.core.pathlib import Path
from mon.core.utils import merge_dicts
from .options import CLI_OPTIONS
from .utils import (
    load_config,
    parse_config_file,
    parse_save_dir,
    parse_weights_file,
)
from .menu_rich import RunCLI


# ----- Parser -----
def parse_default_args(name: str = "main") -> dict | box.Box:
    """Parse direct CLI."""
    parser = argparse.ArgumentParser(description=name)
    
    for opt_name, opt_params in CLI_OPTIONS.items():
        action      = opt_params.get("action",      "store")
        default     = opt_params.get("default",     None)
        opt_type    = opt_params.get("type",        None)
        choices     = opt_params.get("choices",     None)
        required    = opt_params.get("required",    False)
        help_text   = opt_params.get("help",        "")
        prompt_only = opt_params.get("prompt_only", False)  # Use in interactive CLI only, not parse_args
        
        if prompt_only:
            continue
        '''
        if opt_type == bool and default is None:
            default = False
        if action == "store_true" and default is None:
            default = False
        if action == "store_false" and default is None:
            default = True
        '''
        
        kwargs = {
            "action"  : action,
            "default" : default,
            "required": required,
            "help"    : help_text,
        }
        if action in ["store_true", "store_false"]:
            kwargs.pop("default")
            # kwargs["default"] = False if action == "store_true" else True
        if opt_type:
            kwargs["type"] = opt_type
        if choices:
            kwargs["choices"] = choices
        flag = f"--{opt_name.replace('_', '-')}"
        parser.add_argument(flag, **kwargs)

    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional arguments")
    return box.Box(vars(parser.parse_args()))


def parse_cli_args(cli: box.Box = None, root: Path = None, name: str= "main") -> dict | box.Box:
    """Parse arguments from either direct CLI call or interactive prompt.
    
    Args:
        cli: Either a dict/Box of arguments or None to parse from CLI.
        root: Project root directory to use if not specified in CLI. Default: ``None``.
        name: Name of the program to display in the interactive prompt. Default: ``"main"``.
    """
    cli      = cli      or parse_default_args(name)  # Direct CLI
    cli.root = cli.root or root
    cli.root = Path(cli.root) if cli.root else None
    if cli.p:  # Interactive CLI
        cli   = RunCLI(cli).prompt_args()
        cli.p = False  # Disable prompt flag after use
    return cli


# ----- Parse Args -----
def parse_train_args(
    cli       : box.Box = None,
    root      : Path    = None,
    model_root: Path    = None,
    verbose   : bool    = False
) -> dict | box.Box:
    """Parse arguments for training."""
    # CLI
    cli        = parse_cli_args(cli, root=root)
    cli.config = parse_config_file(cli.config, cli.root, model_root=model_root)
    
    # Args
    args = load_config(cli.config, verbose=verbose)
    args = merge_dicts(args, cli)  # Prioritize cli -> args
   
    if args.fullname in [None, "None", ""]:
        args.fullname = args.model
        
    if args.save_dir in [None, ""]:
        if args.use_fullname:
            args.save_dir = parse_save_dir(args.root/"run"/"train", args.arch, args.model, args.fullname)
            # args.save_dir = _utils.parse_save_dir(root/"run"/"train", arch, fullname, None)
        else:
            args.save_dir = parse_save_dir(args.root/"run"/"train", args.arch, args.model, args.data)
    else:
        args.save_dir = Path(args.save_dir)
        # if str("run/train") not in str(args.save_dir):
        #     args.save_dir = pathlib.Path(f"run/train/{args.save_dir}")
        # if str(args.root) not in str(args.save_dir):
        #     args.save_dir = args.root / args.save_dir

    args.hostname = socket.gethostname().lower()
    args.weights  = parse_weights_file(args.root, args.weights)
    args.resume   = parse_weights_file(args.root, args.resume)
    args.tuning   = parse_weights_file(args.root, args.tuning)
    args.device   = parse_device(args.device)

    # Save config file
    if not args.exist_ok:
        args.save_dir.rmdir()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.config and args.config.is_config_file():
        args.config.copy_to(dst=args.save_dir / f"{args.config.name}")

    return args


def parse_predict_args(
    cli       : box.Box = None,
    root      : Path    = None,
    model_root: Path    = None,
    verbose   : bool    = False
) -> dict | box.Box:
    """Parse arguments from either CLI or config file for predicting."""
    # CLI
    cli        = parse_cli_args(cli, root=root)
    cli.config = parse_config_file(cli.config, cli.root, model_root=model_root)
    
    # Args
    args = load_config(cli.config, verbose=verbose)
    args = merge_dicts(args, cli)  # Prioritize cli -> args
    
    if args.fullname in [None, "None", ""]:
        args.fullname = args.model
        
    if args.save_dir in [None, ""]:
        if args.use_fullname or args.save_nearby:
            args.save_dir = parse_save_dir(args.root/"run"/"predict", args.arch, args.fullname, None)
        else:
            args.save_dir = parse_save_dir(args.root/"run"/"predict", args.arch, args.model, args.data)
    else:
        args.save_dir = Path(args.save_dir)
        # args.save_dir = args.save_dir.replace("run/train/", "")
        # if str("run/predict") not in str(args.save_dir):
        #     args.save_dir = pathlib.Path(f"run/predict/{args.save_dir}")
        # if str(args.root) not in str(args.save_dir):
        #     args.save_dir = args.root / args.save_dir
    
    args.hostname = socket.gethostname().lower()
    args.weights  = parse_weights_file(args.root, args.weights)
    args.resume   = parse_weights_file(args.root, args.resume)
    args.tuning   = parse_weights_file(args.root, args.tuning)
    args.device   = parse_device(args.device)
    args.imgsz    = I.imgsz(args.imgsz)
    
    # Save config file
    if not args.exist_ok:
        args.save_dir.rmdir()
    if not args.save_nearby and (args.save_result or args.save_image or args.save_debug):
        args.save_dir.mkdir(parents=True, exist_ok=True)
        if args.config and args.config.is_config_file():
            args.config.copy_to(dst=args.save_dir / f"{args.config.name}")

    return args
