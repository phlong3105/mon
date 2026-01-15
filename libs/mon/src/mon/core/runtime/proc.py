#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CLI argument parsing and preparation utilities.

This module provides helpers to build, parse, and merge CLI arguments.
"""

from __future__ import annotations

__all__ = [
    "parse_cli_args",
    "parse_default_args",
    "parse_predict_args",
    "parse_train_args",
]

import argparse
import socket

import box

from mon.core.console import console
from mon.core.device import parse_device
from mon.core.dtypes import image as I
from mon.core.pathlib import Path
from mon.core.utils import merge_dicts
from .core import CLI_OPTIONS
from .menu_cli import RunCLI
from .resolve import (
    load_config,
    parse_config_file,
    parse_save_dir,
    parse_weights,
)


# ==============================================================================
# region CONTROL
# ==============================================================================

def parse_default_args(name: str = "main") -> box.Box:
    """Build and parse default CLI arguments.

    Construct an argparse.ArgumentParser from ``CLI_OPTIONS`` and return
    parsed arguments.

    Args:
        name: Program description used in the ArgumentParser. Defaults to "main".

    Returns:
        Parsed arguments as a box.Box.
    """
    parser = argparse.ArgumentParser(description=name)

    for opt_name, opt_params in CLI_OPTIONS.items():
        if opt_params.get("prompt_only", False):
            continue

        action = opt_params.get("action", "store")
        kwargs = {
            "action"  : action,
            "help"    : opt_params.get("help", ""),
            "required": opt_params.get("required", False),
        }

        # Boolean actions (store_true/store_false) do not take 'type' or 'choices'
        if action in ["store_true", "store_false"]:
            # Default is usually False for store_true, True for store_false
            kwargs["default"] = opt_params.get("default", action == "store_false")
        else:
            if "type" in opt_params:
                kwargs["type"] = opt_params["type"]
            if "choices" in opt_params:
                kwargs["choices"] = opt_params["choices"]
            kwargs["default"] = opt_params.get("default", None)

        flag = f"--{opt_name.replace('_', '-')}"
        parser.add_argument(flag, **kwargs)

    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional arguments")
    return box.Box(vars(parser.parse_args()))


def parse_cli_args(
    cli : box.Box    | None = None,
    root: Path | str | None = None,
    name: str               = "main"
) -> box.Box:
    """Parse CLI arguments and optionally run the interactive prompt.

    Launch RunCLI to gather values if the ``p`` flag is present.

    Args:
        cli: Pre-parsed CLI arguments. Defaults to None.
        root: Project root to attach to parsed arguments. Defaults to None.
        name: Program description for the parser. Defaults to "main".

    Returns:
        Normalized CLI arguments.
    """
    # Initialize CLI if not provided
    cli = cli or parse_default_args(name)

    # Path Normalization
    # Prioritize root passed to function, then root in cli, then current working dir
    raw_root = root or cli.get("root") or Path.cwd()
    cli.root = Path(raw_root).normalize()

    # Interactive Switch
    # Assuming 'p' is the flag for --prompt
    if cli.get("p", False):
        # RunCLI should return a updated box.Box
        cli   = RunCLI(cli).prompt()
        cli.p = False  # Prevent re-triggering

    return cli


def parse_train_args(
    cli       : box.Box    | None = None,
    root      : Path | str | None = None,
    model_root: Path | str | None = None,
    verbose   : bool              = False
) -> box.Box:
    """Parse and prepare training arguments.

    Merge ``cli`` and configuration values, resolve paths and devices, and
    prepare the save directory.

    Args:
        cli: CLI arguments. Defaults to None.
        root: Project root path. Defaults to None.
        model_root: Model root path for configuration resolution. Defaults to None.
        verbose: Verbosity mode. Defaults to False.

    Returns:
        Finalized training arguments.
    """
    # Resolve CLI and Config Path
    cli         = parse_cli_args(cli, root=root)
    config_path = parse_config_file(cli.config, cli.root, model_root=model_root)

    # Load and Merge
    args = load_config(config_path, verbose=verbose)
    args = merge_dicts(args, cli)  # Prioritize cli -> args

    # Name and Directory Resolution
    args.fullname = args.fullname or args.model or "unnamed_run"

    if not args.save_dir:
        base_run_dir  = args.root / "run" / "train"
        # Determine subdir based on user preference
        subdir        = args.fullname if args.use_fullname else args.data
        args.save_dir = parse_save_dir(base_run_dir, args.arch, args.model, subdir)
    else:
        args.save_dir = Path(args.save_dir)

    # Resource Resolution
    args.hostname = socket.gethostname().lower()
    args.device   = parse_device(args.device)
    # Resolve all potential weight paths
    for key in ["weights", "resume", "tuning"]:
        if key in args:
            args[key] = parse_weights(
                root        = args.root,
                weights     = args[key],
                num_classes = args.num_classes,
            )

    # Save Directory Preparation (Atomic & Safe)
    if args.save_dir.exists() and not args.exist_ok:
        args.save_dir.rmdir(recursive=True)

    args.save_dir.mkdir(parents=True, exist_ok=True)

    # Artifact Logging
    if config_path and config_path.exists():
        # Copying the config to the run dir ensures reproducibility
        config_path.copy_to(dst=args.save_dir / config_path.name)
        args.cli = config_path

    if verbose:
        console.log(f"[green]Run directory:[/green] {args.save_dir}")

    return args


def parse_predict_args(
    cli       : box.Box    | None = None,
    root      : Path | str | None = None,
    model_root: Path | str | None = None,
    verbose   : bool              = False
) -> box.Box:
    """Parse and prepare prediction arguments.

    Merge ``cli`` and configuration values, resolve devices and weights, and
    adjust image size.

    Args:
        cli: CLI arguments. Defaults to None.
        root: Project root path. Defaults to None.
        model_root: Model root path for configuration resolution. Defaults to None.
        verbose: Verbosity mode. Defaults to False.

    Returns:
        Finalized prediction arguments.
    """
    # Resolve CLI and Config Path
    cli         = parse_cli_args(cli, root=root)
    config_path = parse_config_file(cli.config, cli.root, model_root=model_root)

    # Load and Merge
    args = load_config(cli.config, verbose=verbose)
    args = merge_dicts(args, cli)  # Prioritize cli -> args

    # Name and Directory Resolution
    args.fullname = args.fullname or args.model or "unnamed_prediction"

    if not args.save_dir:
        base_run_dir  = args.root / "run" / "predict"
        # Determine subdir grouping
        subdir        = args.fullname if (args.use_fullname or args.save_nearby) else args.data
        args.save_dir = parse_save_dir(base_run_dir, args.arch, args.model, subdir)
    else:
        args.save_dir = Path(args.save_dir)

    # Resource Resolution
    args.hostname = socket.gethostname().lower()
    args.device   = parse_device(args.device)
    # Resolve all potential weight paths
    for key in ["weights", "resume", "tuning"]:
        if key in args:
            args[key] = parse_weights(
                root        = args.root,
                weights     = args[key],
                num_classes = args.num_classes,
            )
    # Ensure imgsz is a list/tuple of [H, W] or a single int normalized to [H, W]
    args.imgsz = I.imgsz(args.imgsz)

    # Save Logic (Conditional for Inference)
    # Only create directories if we actually intend to save something and aren't saving 'nearby' the source
    should_save = any([args.save_result, args.save_image, args.save_debug])

    if not args.save_nearby and should_save:
        if args.save_dir.exists() and not args.get("exist_ok", False):
            args.save_dir.rmdir(recursive=True)

        args.save_dir.mkdir(parents=True, exist_ok=True)

        # Artifact Logging
        if config_path and config_path.exists():
            # Copying the config to the run dir for reproducibility of prediction settings
            config_path.copy_to(dst=args.save_dir / config_path.name)
            cli.config = config_path

    if verbose:
        console.log(f"[green]Run directory:[/green] {args.save_dir}")

    return args

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
