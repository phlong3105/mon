#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements installation pipeline."""

from __future__ import annotations

import subprocess

import click

import mon

_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Install

def install(name: str):
    use_extra_model = mon.is_extra_model(name)
    if use_extra_model:
        requirement_file = mon.EXTRA_MODELS[name]["model_dir"] / "requirements.txt"
        if requirement_file.is_txt_file():
            command = f"pip install -r {str(requirement_file)}"
            subprocess.run(command, cwd=_current_dir)
            
# endregion


# region Main
@click.command(name="install", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--name", type=str, default=None, help="Package name.")
def main(name: str) -> str:
    install(name)
    
    
if __name__ == "__main__":
    main()
    
# endregion
