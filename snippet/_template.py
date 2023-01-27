#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This is a template."""

from __future__ import annotations

import argparse

import munch
from munch import Munch


# region Function

def func(args: munch.Munch):
    pass

# endregion


# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg1", type=str, default="", help="Descriptions for arg1")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = munch.Munch.fromDict(vars(parse_args()))
    func(args=args)

# endregion
