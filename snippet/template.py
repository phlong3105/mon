#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Template.
"""

from __future__ import annotations

import argparse

from munch import Munch


# H1: - Functional -------------------------------------------------------------

def func(args: Munch | dict):
    pass


# H1: - Main -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg1", type=str, default="", help="Descriptions for arg1")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    input_args = vars(parse_args())
    arg1       = input_args.get("arg1", None)
   
    args = Munch(
        arg1 = arg1,
    )
    func(args=args)
