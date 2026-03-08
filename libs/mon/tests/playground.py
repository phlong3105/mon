#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Playground."""

# noinspection PyUnusedImports
import mon
from mon import Path

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


a = mon.to_list(
    "as/dsa, dsad/sad, asds/adas, da/sda"
)
print(a)
