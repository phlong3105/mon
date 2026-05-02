#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Playground."""

# noinspection PyUnusedImports
import mon
from mon import Path

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


a = Path("/Volumes/ssd_01/01_longpham/_/code/mon/projects/aic26_06/src/ecdetseg/configs/ecdet_s_coco.yaml")
b = "./ecdet/ecdet_s.yaml"
c = a.parent / b
print(c)
print(c.is_config_file())
