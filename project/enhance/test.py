#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script crops images."""

from __future__ import annotations


import mon


a = mon.Path("/Volumes/ssd_01/10_workspace/11_code/mon/data/enhance/llie/lol_v1/train/lq/00000.png")
b = "enhance"
c = a.relative_path(b)
print(c)
