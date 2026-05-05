#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Playground."""

# noinspection PyUnusedImports
import mon
from mon import Path

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


a = Path("/Volumes/ssd_01/01_longpham/_/code/mon/projects/aic26_06/src/ecdetseg/configs/ecdet_s_coco.yaml")


class A:

    @classmethod
    def build(cls):
        print(cls.__name__)


class B(A):
    pass


B.build()
