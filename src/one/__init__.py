#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
                              _
                           _ooOoo_
                          o8888888o
                          88" . "88
                          (| -_- |)
                          O\  =  /O
                       ____/`---'\____
                     .'  \\|     |//  `.
                    /  \\|||  :  |||//  \
                   /  _||||| -:- |||||_  \
                   |   | \\\  -  /'| |   |
                   | \_|  `\`---'//  |_/ |
                   \  .-\__ `-. -'__/-.  /
                 ___`. .'  /--.--\  `. .'___
              ."" '<  `.___\_<|>_/___.' _> \"".
             | | :  `- \`. ;`. _/; .'/ /  .' ; |
             \  \ `-.   \_\_`. _.'_/_/  -' _.' /
   ===========`-.`___`-.__\ \___  /__.-'_.'_.-'================
                           `=--=-'


Notes:
	1. This package is `Tensor[C, H, W]` first. At least for all deep learning
	   related tasks.
	2. Operations on Tensors should support batch processing as default,
	   i.e, 4D shape [B, C, H, W].
	3. Naming parameter:
		a) input / output : For basic functions. Ex: conversion,
		b)
"""

from __future__ import annotations

"""
from .core import *
from .data import *
from .math import *
from .nn import *
from .utils import *
from .vision import *
"""

__author__  = "Long H. Pham"
__version__ = "1.1.0"
