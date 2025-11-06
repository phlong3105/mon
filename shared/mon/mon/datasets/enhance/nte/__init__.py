#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements nighttime enhancement (NTE) datasets.

Notes:
    Low-light enhancement (LLE) and nighttime enhancement are related but distinct.
    LLE aims to improve general low-light images by boosting brightness, reducing
    noise, and enhancing overall visibility.
    Nighttime image enhancement is a specialized form of LLIE that specifically
    addresses unique challenges present in nighttime scenes, such as severe
    low light, glare, glow, and uneven illumination.
"""

from .darkface import *
from .exdark import *
from .gta5nighttimefog import *
from .lolistreet import *
from .nightcity import *
from .realnighthaze import *
from .ydld import *
