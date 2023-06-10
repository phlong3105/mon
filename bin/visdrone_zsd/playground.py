#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A playground."""

from __future__ import annotations

import timeit

import numpy as np

from mon import DATA_DIR

# t = timeit.Timer("import mon")
# print(t.timeit(number=1000000))

data_file = DATA_DIR / "visdrone/zsd23/background_seen_and_unseen_word_vec.npy"
out_file  = DATA_DIR / "visdrone/zsd23/background_seen_and_unseen_word_vec.txt"
data      = np.load(data_file, mmap_mode="r")
print(data)