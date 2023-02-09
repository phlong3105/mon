#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from __future__ import annotations

import timeit

import numpy as np

t = timeit.Timer("import mon")
print(t.timeit(number=1000000))


a = [1, 2, 3]
b = np.array(a)
c = np.array(b)
d1, d2, d3 = c.T
print(d1, d2, d3)
