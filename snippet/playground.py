#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from __future__ import annotations

import torch

from mon import core

console = core.console

a = torch.Tensor([[10, 10, 10, 10]])
print(len(a))

# t = timeit.Timer("import mon.foundation")
# print(t.timeit(number = 1000000))
