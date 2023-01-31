#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from __future__ import annotations

import mon
from mon import core

console = core.console

print(mon.Path().absolute())

# t = timeit.Timer("import mon.foundation")
# print(t.timeit(number = 1000000))
