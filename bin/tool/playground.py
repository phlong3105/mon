#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A playground."""

from __future__ import annotations

import timeit

t = timeit.Timer("import mon")
print(t.timeit(number=1000000))
