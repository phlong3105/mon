#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import mon


a = mon.Path("/home/longpham/10_workspace/11_code/mon/data/enhance/llie/lol_blur/train/lq/0001.png")
b = mon.Path("lol_blur")
c = mon.Path("new_dir")
d = a.relative_path_from_part(b)
e = c / d.parent / "gray"
print(d)
print(e)
print(c.name)
