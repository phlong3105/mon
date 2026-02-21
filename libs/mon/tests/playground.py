#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Playground."""
from dataclasses import dataclass, field

# noinspection PyUnusedImports
import mon
from mon import Path

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


"""
output_dir = "/Volumes/ssd_01/10_workspace/11_code/mon/projects/enhance/run/predict/zerodce/zerodce/dicm"
dirname = "pred"
subdirname = ""
src_path = "/Volumes/ssd_01/10_workspace/11_code/mon/projects/enhance/data/dicm/test/image/01.jpg"
print(mon.resolve_save_dir(output_dir, dirname, subdirname, src_path, keep_subdirs=False, near_src=False))
print(mon.resolve_save_dir(output_dir, dirname, subdirname, src_path, keep_subdirs=True,  near_src=False))
print(mon.resolve_save_dir(output_dir, dirname, subdirname, src_path, keep_subdirs=False, near_src=True))
print(mon.resolve_save_dir(output_dir, dirname, subdirname, src_path, keep_subdirs=True,  near_src=True))
"""


"""
weights1 = mon.Path("/Volumes/ssd_01/10_workspace/11_code/mon/zoo/cv/classify/mobileone/mobileone_s0/imagenet1k_v1/mobileone_s0_imagenet1k_v1.pth.tar")
weights2 = mon.Path("/Volumes/ssd_01/10_workspace/11_code/mon/projects/enhance/run/train/clode/clode/clode_siceme/last.pt")
root = mon.ZOO_ROOT
print(weights1.unique_path_from(root).truncate())
print(weights2.unique_path_from(root).truncate())
"""


"""
config_manager = mon.ConfigManager(root=current_dir, config_file="alexnet_v2.yaml")
config = config_manager.config_for("predict")
print(config)
"""


@dataclass
class Foo:
    a: int
    _a: int = field(init=False, repr=False)

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a = value


foo = Foo(a=3)
bar = Foo(a=2)
print(foo.a)
print(bar.a)
