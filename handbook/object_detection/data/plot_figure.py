#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from one.io import is_json_file
from one.io import load

x_units       = {
    "ms" : "(ms)",
    "fps": "(FPS)",
}
acceptable_xs = {
    "speed_v100_b1" : "V100 batch 1",
    "speed_v100_b32": "V100 batch 32",
}
acceptable_ys = {
    "map_val"   : "COCO AP val",
    "map50_val" : "COCO AP50 val",
    "map75_val" : "COCO AP75 val",
    "maps_val"  : "COCO AP small val",
    "mapm_val"  : "COCO AP medium val",
    "mapl_val"  : "COCO AP large val",
    "map_test"  : "COCO AP test",
    "map50_test": "COCO AP50 test",
    "map75_test": "COCO AP75 test",
    "maps_test" : "COCO AP small test",
    "mapm_test" : "COCO AP medium test",
    "mapl_test" : "COCO AP large test"
}


# MARK: - Functional

def plot_sota(opt):
    data = load(opt.data)
    for model, values in data.items():
        points     = values["points"]
        linestyle  = values["linestyle"]
        marker     = values["marker"]
        show_label = values["show_label"]
        m          = []
        x          = []
        y          = []
        for k, v in points.items():
            m.append(k)
            if opt.x_unit == "ms":
                x.append(v[opt.x])
            elif opt.x_unit == "fps":
                x.append(1.0 / (v[opt.x] / 1000))
            else:
                raise ValueError
            y.append(v[opt.y])
        plt.plot(
            x, y,
            label      = model,
            linestyle  = linestyle,
            marker     = marker,
            markersize = 6,
        )
        if show_label:
            for x, y, m in zip(x, y, m):
                plt.text(x, y, m, fontsize=8)

    plt.title("Object Detection SOTA")
    plt.xlabel(f"{acceptable_xs[opt.x]} {x_units[opt.x_unit]}")
    plt.ylabel(acceptable_ys[opt.y])
    if opt.x_unit == "fps":
        ax = plt.gca()
        ax.invert_xaxis()
    plt.legend()
    if opt.save:
        plt.savefig("object_detection_sota.png")
    plt.show()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data.json",     type=str)
    parser.add_argument("--x",      default="speed_v100_b1", type=str)
    parser.add_argument("--y",      default="map_val",       type=str)
    parser.add_argument("--x_unit", default="ms",            type=str)
    parser.add_argument("--save",   default=True,            type=bool)
    opt = parser.parse_args()

    if not is_json_file(opt.data):
        raise RuntimeError(f"`data` cannot be read from: {opt.data}.")
    if opt.x not in acceptable_xs:
        raise RuntimeError(f"`x` axis must be one of: {acceptable_xs.keys()}. "
                           f"But got: {opt.x}.")
    if opt.y not in acceptable_ys:
        raise RuntimeError(f"`x` axis must be one of: {acceptable_ys.keys()}. "
                           f"But got: {opt.y}.")
    if opt.x_unit not in x_units:
        raise RuntimeError(f"`x_unit` axis must be one of: {x_units.keys()}. "
                           f"But got: {opt.x_unit}.")
    plot_sota(opt)
