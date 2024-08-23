# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements visualization functions.

https://www.geeksforgeeks.org/concatenate-images-using-opencv-in-python/
"""

from __future__ import annotations

__all__ = [
    "get_grid_size",
    "move_figure",
    "plt",
]

import math

import matplotlib
from matplotlib import pyplot as plt

from mon import core

console = core.console


# mpl.use("wxAgg")

plt.ion()
plt.show()
# plt.switch_backend("qt6agg")
plt.rcParams["savefig.bbox"] = "tight"


# region Window Positioning

def get_grid_size(n: int, nrow: int = 4) -> list[int]:
    """Calculate the number of rows and columns needed to display the items
    in a grid.
    
    Args:
        n: The number of items.
        nrow: The number of items in a row. The final grid size is
            `(n / nrow, nrow)`. If ``None``, put all items in a single
            row. Default: ``4``.
    
    Returns:
        A :obj:`tuple` of `(nrows, ncols)`, where nrows is the number of
        rows and ncols is the number of columns.
    """
    if isinstance(nrow, int) and nrow > 0:
        ncols = nrow
    else:
        ncols = n
    nrows = math.ceil(n / ncols)
    return [nrows, ncols]


def move_figure(x: int, y: int):
    """Move the matplotlib figure around the window. The upper-left corner to
    the location specified by `(x, y)`.
    """
    mngr = plt.get_current_fig_manager()
    fig  = plt.gcf()
    backend = matplotlib.get_backend()
    if backend == "TkAgg":
        mngr.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == "WXAgg":
        mngr.window.SetPosition((x, y))
    else:  # This works for QT and GTK. You can use window.setGeometry
        mngr.window.move(x, y)


# endregion
