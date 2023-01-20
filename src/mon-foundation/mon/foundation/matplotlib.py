#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module extends the :mod:`matplotlib` package.
"""

from __future__ import annotations

__all__ = [
    "get_grid_size", "matplotlib", "move_figure", "plt",
]

import matplotlib
from matplotlib import pyplot as plt

from mon.foundation import math


# plt.ion()
# plt.show()
# plt.switch_backend("qt5agg")
# plt.rcParams["savefig.bbox"] = "tight"


# region Window Positioning

def get_grid_size(n: int, nrow: int | None = 4) -> tuple[int, int]:
    """Calculate the number of rows and columns needed to display the items
    in a grid.
    
    Args:
        n: The number of items.
        nrow: The number of items in a row. The final grid size is
            (:param:`n` / :param:`nrow`, :param:`nrow`). If None, put all items
            in a single row. Defaults to 4.
    
    Returns:
        A tuple (nrows, ncols), where nrows is the number of rows and ncols is
        the number of columns.
    """
    if isinstance(nrow, int) and nrow > 0:
        ncols = nrow
    else:
        ncols = n
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def move_figure(x: int, y: int):
    """Move the matplotlib figure around the window. The upper-left corner to
    the location speficied by :param:`(x,  y)`.
    """
    mngr    = plt.get_current_fig_manager()
    fig     = plt.gcf()
    backend = matplotlib.get_backend()
    
    if backend == "TkAgg":
        mngr.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == "WXAgg":
        mngr.window.SetPosition((x, y))
    else:  # This works for QT and GTK. You can use window.setGeometry
        mngr.window.move(x, y)

# endregion
