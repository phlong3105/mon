#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Line Chart.

This module provides functionality for plotting line charts using Matplotlib.
"""

from __future__ import annotations

__all__ = [
    "plot_line_chart",
]

import numpy as np
from matplotlib import pyplot as plt

from mon.core import ColorPalette
from mon.plot.base import (
    Axes,
    Axis,
    get_figsize,
    Legend,
    plot_marker_labels,
    Series,
    set_plot_style,
    set_spines,
    set_ticklabels,
)


# ==============================================================================
# region VISUALIZATION
# ==============================================================================

def plot_line_chart(ax: plt.Axes, axes: Axes, *args, **kwargs):
    """Plot a default line chart with multiple series.

    Args:
        ax (plt.Axes): The Matplotlib Axes object to plot on.
        axes (Axes): An instance of the Axes class containing the data and
            configuration for the plot.
        *args: Additional positional arguments to pass to the plotting function.
        **kwargs: Additional keyword arguments to pass to the plotting function.
    """
    # 1. Define data
    # Prioritize axes colors, then default color cycle
    colors = axes.colors or plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = ColorPalette.from_any(colors=colors).cycler

    # 2. Create canvas
    # Spines
    set_spines(ax, axes.spines) if axes.spines else None
    # Grid
    ax.grid(**axes.grid_kw)
    # Axis
    ax.set_xlim(*axes.x.limits) if axes.x.limits else None
    ax.set_ylim(*axes.y.limits) if axes.y.limits else None

    # 3. Plot data
    for i, series in enumerate(axes.data):
        # Data
        X = axes.x.ticks
        Y = series.Y
        X_clean = [xi for xi, yi in zip(X, Y) if yi is not None]
        Y_clean = [yi for yi in Y if yi is not None]

        color = series.color or next(colors)
        text_color = color if series.text_color is True else series.text_color

        # Plot
        plot_kw = series.line_kw | {
            "color": color,
        }
        ax.plot(X_clean, Y_clean, **plot_kw)

        plot_marker_labels(ax=ax, X=X, series=series, color=text_color)

    # 4. Customize plot
    # Title
    ax.set_title(axes.title) if axes.title else None
    # Ticks
    ax.set_xlabel(axes.x.label) if axes.x.label else None
    ax.set_ylabel(axes.y.label) if axes.y.label else None
    ax.set_xticks(axes.x.ticks, axes.x.tick_labels)
    ax.set_yticks(axes.y.ticks, axes.y.tick_labels)
    set_ticklabels(ax.get_xticklabels(), axes.x)
    set_ticklabels(ax.get_yticklabels(), axes.y)
    # Legend
    ax.legend(**axes.legend.legend_kw) if axes.legend else None

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    # 1. Setup figure
    set_plot_style(font_size=7)
    fig, ax = plt.subplots(figsize=get_figsize("single", height_ratio=0.7))

    # 2. Define data
    X = np.arange(1, 11)
    Y = np.arange(40, 105, 10)
    axes = Axes(
        title="Sample Line Chart",
        x=Axis(
            label="Training Epochs",
            ticks=X,
            limits=(0.5, 10.5),
            grid=True,
        ),
        y=Axis(
            label="mAP (%)",
            ticks=Y,
            limits=(50, 105),
            grid=True,
        ),
        data=[
            Series(
                label="Baseline (ResNet-50)",
                Y=100 - (80 / X) + np.random.normal(0, 1, 10),
                marker="o", linestyle="--",
                ha="center", va="bottom", ho=0.0, vo=1.5,
                text_color=True,
            ),
            Series(
                label="Ours (DetGain-Lite)",
                Y=100 - (50 / X) + np.random.normal(0, 0.5, 10),
                marker="s", linestyle="-",
                ha="center", va="bottom", ho=0.0, vo=1.5,
                text_color=True,
            ),
        ],
        legend=Legend(loc="upper left", bbox_to_anchor=(0, 1)),
    )

    # 3. Plot
    plot_line_chart(ax, axes)

    # 4. Save
    plt.tight_layout()
    plt.savefig("line.pdf", bbox_inches="tight", format="pdf")
    plt.show()

# endregion
