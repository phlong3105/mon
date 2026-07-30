#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Polar Chart.

This module provides functionality for plotting polar charts using Matplotlib.
"""

from __future__ import annotations

__all__ = [
    "plot_polar_bar_chart",
    "plot_radar_chart",
]

import numpy as np
from cycler import cycler
from matplotlib import pyplot as plt

from mon import ColorPalette
from mon.plot.base import (
    Axes,
    Axis,
    get_figsize,
    plot_marker_labels,
    plot_polar_marker_labels,
    Series,
    set_plot_style,
    set_spines,
)


# ==============================================================================
# region VISUALIZATION
# ==============================================================================

def plot_radar_chart(ax: plt.Axes, axes: Axes, *args, **kwargs):
    """Plot a default polar radar chart with multiple series.

    Args:
        ax (plt.Axes): The Matplotlib Axes object to plot on.
        axes (Axes): An instance of the Axes class containing the data and
            configuration for the plot.
        *args: Additional positional arguments to pass to the plotting function.
        **kwargs: Additional keyword arguments to pass to the plotting function.
    """
    # 1. Define data
    X = axes.x.angles()
    categories = axes.x.tick_labels
    # Close the loop for radar chart
    X += X[:1]
    categories += categories[:1]

    # Prioritize axes colors, then default color cycle
    colors = axes.colors or plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = ColorPalette.from_any(colors=colors).cycler
    labelsize = plt.rcParams["ytick.labelsize"]

    # 2. Create canvas
    # Spines
    set_spines(ax, axes.spines) if axes.spines else None
    # Grid
    # ax.grid(**axes.grid_kw)
    # Axis
    ax.set_theta_offset(np.pi / 2)
    ax.set_rlabel_position(0)
    ax.set_ylim(*axes.y.limits) if axes.y.limits else None

    # 3. Plot data
    for i, series in enumerate(axes.data):
        # Data
        Y = series.Y
        Y += Y[:1]  # Close the loop for radar chart

        color = series.color or next(colors)
        text_color = color if series.text_color is True else series.text_color

        # Plot
        plot_kw = series.line_kw | {
            "color": color,
        }
        ax.plot(X, Y, **plot_kw)
        ax.fill(X, Y, alpha=series.alpha)

        # 3.4. Markers
        plot_marker_labels(ax=ax, X=X, series=series, color=text_color)

    # 4. Customize plot
    # Title
    ax.set_title(axes.title) if axes.title else None
    # Ticks
    ax.set_xticks(X, categories)
    ax.set_yticks(axes.y.ticks, axes.y.tick_labels, ha="center")
    ax.tick_params(axis="y", labelsize=labelsize - 1)
    # Legend
    ax.legend(**axes.legend.legend_kw) if axes.legend else None


def plot_polar_bar_chart(ax: plt.Axes, axes: Axes, *args, **kwargs):
    """Plot a default polar bar chart with 1 series.

    Reference:
        - https://python-graph-gallery.com/web-circular-barplot-with-matplotlib/

    Args:
        ax (plt.Axes): The Matplotlib Axes object to plot on.
        axes (Axes): An instance of the Axes class containing the data and
            configuration for the plot.
        *args: Additional positional arguments to pass to the plotting function.
        **kwargs: Additional keyword arguments to pass to the plotting function.
    """
    # 1. Define data
    series = axes.data[0]
    X = axes.x.angles()
    Y = series.Y
    widths = series.wmap(polar=True)

    # Prioritize series color, then axes colors, then default color cycle
    colors = series.color or axes.colors or plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = ColorPalette.from_any(colors=colors).gradient_cmap(values=Y)
    labelsize = plt.rcParams["ytick.labelsize"] - 1
    text_colors = colors if series.text_color is True else series.text_color

    # 2. Create canvas
    # Spines
    # set_spines(ax, axes.spines) if axes.spines else None
    ax.spines["start"].set_color("none")
    ax.spines["polar"].set_color("none")
    # Grid
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="gray")
    # Axis
    ax.set_theta_offset(np.pi / 2)
    ax.set_rlabel_position(0)
    ax.set_ylim(*axes.y.limits) if axes.y.limits else None

    # 3. Plot data
    bar_kw = series.bar_kw | {
        "zorder": 10,
        "bottom": 0.0,
        "width": widths,
        "color": colors,
        "alpha": series.alpha,
    }
    ax.bar(X, Y, **bar_kw)

    _, ymax = axes.y.limits
    ax.vlines(X, 0, ymax, zorder=9, linestyles=(0, (4, 4)), linewidth=0.5, color="black")

    plot_polar_marker_labels(ax=ax, X=X, series=series, zorder=11, fontsize=labelsize, color=text_colors)

    # 4. Customize plot
    # Title
    ax.set_title(axes.title, pad=15) if axes.title else None
    # Ticks
    ax.set_xticks(X, axes.x.tick_labels)
    ax.set_yticks(axes.y.ticks, axes.y.tick_labels, ha="center")
    ax.tick_params(axis="y", labelsize=labelsize)
    # ax.set_yticklabels([])
    # Legend
    ax.legend(**axes.legend.legend_kw) if axes.legend else None

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    # 1. Setup figure
    set_plot_style(font_size=7)
    fig, ax = plt.subplots(
        figsize=get_figsize("single", height_ratio=0.7),
        subplot_kw=dict(polar=True),
    )

    # 2. Define data
    axes = Axes(
        title="Sample Polar Chart",
        x=Axis(
            label="Categories",
            ticks=[1, 2, 3, 4, 5],
            tick_labels=["Cat 1", "Cat 2", "Cat 3", "Cat 4", "Cat 5"],
        ),
        y=Axis(
            label="Scores",
            ticks=[1, 2, 3, 4, 5],
            limits=(0, 5.5),
        ),
        data=[
            Series(
                label="A",
                Y=[5, 3, 1, 1, 1],
                alpha=0.5,
                ha="center", va="center", ho=0.0, vo=0.3,
            ),
            Series(
                label="B",
                Y=[3, 2, 5, 3, 4],
                width=0.6,
                alpha=0.5,
                ha="center", va="center", ho=0.0, vo=0.3,
            ),
        ],
        legend=None,
    )

    # 3. Plot
    plot_radar_chart(ax, axes)

    # 4. Save
    plt.tight_layout()
    plt.savefig("polar_bar.pdf", bbox_inches="tight", format="pdf")
    plt.show()

# endregion
