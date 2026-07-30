#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bar Chart.

This module provides functionality for plotting bar charts using Matplotlib.
"""

from __future__ import annotations

__all__ = [
    "plot_stacked_bar_chart",
]

import numpy as np
from matplotlib import pyplot as plt

from mon.core import ColorPalette, wcag_2_contrast
from mon.plot.base import (
    Axes,
    Axis,
    get_figsize,
    Legend,
    Series,
    set_plot_style,
    set_spines,
    set_ticklabels,
)


# ==============================================================================
# region VISUALIZATION
# ==============================================================================

def plot_stacked_bar_chart(ax: plt.Axes, axes: Axes, *args, **kwargs):
    """Plot a default stacked bar chart with multiple series.

    Args:
        ax (plt.Axes): The Matplotlib Axes object to plot on.
        axes (Axes): An instance of the Axes class containing the data and
            configuration for the plot.
        *args: Additional positional arguments to pass to the plotting function.
        **kwargs: Additional keyword arguments to pass to the plotting function.
    """
    # 1. Define data
    # Create a stacking floor vector initialized to zero
    bottom = np.zeros(len(axes.x.ticks))
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
        S = series.S

        color = series.color or next(colors)
        text_color = wcag_2_contrast(color)["text_color"] \
            if series.text_color is True else series.text_color

        # Plot
        bar_kw = series.bar_kw | {
            "bottom": bottom,
            "color": color,
        }
        bars = ax.bar(X, Y, **bar_kw)

        # Add text labels to each bar segment
        for i, bar in enumerate(bars):
            # Calculate the vertical midpoint inside this specific stack segment
            xi = bar.get_x() + (bar.get_width() / 2)
            yi = bar.get_y() + (bar.get_height() / 2)

            if Y[i] > 0:  # Avoid cluttering with zero entries
                text_kwargs = {
                    "x": xi,
                    "y": yi,
                    "s": S[i],
                    "ha": "center",
                    "va": "center",
                    # "color": text_color,
                }
                ax.text(**text_kwargs)

        # Step the floor up by the current segment heights before the next loop
        bottom += Y

    # 2. Draw Axes
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
    axes = Axes(
        title="Sample Stacked Bar Chart",
        x=Axis(
            label="Categories",
            ticks=[1, 2, 3, 4, 5],
            tick_labels=["Cat 1", "Cat 2", "Cat 3", "Cat 4", "Cat 5"],
        ),
        y=Axis(
            label="",
            ticks=np.arange(0, 11),
            limits=(0, 11),
            grid=True,
        ),
        data=[
            Series(
                label="A",
                Y=[1, 1, 1, 1, 1],
                width=0.6,
            ),
            Series(
                label="B",
                Y=[3, 2, 5, 3, 5],
                width=0.6,
            ),
            Series(
                label="C",
                Y=[2, 0, 4, 3, 1],
                width=0.6,
            ),
        ],
        legend=Legend(loc="upper left", bbox_to_anchor=(0, 1)),
    )

    # 3. Plot
    plot_stacked_bar_chart(ax, axes)

    # 4. Save
    plt.tight_layout()
    plt.savefig("stacked_bar.pdf", bbox_inches="tight", format="pdf")
    plt.show()

# endregion
