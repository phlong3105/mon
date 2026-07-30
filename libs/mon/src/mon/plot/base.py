#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Plotting Classes.

This module provides base plotting classes.

References:
    - https://matplotlib.org/stable/gallery/showcase/anatomy.html
"""

from __future__ import annotations

__all__ = [
    "Axes",
    "Axis",
    "Legend",
    "Series",
    "Spine",
    "get_figsize",
    "plot_marker_labels",
    "plot_polar_marker_labels",
    "set_plot_style",
    "set_spines",
    "set_ticklabels",
]

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from cycler import cycler
from matplotlib import pyplot as plt

from mon.core import (
    COLOR_PALETTES,
    ColorPalette,
    is_list_of,
    K,
    validate_hex_color,
)

# ==============================================================================
# region CONSTANTS
# ==============================================================================

PAGE_WIDTHS = {
    "single": 3.4,  # inches
    "double": 6.8,
    "eccv"  : 4.8,
}
YTICK_POLAR_WIDTH = 0.1 * np.pi  # Width of the Y-tick labels in polar plots

# endregion


# ==============================================================================
# region TYPE DEFINITIONS
# ==============================================================================

Markers = Literal[
    ".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4",
    "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"
]
LineStyles = Literal["-", "--", "-.", ":"]
SpineTypes = Literal["top", "bottom", "left", "right", "start", "polar"]
Locations = Literal[
    "best", "upper right", "upper left", "lower left", "lower right", "right",
    "center left", "center right", "lower center", "upper center", "center",
    # +--------------+--------------+---------------+
    # | 'upper left' |'upper center'| 'upper right' |
    # +--------------+--------------+---------------+
    # |'center left' |   'center'   |'center right' |
    # +--------------+--------------+---------------+
    # | 'lower left' |'lower center'| 'lower right' |
    # +--------------+--------------+---------------+
]
RotationModes = Literal["default", "anchor", "xtick", "ytick"]
Alignments = Literal["left", "center", "right", "bottom", "top"]
TextDirection = Literal["default", "xtick", "ytick"]
FontWeights = Literal["normal", "bold"]

# endregion


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

@dataclass
class Axis:
    """Data class for an axis.

    The directional scale lines setting data limits (X-axis, Y-axis, Z-axis).

    Attributes:
        label (str | None): The text describing the axis. Defaults to None.`
        ticks (Sequence[Any]): The tick marks for the axis.
            Defaults to an empty list.
        limits (tuple[float | int, float | int] | None): The limits of the axis.
            If not provided, use the min and max of the ``ticks``.
            Defaults to None.
        grid (bool): Whether to show grid lines for the axis. Defaults to True.
        tick_labels (list[str]): The labels for the tick marks.
            If not provided, use the ``ticks`` as labels.
            Defaults to an empty list.
        tick_fontweight (FontWeights): The font weight for the tick labels.
            Defaults to "normal".
        tick_colors (str | list[str] | None): The color(s) for the tick labels.
            If not provided, use the default color. Defaults to None.
    """

    label: str | None = None
    ticks: Sequence[Any] = field(default_factory=list)
    limits: tuple[float, float] | None = None
    grid: bool = True

    tick_labels: list[str] = field(default_factory=list)
    tick_fontweight: FontWeights = "normal"
    tick_colors: str | list[str] | None = None

    # --- Lifecycle & Initialization ---
    def __post_init__(self):
        """Perform post-initialization tasks."""
        if self.limits is None:
            if is_list_of(self.ticks, (int, float)):
                lower = max(0.0, min(self.ticks) - 0.5)
                self.limits = (lower, max(self.ticks) + 0.5)
            else:
                self.limits = (0.5, len(self.ticks) + 0.5)

        # If tick_labels is not provided, use the ticks as labels
        if self.tick_labels is None or len(self.tick_labels) == 0:
            self.tick_labels = [str(t) for t in self.ticks]

        # If tick_colors is a string, check if it's a valid palette name and
        # set text_colors accordingly
        if self.tick_colors is not None:
            self.tick_colors = validate_hex_color(self.tick_colors)

    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the number of ticks on the axis."""
        return len(self.ticks)

    # --- Computation ---
    def angles(self, full: bool = False, deg: bool = False) -> list[float]:
        """Return the angles for the axis ticks in radians.

        Args:
            full (bool): If True, span the full circle (2π) for the series.
                If False, add 1 more arc for the Y-axis tick labels.
                Defaults to False.
            deg (bool): If True, return the angles in degrees.
                If False, return the angles in radians. Defaults to False.

        Returns:
            list[float]: A list of angles in radians corresponding to the
                tick values.
        """
        start = 0
        end = 2 * np.pi if full else 2 * np.pi - YTICK_POLAR_WIDTH
        num = self.__len__()

        # We split the circle ($2\pi$) into equal parts based on the number of variables
        angles = np.linspace(start, end, num, endpoint=False)

        if not full:
            # We need to shift the angles by half the width of the Y-axis tick labels
            offset = 0.5 * (angles[1] + YTICK_POLAR_WIDTH)
            angles += offset

        # Convert to degrees if requested
        if deg:
            angles = np.degrees(angles)

        return angles.tolist()


@dataclass
class Series:
    """Data class for a data series.

    Attributes:
        label (str): The label for the data series.
        Y (Sequence): The Y-values of the data series.
        marker (Markers): The marker style for the data series. Defaults to "o".
        linestyle (LineStyles): The line style for the data series. Defaults to "-".
        width (float): The line width for the data series. Defaults to 1.0.
        scale_width (bool): Whether to scale the width depending on the
            Y-values. Defaults to False.
        scale_color (bool): Whether to scale the color depending on the
            Y-values. Defaults to False.
        color (str | None): The color for the data series. Defaults to None.
        alpha (float): The transparency level for the data series. Defaults to 1.0.
        S (list[str]): A list of strings associated with the data series.
            Defaults to an empty list.
        scale (int): The scale factor for the data series. Defaults to 1.
        ha (Alignments): The horizontal alignment for text associated with the
            data series. Defaults to "center".
        va (Alignments): The vertical alignment for text associated with the
            data series. Defaults to "center".
        ho (float): The horizontal offset for text associated with the data series.
            Defaults to 0.0.
        vo (float): The vertical offset for text associated with the data series.
            Defaults to 0.0.
        fontweight (FontWeights): The font weight for text associated with the
            data series. Defaults to "normal".
        text_direction (TextDirection): The direction for text associated with
            the axis. Defaults to "default".
        text_color (str | bool): The color of text associated with the data series.
            If True, use the color assigned to the series. If False, use the
            default matplotlib text color. Defaults to False.
        text_alpha (float): The transparency level for text associated with the
            data series. Defaults to 1.0.
    """

    label: str
    Y: Sequence

    marker: Markers = "o"
    linestyle: LineStyles = "-"
    width: float = 1.0
    color: str | list[str] | None = None
    alpha: float = 1.0
    scale_width: bool = False
    scale_color: bool = False

    S: list[str] = field(default_factory=list)
    scale: int = 1
    ha: Alignments = "center"
    va: Alignments = "center"
    ho: float = 0.0
    vo: float = 0.0
    fontweight: FontWeights = "normal"
    text_direction: TextDirection = "default"
    text_color: str | list[str] | bool | None = False
    text_alpha: float = 1.0

    # --- Lifecycle & Initialization ---
    def __post_init__(self):
        """Perform post-initialization tasks."""
        if self.S is None or len(self.S) == 0:
            S = []
            for yi in self.Y:
                if isinstance(yi, (int, float)):
                    S.append(f"{yi:.{self.scale}f}")
                elif yi is None:
                    S.append("")
                else:
                    S.append(str(yi))
            self.S = S

        if self.color is not None:
            self.color = validate_hex_color(self.color)

        if not isinstance(self.text_color, bool):
            self.text_color = validate_hex_color(self.text_color)

    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the length of the series."""
        return len(self.Y)

    # --- Properties ---
    @property
    def line_kw(self) -> dict:
        """Return the keyword arguments for the line."""
        return {
            "label": self.label,
            "marker": self.marker,
            "linestyle": self.linestyle,
            "linewidth": self.width,
            "alpha": self.alpha,
        }

    @property
    def bar_kw(self) -> dict:
        """Return the keyword arguments for the bar."""
        return {
            "label": self.label,
            "width": self.width,
        }

    @property
    def label_kw(self) -> dict:
        """Return the keyword arguments for the label."""
        kw = {
            "ha": self.ha,
            "va": self.va,
            "fontweight": self.fontweight,
            "alpha": self.text_alpha,
        }
        if self.text_direction in ["xtick", "ytick"]:
            kw["rotation_mode"] = "anchor"
        if self.text_direction == "xtick":
            kw["transform_rotates_text"] = True
        return kw

    # --- Computation ---
    def polar_width(self, full: bool = False) -> float:
        """Return the width for polar plots.

        Args:
            full (bool, optional): If True, span the full circle (2π) for the series.
                If False, add 1 more arc for the Y-axis tick labels. Defaults to False.

        Returns:
            float: The width for polar plots.
        """
        start = 0
        end = 2 * np.pi if full else 2 * np.pi - YTICK_POLAR_WIDTH
        num = self.__len__()
        return self.width * (end - start) / num

    def wmap(self, polar: bool = False, scale: bool = True) -> list[float]:
        """Create a width map for the series depending on the Y-values.

        Args:
            polar (bool, optional): Whether to compute the width map for polar plots.
                Defaults to False.
            scale (bool, optional): Whether to scale the width depending on the Y-values.
                Defaults to True.

        Returns:
            list[float]: A list of widths corresponding to the Y-values of the
                series.
        """
        if polar:
            width = self.polar_width()
        else:
            width = self.width

        scale = scale and self.scale_width
        if scale:
            y_max = max(self.Y)
            y_max = y_max if y_max != 0 else 1
            return [yi / y_max * width for yi in self.Y]
        else:
            return [width] * self.__len__()

    def cmap(self, colors: Any = None, gradient: bool = True) -> list[str]:
        """Create a gradient color map for the series.

        Prioritize the series' own color if available, otherwise use the
        provided colors.

        Args:
            colors (Any, optional): The base colors to use for the color map.
                If None, use the series' own color. Defaults to None.
            gradient (bool, optional): Whether to create a gradient color map
                or a categorical color map. Defaults to True.

        Returns:
            list[str]: A list of hex colors corresponding to the Y-values of
                the series.

        Raises:
            ValueError: If the provided colors are not a list of strings.
        """
        # Get the base colors
        colors = self.color or colors

        # Get the cmap
        cmap = ColorPalette.from_any(colors)
        if gradient:
            return cmap.gradient_cmap(self.Y)
        else:
            return cmap.categorical_cmap(self.Y)


@dataclass
class Spine:
    """Data class for a spine.

    Spines are the lines connecting the axis tick marks and noting the boundaries
    of the data area.

    Attributes:
        spine_type (SpineTypes): The type of the spine.
        visible (bool): Whether the spine is visible.
    """

    spine_type: SpineTypes
    visible: bool


@dataclass
class Legend:
    """Data class for a legend.

    The legend is a visual guide box explaining data line styles, markers, or
    colors.

    References:
        - https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.FigureBase.legend

    Attributes:
        title (str | None): The title of the legend. Defaults to None.
        loc (Locations): The location of the legend. Defaults to "upper right".
        bbox_to_anchor (tuple[float, float] | None): The bounding box used to
            position the legend in conjunction with ``loc``. Defaults to None.
        ncol (int): The number of columns in the legend. Defaults to 1.
    """

    title: str | None = None
    loc: Locations = "upper right"
    bbox_to_anchor: tuple[float, float] | None = None
    ncol: int = 1

    # --- Properties ---
    @property
    def legend_kw(self) -> dict:
        """Return the bounding box keyword arguments for the legend."""
        return {
            "title": self.title,
            "loc": self.loc,
            "bbox_to_anchor": self.bbox_to_anchor,
            "ncol": self.ncol,
        }


@dataclass
class Axes:
    """Data class for axes.

    This is a container for what is often colloquially called a plot/chart/graph.
    It includes the key components (sorted in plotting order) of a chart::

        - Title: Text centered at the top summarizing the dataset.
        - Spines: Border lines outlining the data plotting box.
        - Grid: Structural background guide lines mapping data intersections.
        - Major Ticks: Primary numerical tick marks identifying major data intervals.
        - Minor Ticks: Small, sub-interval markings located between major ticks.
        - Tick Labels: The text labels mapping precise coordinates alongside ticks.
        - Axis Labels: Explanatory name text placed alongside the X or Y
        - Data: The plotted data lines, bars, or points representing the dataset.
        - Legend: Visual guide box explaining data line styles, markers, or colors.

    Attributes:
        title (str | None): The title of the axes.
        x (Axis): The X-axis.
        y (Axis): The Y-axis.
        legend (Legend | None): The legend for the axes. Defaults to None.
        colors (str | list[str] | None): The color scheme for the axes.
            Defaults to None.
    """

    title: str | None = None
    spines: list[Spine] | None = None
    x: Axis = field(default_factory=Axis)
    y: Axis = field(default_factory=Axis)
    data: list[Series] = field(default_factory=list)
    legend: Legend | None = None
    colors: str | list[str] | None = None

    # --- Lifecycle & Initialization ---
    def __post_init__(self):
        """Perform post-initialization tasks."""
        if self.colors is not None:
            self.colors = validate_hex_color(self.colors)

    # --- Properties ---
    @property
    def grid_kw(self) -> dict:
        """Return whether grid lines are enabled for the subplot."""
        return {
            "visible": self.x.grid or self.y.grid,
            "which": "both",
            "axis": "both" if self.x.grid and self.y.grid else "x" if self.x.grid else "y",
        }

# endregion


# ==============================================================================
# region VISUALIZATION
# ==============================================================================

def set_spines(ax: plt.Axes, spines: list[Spine], *args, **kwargs):
    """Set the spines on the specified axis.

    Args:
        ax (plt.Axes): The Matplotlib Axes object.
        spines (list[Spine]): A list of Spine objects specifying the spine
            properties to set.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    for spine in spines:
        ax.spines[spine.spine_type].set_visible(spine.visible)


def set_ticklabels(ticks: list[plt.Text], axis: Axis, *args, **kwargs):
    """Set the tick labels on the specified axis.

    Args:
        ticks (list[plt.Text]): The list of tick label Text objects.
        axis (Axis): The Axis object containing tick label settings.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    colors = axis.tick_colors or []

    for i, tick in enumerate(ticks):
        tick.set_fontweight(axis.tick_fontweight)
        if i < len(colors):
            tick.set_color(colors[i])


def plot_marker_labels(
    ax: plt.Axes,
    X: Sequence,
    series: Series,
    color: Any = plt.rcParams["text.color"],
    *args, **kwargs
) -> list[plt.Text]:
    """Plot marker labels on the specified axis.

    Args:
        ax (plt.Axes): The Matplotlib Axes object.
        X (Sequence): The X-values for the plot.
        series (Series): The Series object containing Y-values and labels.
        color (Any, optional): The color for the text labels. If None, use the
            default text color. Defaults to None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments for text properties.

    Returns:
        list[plt.Text]: A list of Text objects for the plotted labels.
    """
    # 1. Define data
    Y = series.Y
    S = series.S

    colors = ColorPalette.from_any(color).cycler
    ymin, ymax = ax.get_ylim()

    # 2. Plot marker labels
    label_kw = series.label_kw | kwargs
    labels = []
    for i, (x, y, s) in enumerate(zip(X, Y, S)):
        # Check if the value is outside the auto-determined vertical limits
        if y is None or y < ymin or y > ymax:
            continue

        color = next(colors)

        label = ax.text(x=x + series.ho, y=y + series.vo, s=s, color=color, **label_kw)
        labels.append(label)

    return labels


def plot_polar_marker_labels(
    ax: plt.Axes,
    X: Sequence,
    series: Series,
    color: Any = plt.rcParams["text.color"],
    *args, **kwargs
) -> list[plt.Text]:
    """Plot marker labels on a polar axis.

    Args:
        ax (plt.Axes): The Matplotlib Axes object.
        X (Sequence): The X-values (angles in radians) for the polar plot.
        series (Series): The Series object containing Y-values and labels.
        color (Any, optional): The color for the text labels. If None, use the
            default text color. Defaults to None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments for text properties.

    Returns:
        list[plt.Text]: A list of Text objects for the plotted labels.
    """
    # 1. Define data
    X_deg = np.degrees(X).tolist()
    Y = series.Y
    S = series.S

    colors = ColorPalette.from_any(color).cycler
    ymin, ymax = ax.get_ylim()

    # 2. Plot marker labels
    label_kw = series.label_kw | kwargs
    labels = []
    for i, (x, y, s, d) in enumerate(zip(X, Y, S, X_deg)):
        # Check if the value is outside the auto-determined vertical limits
        if y is None or y < ymin or y > ymax:
            continue

        color = next(colors)
        rotation = 0
        direction = series.text_direction

        if direction == "xtick":
            # Determine vertical alignment and relative rotation for x-axis ticks
            is_top = d <= 90 or d >= 270
            label_kw["ha"] = "center"
            label_kw["va"] = "bottom" if is_top else "top"
            rotation = 180 if is_top else 0
        if direction == "ytick":
            # Determine horizontal alignment and relative rotation for y-axis ticks
            is_left = 0 <= d < 180
            label_kw["ha"] = "right" if is_left else "left"
            label_kw["va"] = "center"
            rotation = d - 90 if is_left else d + 90

        label = ax.text(
            x=x + series.ho,
            y=y + series.vo,
            s=s,
            color=color,
            **label_kw
        )
        label.set_rotation(rotation)
        labels.append(label)

    return labels

# endregion


# ==============================================================================
# region UTILITIES
# ==============================================================================

def set_plot_style(
    font_family: str = K.FONT_FAMILY,
    font_size: int = K.FONT_SIZE,
    palette: str = K.PALETTE,
):
    """Set the global plot style for Matplotlib.

    Args:
        font_family (str, optional): The font family for the plot.
            Defaults to K.FONT_FAMILY.
        font_size (int, optional): The font size for the plot.
            Defaults to K.FONT_SIZE.
        palette (str, optional): The color palette for the plot.
            Defaults to K.PALETTE.
    """
    dpi = 300
    linewidth = 0.5
    colors = cycler(color=COLOR_PALETTES.get(palette, COLOR_PALETTES[K.PALETTE]).hex)
    labelcolor = "black"
    alpha = 1.0

    plt.rcParams.update({
        # Global
        "font.family": font_family,
        "font.size": font_size,
        # Figure
        "figure.dpi": dpi,
        "figure.titlesize": font_size + 2,
        # Axes
        "axes.axisbelow": True,         # Force grid lines behind data elements
        "axes.linewidth": linewidth,
        "axes.prop_cycle": colors,      # Color Palette
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size + 1,
        # Legend
        "legend.loc": "upper right",
        "patch.linewidth": linewidth,
        "legend.fontsize": font_size - 1,
        "legend.title_fontsize": font_size - 1,
        "legend.framealpha": alpha,
        # Grid
        "grid.linestyle": "--",
        "grid.linewidth": linewidth,
        "grid.alpha": alpha / 2,
        # Ticks
        "xtick.major.width": linewidth,
        "xtick.labelsize": font_size,
        "xtick.labelcolor": labelcolor,
        "ytick.major.width": linewidth,
        "ytick.labelsize": font_size,
        "ytick.labelcolor": labelcolor,
        # Line
        "lines.markersize": linewidth * 3,
        "lines.linewidth": linewidth * 2,
        # Saving
        "savefig.dpi": dpi,
        "pdf.compression": 9,
    })


def get_figsize(
    width: float | str = "single",
    height_ratio: float = 0.5,
) -> tuple[float, float]:
    """Computes exact inches for column bounds so text doesn't warp in LaTeX.

    Args:
        width (float | str, optional): Width for the figure.
            Options are "single" for single-column width (3.4 inches),
            "double" for double-column width (6.8 inches), or a custom float
            value for width in inches. Defaults to "single".
        height_ratio (float, optional): Aspect ratio of the figure (height/width).
            Defaults to 0.5.

    Returns:
        tuple[float, float]: A tuple containing the width and height of the
            figure in inches.
    """
    if isinstance(width, str):
        width = PAGE_WIDTHS.get(width, PAGE_WIDTHS["single"])

    height = width * height_ratio
    return width, height

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
