#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Color Palettes.

This module provides color palettes used in plotting and visualization.
"""

from __future__ import annotations

__all__ = [
    "COLOR_PALETTES",
    "ColorPalette",
    "is_hex_color",
    "to_hex",
    "to_rgb",
    "to_rgb_frac",
    "validate_hex_color",
    "wcag_2_contrast",
]

import itertools
import re
from dataclasses import dataclass
from typing import Any, Iterator, Sequence
from cycler import cycler
import matplotlib as mpl
import numpy as np
from kontrasto import wcag_2

# ==============================================================================
# region CONSTANTS
# ==============================================================================

# Pre-compiled regex patterns for validation and splitting.
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{3}$|^#[0-9a-fA-F]{6}$")

# endregion


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

@dataclass
class ColorPalette:
    """Dataclass for creating a new color palette.

    Args:
        name (str): The name of the color palette.
        colors (list[str]): A list of hex color codes for the palette.
    """

    name: str
    colors: list[str]

    # --- Lifecycle & Initialization ---
    def __post_init__(self):
        if any(not is_hex_color(color) for color in self.colors):
            raise ValueError("expected all colors must be valid hex color codes.")

    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the length of the container."""
        return len(self.colors)

    def __getitem__(self, index: int) -> Any:
        """Return an item at the given ``index``."""
        return self.colors[index]

    def __setitem__(self, index: int, value: Any):
        """Define behavior for when an item is assigned to, using the notation
        self[key] = value.
        """
        self.colors[index] = value

    def __iter__(self) -> Iterator:
        """Return an iterator for the container."""
        return iter(self.colors)

    def __contains__(self, item: Any) -> bool:
        """Define behavior for membership tests using in and not in."""
        return item in self.colors

    # --- Properties ---
    @property
    def hex(self) -> list[str]:
        """Return list of hex color codes in order."""
        return self.colors

    @property
    def rgb(self) -> list[tuple[int, int, int]]:
        """Return list of RGB color tuples in order."""
        return [to_rgb(color) for color in self.colors]

    @property
    def cmap(self) -> mpl.colors.LinearSegmentedColormap:
        """Return a Matplotlib colormap object for this color palette."""
        colors = self.colors
        if len(colors) == 1:
            colors.append("#FFFFFF")
        return mpl.colors.LinearSegmentedColormap.from_list(self.name, colors, N=256)

    @property
    def cycler(self) -> itertools.cycle:
        """Return an infinite cycle iterator for the color palette."""
        return itertools.cycle(self.colors)

    @property
    def prop_cycler(self):
        """Return a Matplotlib cycler object for the color palette."""
        return cycler(color=self.colors)

    # --- Creation ---
    @classmethod
    def from_any(cls, colors: Any, name: str = "custom", *args, **kwargs) -> ColorPalette:
        """Create a ColorPalette object from arbitrary input."""
        colors = validate_hex_color(colors)
        colors = colors or ["#000000"]
        colors = [colors] if not isinstance(colors, list) else colors
        return cls(name=name, colors=colors)

    # --- Computation ---
    def categorical_cmap(self, values: int | Sequence) -> list[str]:
        """Create a categorical colormap from a list of base colors."""
        N = len(values) if isinstance(values, Sequence) else int(values)
        cmap = self.cmap
        color_positions = np.linspace(0, 1, N)
        rgba_colors = cmap(color_positions)
        return [mpl.colors.to_hex(c) for c in rgba_colors]

    def gradient_cmap(self, values: Sequence) -> list[str]:
        """Create a gradient of colors from a list of base colors."""
        cmap = self.cmap
        norm = mpl.colors.Normalize(vmin=min(values), vmax=max(values))
        colors = cmap(norm(values))
        return [mpl.colors.to_hex(c) for c in colors]

# endregion


# ==============================================================================
# region VALIDATION
# ==============================================================================

def is_hex_color(value: Any) -> bool:
    """Check if the input string is a valid hex color code.

    Regex breakdown:
        ^#             - Starts with a '#' symbol
        [0-9a-fA-F]    - Followed by valid hex characters (case-insensitive)
        {3}            - Exactly 3 characters long (e.g., #FFF)
        |              - OR
        [0-9a-fA-F]    - Followed by valid hex characters
        {6}            - Exactly 6 characters long (e.g., #FFFFFF)
        $              - Ends the string tightly with no trailing characters
    """
    return isinstance(value, str) and bool(_HEX_COLOR_RE.match(value))


def validate_hex_color(value: Any) -> str | list[str] | None:
    """Validate if the input is a valid hex color code or a list of hex color codes.

    Args:
        value (Any): The input value to validate.

    Returns:
        str | list[str] | None: Returns the validated hex color code(s) if valid,
            otherwise None.
    """
    if isinstance(value, str):
        if value in COLOR_PALETTES:
            return COLOR_PALETTES[value].hex
        elif is_hex_color(value):
            return value
    elif isinstance(value, list) and all(is_hex_color(v) for v in value):
        return value
    return None

# endregion


# ==============================================================================
# region COMPUTATION
# ==============================================================================

# --- Arithmetic ---

def wcag_2_contrast(
    background: str,
    light: str = "#FFFFFF",
    dark: str = "#000000"
) -> dict[str, str]:
    """Calculate the WCAG 2.0 contrast ratio between a background and a
    foreground color.

    Args:
        background (str): The background color in hex format.
        light (str, optional): The light color to compare against.
            Defaults to white (#FFFFFF).
        dark (str, optional): The dark color to compare against.
            Defaults to black (#000000).

    Returns:
        dict[str, str]: A dictionary containing the recommended text color,
            its theme (light or dark), the background color, and its theme.

    Raises:
        ValueError: If any of the provided colors are not valid hex color codes.
    """
    if any(not is_hex_color(color) for color in [background, light, dark]):
        raise ValueError("expected all colors must be valid hex color codes.")

    light_contrast = wcag_2.wcag2_contrast(background, light)
    dark_contrast = wcag_2.wcag2_contrast(background, dark)
    lighter = light_contrast > dark_contrast
    return {
        "text_color": light if lighter else dark,
        "text_theme": "light" if lighter else "dark",
        "bg_color": background,
        "bg_theme": "dark" if lighter else "light",
    }


# --- Comparison ---


# --- Logical ---


# --- Geometric ---

# endregion


# ==============================================================================
# region TRANSFORMATION
# ==============================================================================

# --- Casting ---

def to_rgb_frac(color: str) -> tuple[float, float, float]:
    """Convert a hex color code to an RGB tuple with floating point values.

    Args:
        color (str): The hex color code (e.g., "#RRGGBB").

    Returns:
        tuple[float, float, float]: A tuple containing the RGB values as floats
            in the range [0.0, 1.0].

    Raises:
        ValueError: If the provided hex code is not a valid hex color code.
    """
    if not is_hex_color(color):
        raise ValueError(f"expected a valid hex color code, got {color}.")

    h = color.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return r, g, b


def to_rgb(color: str) -> tuple[int, int, int]:
    """Convert a hex color code to an RGB tuple.

    Args:
        color (str): The hex color code (e.g., "#RRGGBB").

    Returns:
        tuple[int, int, int]: A tuple containing the RGB values as integers
            in the range [0, 255].

    Raises:
        ValueError: If the provided hex code is not a valid hex color code.
    """
    if not is_hex_color(color):
        raise ValueError(f"expected a valid hex color code, got {color}.")

    h = color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return r, g, b


def to_hex(color: Sequence[int]) -> str:
    """Convert an RGB tuple to a hex color code.

    Args:
        color (Sequence[int]): A sequence containing the RGB values as integers
            in the range [0, 255].

    Returns:
        str: The hex color code (e.g., "#RRGGBB").
    """
    return "#%02x%02x%02x" % tuple(color)


# --- Encoding ---


# --- Standardization ---


# --- Structural ---


# --- Statistical ---


# --- Geometric ---

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

COLOR_PALETTES = {
    "okabe_ito" : ColorPalette("okabe_ito",  colors=["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D56000", "#CC79A7", "#000000"]),
    "tol_bright": ColorPalette("tol_bright", colors=["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377", "#BBBBBB"]),
    "tol_light" : ColorPalette("tol_light",  colors=["#77AADD", "#EE8866", "#EEDD88", "#FFAABB", "#99DDFF", "#44BB99", "#BBCC33", "#AAAA00", "#DDDDDD"]),
    "tol_muted" : ColorPalette("tol_muted",  colors=["#CC6677", "#332288", "#DDCC77", "#117733", "#88CCEE", "#882255", "#44AA99", "#999933", "#AA4499"]),
    "viridis"   : ColorPalette("viridis",    colors=["#440154", "#31688E", "#35B779", "#90D743", "#FDE725"]),
    "plasma"    : ColorPalette("plasma",     colors=["#0D0887", "#7E03A8", "#CC4778", "#F89540", "#F0F921"]),
    "thoughts"  : ColorPalette("thoughts",   colors=["#355C7D", "#6C5B7B", "#C06C84", "#F67280", "#F8B195"]),
    "thoughts4" : ColorPalette("thoughts4",  colors=["#6C5B7B", "#C06C84", "#F67280", "#F8B195"]),
}

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
