#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Constants.

This module defines various global constants used throughout the ``mon`` package.
"""

from __future__ import annotations

__all__ = [
    "K",
    "STI",
]

from types import SimpleNamespace

from .dtype import ConfigExtension, ImageExtension, WeightExtension
from .path import Path

# ==============================================================================
# region PATHS
# ==============================================================================

# Find the ``mon`` package root (first pyproject.toml up)
current_file = Path(__file__).normalize()
root = current_file
for parent in current_file.parents:
    if (parent / "pyproject.toml").exists():
        root = parent
        break

# Find the monorepo root (highest pyproject.toml up)
all_roots = [p for p in root.parents if (p / "pyproject.toml").exists()]
mono_root = all_roots[-1] if all_roots else root

# Find the assets directory (prefer root-level assets if it exists)
assets_root = root / "assets"
if not assets_root.exists():
    assets_root = mono_root / "assets"

# Find the model zoo directory (prefer root-level zoo if it exists)
zoo_root = root / "zoo"
if not zoo_root.exists():
    zoo_root = mono_root / "zoo"

# endregion


# ==============================================================================
# region VALUES
# ==============================================================================

class STI(SimpleNamespace):
    """Class for standard test image paths."""

    # Classic
    AIRPLANE    = assets_root / "sti" / "classic" / "airplane.jpg"
    BABOON      = assets_root / "sti" / "classic" / "baboon.jpg"
    BARBARA     = assets_root / "sti" / "classic" / "barbara.jpg"
    BOATS       = assets_root / "sti" / "classic" / "boats.jpg"
    BOATS_COLOR = assets_root / "sti" / "classic" / "boats_color.jpg"
    GOLDHILL    = assets_root / "sti" / "classic" / "goldhill.jpg"
    LENNA       = assets_root / "sti" / "classic" / "lenna.jpg"
    PEPPER      = assets_root / "sti" / "classic" / "pepper.jpg"

    # Finger-print
    FINGER        = assets_root / "sti" / "fingerprint" / "finger.jpg"
    FINGER_PRINT1 = assets_root / "sti" / "fingerprint" / "finger_print1.jpg"
    FINGER_PRINT2 = assets_root / "sti" / "fingerprint" / "finger_print2.jpg"

    # High-resolution
    MALAMUTE = assets_root / "sti" / "high_resolution" / "malamute.jpg"
    MALTESE  = assets_root / "sti" / "high_resolution" / "maltese.jpg"
    RAINIER  = assets_root / "sti" / "high_resolution" / "rainier.jpg"
    SUNRISE  = assets_root / "sti" / "high_resolution" / "sunrise.jpg"

    # Medical
    BONE_SCINT    = assets_root / "sti" / "medical" / "bone_scint.jpg"
    BRAIN1        = assets_root / "sti" / "medical" / "brain1.jpg"
    BRAIN2        = assets_root / "sti" / "medical" / "brain2.jpg"
    BRAIN3        = assets_root / "sti" / "medical" / "brain3.jpg"
    BRAIN4        = assets_root / "sti" / "medical" / "brain4.jpg"
    BRAIN5        = assets_root / "sti" / "medical" / "brain5.jpg"
    LUNGS         = assets_root / "sti" / "medical" / "lungs.jpg"
    MR            = assets_root / "sti" / "medical" / "mr.jpg"
    SHOULDER_CR   = assets_root / "sti" / "medical" / "shoulder_cr.jpg"
    THYROID_SCINT = assets_root / "sti" / "medical" / "thyroid_scint.jpg"
    ULTRASOUND    = assets_root / "sti" / "medical" / "ultrasound.jpg"

    # Old Classic
    BRIDGE     = assets_root / "sti" / "old_classic" / "bridge.jpg"
    CAMERA_MAN = assets_root / "sti" / "old_classic" / "camera_man.jpg"
    CLOWN      = assets_root / "sti" / "old_classic" / "clown.jpg"
    COUPLE     = assets_root / "sti" / "old_classic" / "couple.jpg"
    CROWD      = assets_root / "sti" / "old_classic" / "crowd.jpg"
    GIRL_FACE  = assets_root / "sti" / "old_classic" / "girl_face.jpg"
    MAN        = assets_root / "sti" / "old_classic" / "man.jpg"
    SAILBOAT   = assets_root / "sti" / "old_classic" / "sailboat.jpg"
    TANK       = assets_root / "sti" / "old_classic" / "tank.jpg"
    TRUCK      = assets_root / "sti" / "old_classic" / "truck.jpg"
    TRUCKS     = assets_root / "sti" / "old_classic" / "trucks.jpg"

    # Special
    COLOR_CHECKER = assets_root / "sti" / "special" / "color_checker.jpg"
    LENSTARG      = assets_root / "sti" / "special" / "lenstarg.jpg"
    MACH_COLOR    = assets_root / "sti" / "special" / "mach_color.jpg"
    ZONEPLATE     = assets_root / "sti" / "special" / "zoneplate.jpg"

    # Sun and planets
    EARTH   = assets_root / "sti" / "sun_and_planets" / "earth.jpg"
    JUPITER = assets_root / "sti" / "sun_and_planets" / "jupiter.jpg"
    MARS    = assets_root / "sti" / "sun_and_planets" / "mars.jpg"
    MERCURY = assets_root / "sti" / "sun_and_planets" / "mercury.jpg"
    NEPTUNE = assets_root / "sti" / "sun_and_planets" / "neptune.jpg"
    SATURN  = assets_root / "sti" / "sun_and_planets" / "saturn.jpg"
    SUN     = assets_root / "sti" / "sun_and_planets" / "sun.jpg"
    URANUS  = assets_root / "sti" / "sun_and_planets" / "uranus.jpg"
    VENUS   = assets_root / "sti" / "sun_and_planets" / "venus.jpg"

    # Texture
    BLOBS       = assets_root / "sti" / "texture" / "blobs.jpg"
    BRICK_WALL  = assets_root / "sti" / "texture" / "brick_wall.jpg"
    CARPET      = assets_root / "sti" / "texture" / "carpet.jpg"
    CELL_COLONY = assets_root / "sti" / "texture" / "cell_colony.jpg"
    TEXTURE_A   = assets_root / "sti" / "texture" / "textureA.jpg"
    TEXTURE_B   = assets_root / "sti" / "texture" / "textureB.jpg"


class K(SimpleNamespace):
    """Class for constants."""

    # --- Paths ---
    ROOT        = root
    MONO_ROOT   = mono_root
    ASSETS_ROOT = assets_root
    ZOO_ROOT    = zoo_root

    # --- Directories ---
    ANN_DIR     = "ann"
    DEBUG_DIR   = "debug"
    DEPTH_DIR   = "depth"
    IMAGE_DIR   = "image"
    VIS_DIR     = "vis"

    # --- Extensions ---
    ANN_EXT     = ConfigExtension.TXT
    CKPT_EXT    = WeightExtension.CKPT
    IMAGE_EXT   = ImageExtension.JPG
    WEIGHTS_EXT = WeightExtension.PT

    # --- Values ---
    DUMMY_IMAGE = STI.LENNA
    EPS = 1e-8
    INF = 1e8
    NAN = float("nan")
    PI  = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679

    # --- Strings ---
    INCLUDE_KEY = "__include__"
    ORIGINAL    = "orig"

    # --- Logging ---
    VERBOSE = True

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
