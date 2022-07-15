#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Color palettes.
"""

from __future__ import annotations

import inspect
import sys

from ordered_enum import OrderedEnum


# MARK: - Enum

class AppleRGB(OrderedEnum):
	"""Define 12 Apple colors."""
	GRAY   = (128, 128, 128)
	RED    = (255, 59 , 48 )
	GREEN  = ( 52, 199, 89 )
	BLUE   = (  0, 122, 255)
	ORANGE = (255, 149, 5  )
	YELLOW = (255, 204, 0  )
	BROWN  = (162, 132, 94 )
	PINK   = (255, 45 , 85 )
	PURPLE = ( 88, 86 , 214)
	TEAL   = ( 90, 200, 250)
	INDIGO = ( 85, 190, 240)
	BLACK  = (  0, 0  , 0  )
	WHITE  = (255, 255, 255)
	
	@staticmethod
	def values():
		"""Return the list of all values.

		Returns:
			(list):
				List of all color tuple.
		"""
		return [e.value for e in AppleRGB]


class BasicRGB(OrderedEnum):
	"""Define 12 basic colors."""
	BLACK   = (0  , 0  , 0  )
	WHITE   = (255, 255, 255)
	RED     = (255, 0  , 0  )
	LIME    = (0  , 255, 0  )
	BLUE    = (0  , 0  , 255)
	YELLOW  = (255, 255, 0  )
	CYAN    = (0  , 255, 255)
	MAGENTA = (255, 0  , 255)
	SILVER  = (192, 192, 192)
	GRAY    = (128, 128, 128)
	MAROON  = (128, 0  , 0  )
	OLIVE   = (128, 128, 0  )
	GREEN   = (0  , 128, 0  )
	PURPLE  = (128, 0  , 128)
	TEAL    = (0  , 128, 128)
	NAVY    = (0  , 0  , 128)
	
	@staticmethod
	def values():
		"""Return the list of all values.
		
		Returns:
			(list):
				List of all color tuple.
		"""
		return [e.value for e in BasicRGB]


class RGB(OrderedEnum):
	"""Define list of 138 colors."""
	MAROON                  = (128, 0  , 0  )
	DARK_RED                = (139, 0  , 0  )
	BROWN                   = (165, 42 , 42 )
	FIREBRICK               = (178, 34 , 34 )
	CRIMSON                 = (220, 20 , 60 )
	RED                     = (255, 0  , 0  )
	TOMATO                  = (255, 99 , 71 )
	CORAL                   = (255, 127, 80 )
	INDIAN_RED              = (205, 92 , 92 )
	LIGHT_CORAL             = (240, 128, 128)
	DARK_SALMON             = (233, 150, 122)
	SALMON                  = (250, 128, 114)
	LIGHT_SALMON            = (255, 160, 122)
	ORANGE_RED              = (255, 69 , 0  )
	DARK_ORANGE             = (255, 140, 0  )
	ORANGE                  = (255, 165, 0  )
	GOLD                    = (255, 215, 0  )
	DARK_GOLDEN_ROD         = (184, 134, 11 )
	GOLDEN_ROD              = (218, 165, 32 )
	PALE_GOLDEN_ROD         = (238, 232, 170)
	DARK_KHAKI              = (189, 183, 107)
	KHAKI                   = (240, 230, 140)
	OLIVE                   = (128, 128, 0  )
	YELLOW                  = (255, 255, 0  )
	YELLOW_GREEN            = (154, 205, 50 )
	DARK_OLIVE_GREEN        = (85 , 107, 47 )
	OLIVE_DRAB              = (107, 142, 35 )
	LAWN_GREEN              = (124, 252, 0  )
	CHART_REUSE             = (127, 255, 0  )
	GREEN_YELLOW            = (173, 255, 47 )
	DARK_GREEN              = (0  , 100, 0  )
	GREEN                   = (0  , 128, 0  )
	FOREST_GREEN            = (34 , 139, 34 )
	LIME                    = (0  , 255, 0  )
	LIME_GREEN              = (50 , 205, 50 )
	LIGHT_GREEN             = (144, 238, 144)
	PALE_GREEN              = (152, 251, 152)
	DARK_SEA_GREEN          = (143, 188, 143)
	MEDIUM_SPRING_GREEN     = (0  , 250, 154)
	SPRING_GREEN            = (0  , 255, 127)
	SEA_GREEN               = (46 , 139, 87 )
	MEDIUM_AQUA_MARINE      = (102, 205, 170)
	MEDIUM_SEA_GREEN        = (60 , 179, 113)
	LIGHT_SEA_GREEN         = (32 , 178, 170)
	DARK_SLATE_GRAY         = (47 , 79 , 79 )
	TEAL                    = (0  , 128, 128)
	DARK_CYAN               = (0  , 139, 139)
	AQUA                    = (0  , 255, 255)
	CYAN                    = (0  , 255, 255)
	LIGHT_CYAN              = (224, 255, 255)
	DARK_TURQUOISE          = (0  , 206, 209)
	TURQUOISE               = (64 , 224, 208)
	MEDIUM_TURQUOISE        = (72 , 209, 204)
	PALE_TURQUOISE          = (175, 238, 238)
	AQUA_MARINE             = (127, 255, 212)
	POWDER_BLUE             = (176, 224, 230)
	CADET_BLUE              = (95 , 158, 160)
	STEEL_BLUE              = (70 , 130, 180)
	CORN_FLOWER_BLUE        = (100, 149, 237)
	DEEP_SKY_BLUE           = (0  , 191, 255)
	DODGER_BLUE             = (30 , 144, 255)
	LIGHT_BLUE              = (173, 216, 230)
	SKY_BLUE                = (135, 206, 235)
	LIGHT_SKY_BLUE          = (135, 206, 250)
	MIDNIGHT_BLUE           = (25 , 25 , 112)
	NAVY                    = (0  , 0  , 128)
	DARK_BLUE               = (0  , 0  , 139)
	MEDIUM_BLUE             = (0  , 0  , 205)
	BLUE                    = (0  , 0  , 255)
	ROYAL_BLUE              = (65 , 105, 225)
	BLUE_VIOLET             = (138, 43 , 226)
	INDIGO                  = (75 , 0  , 130)
	DARK_SLATE_BLUE         = (72 , 61 , 139)
	SLATE_BLUE              = (106, 90 , 205)
	MEDIUM_SLATE_BLUE       = (123, 104, 238)
	MEDIUM_PURPLE           = (147, 112, 219)
	DARK_MAGENTA            = (139, 0  , 139)
	DARK_VIOLET             = (148, 0  , 211)
	DARK_ORCHID             = (153, 50 , 204)
	MEDIUM_ORCHID           = (186, 85 , 211)
	PURPLE                  = (128, 0  , 128)
	THISTLE                 = (216, 191, 216)
	PLUM                    = (221, 160, 221)
	VIOLET                  = (238, 130, 238)
	MAGENTA                 = (255, 0  , 255)
	ORCHID                  = (218, 112, 214)
	MEDIUM_VIOLET_RED       = (199, 21 , 133)
	PALE_VIOLET_RED         = (219, 112, 147)
	DEEP_PINK               = (255, 20 , 147)
	HOT_PINK                = (255, 105, 180)
	LIGHT_PINK              = (255, 182, 193)
	PINK                    = (255, 192, 203)
	ANTIQUE_WHITE           = (250, 235, 215)
	BEIGE                   = (245, 245, 220)
	BISQUE                  = (255, 228, 196)
	BLANCHED_ALMOND         = (255, 235, 205)
	WHEAT                   = (245, 222, 179)
	CORN_SILK               = (255, 248, 220)
	LEMON_CHIFFON           = (255, 250, 205)
	LIGHT_GOLDEN_ROD_YELLOW = (250, 250, 210)
	LIGHT_YELLOW            = (255, 255, 224)
	SADDLE_BROWN            = (139, 69 , 19 )
	SIENNA                  = (160, 82 , 45 )
	CHOCOLATE               = (210, 105, 30 )
	PERU                    = (205, 133, 63 )
	SANDY_BROWN             = (244, 164, 96 )
	BURLY_WOOD              = (222, 184, 135)
	TAN                     = (210, 180, 140)
	ROSY_BROWN              = (188, 143, 143)
	MOCCASIN                = (255, 228, 181)
	NAVAJO_WHITE            = (255, 222, 173)
	PEACH_PUFF              = (255, 218, 185)
	MISTY_ROSE              = (255, 228, 225)
	LAVENDER_BLUSH          = (255, 240, 245)
	LINEN                   = (250, 240, 230)
	OLD_LACE                = (253, 245, 230)
	PAPAYA_WHIP             = (255, 239, 213)
	SEA_SHELL               = (255, 245, 238)
	MINT_CREAM              = (245, 255, 250)
	SLATE_GRAY              = (112, 128, 144)
	LIGHT_SLATE_GRAY        = (119, 136, 153)
	LIGHT_STEEL_BLUE        = (176, 196, 222)
	LAVENDER                = (230, 230, 250)
	FLORAL_WHITE            = (255, 250, 240)
	ALICE_BLUE              = (240, 248, 255)
	GHOST_WHITE             = (248, 248, 255)
	HONEYDEW                = (240, 255, 240)
	IVORY                   = (255, 255, 240)
	AZURE                   = (240, 255, 255)
	SNOW                    = (255, 250, 250)
	BLACK                   = (0  , 0  , 0  )
	DIM_GRAY                = (105, 105, 105)
	GRAY                    = (128, 128, 128)
	DARK_GRAY               = (169, 169, 169)
	SILVER                  = (192, 192, 192)
	LIGHT_GRAY              = (211, 211, 211)
	GAINSBORO               = (220, 220, 220)
	WHITE_SMOKE             = (245, 245, 245)
	WHITE                   = (255, 255, 255)
	
	@staticmethod
	def values():
		"""Return the list of all values.

		Returns:
			(list):
				List of all color tuple.
		"""
		return [e.value for e in RGB]


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
