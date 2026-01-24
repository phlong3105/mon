#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Box mask extraction tool.

This module provides a tool for extracting masks from images using bounding box
annotations and a segmentation model.
"""

from __future__ import annotations

__all__ = []

import mon
from mon.tools import BoxMaskExtractor

current_file = mon.Path(__file__).normalize(exist=True)
current_dir  = current_file.parents[0]
data_dir     = current_dir.data_dir()


# ==============================================================================
# region MAIN
# ==============================================================================

def main():
    args   = BoxMaskExtractor.parse_args()
    runner = BoxMaskExtractor(args)
    runner.run()


if __name__ == "__main__":
    main()

# endregion
