#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model weights I/O operations.

This module provides input and output operations for model weights.
"""

from __future__ import annotations

__all__ = [
    "list_weights",
]

from mon.core.pathlib import Path


# ==============================================================================
# region DISCOVERY
# ==============================================================================

def list_weights(root: Path | str) -> list[Path]:
    """List all weights files in the given root directory.

    Args:
        root: Root directory to search for weights files.

    Returns:
        List of paths to weights files.
    """
    root = Path(root).normalize()
    if not root.exists():
        return []
    # Optimization: rglob with specific extensions if is_weights_file permits
    # Otherwise, stick to * but ensure it's a file
    return [f for f in root.rglob("*") if f.is_weights_file(exist=True)]

# endregion


# ==============================================================================
# region CONNECTION
# ==============================================================================


# endregion


# ==============================================================================
# region INPUT
# ==============================================================================


# endregion


# ==============================================================================
# region OUTPUT
# ==============================================================================


# endregion
