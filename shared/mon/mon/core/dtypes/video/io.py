#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements video I/O operations using ``cv2`` and ``ffmpeg``."""

__all__ = [
    "load_video_ffmpeg",
    "write_video_ffmpeg",
]

from typing import Union

import numpy as np
import torch


# ----- Reading -----
def load_video_ffmpeg(process, height: int, width: int) -> np.ndarray:
    """Read video frame bytes using ``ffmpeg``.

    Args:
        process: Subprocess managing ``ffmpeg`` instance as ``subprocess.Popen``.
        height: Video frame height.
        width: Video frame width.

    Returns:
        Frame as a ``numpy.ndarray`` of shape :math:`(H, W, C)` in range :math:`[0, 255]`,
        or ``None`` if no data.

    Raises:
        ValueError: If read bytes do not match expected frame size.
    """
    # RGB24: 3 bytes per pixel
    img_size = height * width * 3
    in_bytes = process.stdout.read(img_size)
    if len(in_bytes) == 0:
        image = None
    else:
        if len(in_bytes) != img_size:
            raise ValueError(f"``in_bytes`` length [{len(in_bytes)}] != expected size [{img_size}].")
        image = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
    return image


# ----- Writing -----
def write_video_ffmpeg(process, frame: Union[torch.Tensor, np.ndarray]):
    """Write frame to video using ``ffmpeg``.

    Args:
        process: Subprocess managing ``ffmpeg`` as ``subprocess.Popen``.
        frame: Frame/image as a ``numpy.ndarray`` of shape :math:`(H, W, C)`.

    Raises:
        ValueError: If ``frame`` is not a ``numpy.ndarray``.
    """
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"``frame`` must be a numpy.ndarray, got {type(frame).__name__}.")
    process.stdin.write(
        frame
        .astype("uint8")
        .tobytes()
    )
    return None
