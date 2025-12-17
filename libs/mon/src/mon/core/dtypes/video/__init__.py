#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for video data type.

This package provides data structures and utilities for handling video data,
including frame representation and video writing capabilities using different
backends.
"""

__all__ = [
    "Frame",
    "VideoWriter",
    "VideoWriterCV",
    "VideoWriterFFmpeg",
    "load_video_ffmpeg",
    "write_video_ffmpeg",
]

from .core import Frame
from .io import (
    VideoWriter,
    VideoWriterCV,
    VideoWriterFFmpeg,
    load_video_ffmpeg,
    write_video_ffmpeg,
)
