#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements data structure and processing functions for videos."""

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
