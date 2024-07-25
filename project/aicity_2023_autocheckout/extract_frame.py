#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script extracts frames from a video using ffmpeg."""

from __future__ import annotations

import click
import ffmpeg

import mon
import mon.vision.core.image


# region Function

@click.command()
@click.option("--source",      default=mon.DATA_DIR/"aic23-autocheckout/testA/", type=click.Path(exists=True), help="Video filepath or directory.")
@click.option("--destination", default=mon.DATA_DIR/"aic23-autocheckout/testA/convert", type=click.Path(exists=False), help="Output video filepath or directory.")
@click.option("--size",        default=None, type=int, nargs="+", help="Output images/video size.")
@click.option("--extension",   default="png", type=click.Choice(["jpg", "png"], case_sensitive=False), help="Image extension.")
@click.option("--verbose",     is_flag=True)
def convert_video(
    source     : mon.Path,
    destination: mon.Path,
    size       : int | list[int],
    extension  : str,
    verbose    : bool
):
    assert source is not None and (mon.Path(source).is_video_file() or mon.Path(source).is_dir())

    source = mon.Path(source)
    source = [source] if mon.Path(source).is_video_file() else list(source.glob("*"))
    source = [s for s in source if s.is_video_file()]
    
    if destination is not None:
        destination = mon.Path(destination)
        destination = [destination / f"{s.stem}" for s in source]
    else:
        destination = [s.parent / f"{s.stem}-convert" for s in source]
    
    if size is not None:
        size = mon.vision.core.image.parse_hw(size=size)
    
    for src, dst in zip(source, destination):
        dst.mkdir(parents=True, exist_ok=True)
        (
            ffmpeg
            .input(str(src))
            .output(str(dst / f"%06d.{extension}"))
            .run()
        )
        
# endregion


# region Main

if __name__ == "__main__":
    convert_video()

# endregion
