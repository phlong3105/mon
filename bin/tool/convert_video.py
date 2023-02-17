#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script converts video."""

from __future__ import annotations

import click
import cv2

import mon


# region Function

@click.command()
@click.option("--source",      default=mon.DATA_DIR/"aic23-checkout/testA/convert/testA_1.mp4", type=click.Path(exists=True), help="Video filepath or directory.")
@click.option("--destination", default=mon.DATA_DIR/"aic23-checkout/testA/convert/testA_1-short.mp4", type=click.Path(exists=False), help="Output video filepath or directory.")
@click.option("--from-index",  default=None, type=int, help="From/to frame index.")
@click.option("--to-index",    default=None, type=int, help="From/to frame index.")
@click.option("--size",        default=None, type=int, nargs="+", help="Output images' sizes.")
@click.option("--verbose",     is_flag=True)
def extract_video_image(
    source     : mon.Path,
    destination: mon.Path,
    from_index : int,
    to_index   : int,
    size       : int | list[int],
    verbose    : bool
):
    assert source is not None and (mon.Path(source).is_video_file() or mon.Path(source).is_dir())

    source = mon.Path(source)
    source = [source] if mon.Path(source).is_video_file() else list(source.glob("*"))
    source = [s for s in source if s.is_video_file()]
    
    if destination is not None:
        destination = mon.Path(destination)
        if destination.suffix in mon.VideoFormat.str_mapping():
            destination = [destination]
        else:
            destination = [destination / f"{s.stem}.mp4" for s in source]
    else:
        destination = [s.parent / f"{s.stem}-convert.mp4" for s in source]
   
    if size is not None:
        size = mon.get_hw(size=size)
        
    for src, dst in zip(source, destination):
        dst.parent.mkdir(parents=True, exist_ok=True)
        cap         = cv2.VideoCapture(str(src))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps         = cap.get(cv2.CAP_PROP_FPS)
        w           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        from_index  = from_index or 0
        to_index    = to_index   or frame_count
        if not to_index >= from_index:
            raise ValueError(
                f"to_index must >= to from_index, but got {from_index} and "
                f"{to_index}."
            )
        
        wrt = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        with mon.get_progress_bar() as pbar:
            for i in pbar.track(
                sequence    = range(frame_count),
                total       = frame_count,
                description = f"[bright_yellow] Extracting {src.name}"
            ):
                ret, image = cap.read()
                if not ret:
                    continue
                if i < from_index or i > to_index:
                    continue
                if size is not None:
                    image = cv2.resize(image, size)
                wrt.write(image)
                if verbose:
                    cv2.imshow("Image", image)
                    cv2.waitKey(0)
            
# endregion


# region Main

if __name__ == "__main__":
    extract_video_image()

# endregion
