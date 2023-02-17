#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script performs background subtraction."""

from __future__ import annotations

import click
import cv2

import mon


# region Function

@click.command()
@click.option("--source",     default=mon.DATA_DIR/"aic23-checkout/testA/inpainting/", type=click.Path(exists=True), help="Video path or directory.")
@click.option("--destination", default=mon.DATA_DIR/"aic23-checkout/testA/background/", type=click.Path(exists=False), help="Output video filepath or directory.")
@click.option("--from-index", default=None, type=int, help="From/to frame index.")
@click.option("--to-index",   default=None, type=int, help="From/to frame index.")
@click.option("--verbose",    default=True, is_flag=True)
def convert_video(
    source     : mon.Path,
    destination: mon.Path,
    from_index : int,
    to_index   : int,
    verbose    : bool
):
    """Visualize bounding boxes on images."""
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
    
    for src, dst in zip(source, destination):
        dst.parent.mkdir(parents=True, exist_ok=True)
        cap         = cv2.VideoCapture(str(src))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps         = cap.get(cv2.CAP_PROP_FPS)
        w           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        from_index  = from_index or 0
        to_index    = to_index or frame_count
        if not to_index >= from_index:
            raise ValueError(
                f"to_index must >= to from_index, but got {from_index} and "
                f"{to_index}."
            )
        
        bg_dst = dst.parent / f"{dst.stem}-bg.mp4"
        fg_dst = dst.parent / f"{dst.stem}-fg.mp4"
        fg_wrt = cv2.VideoWriter(str(fg_dst), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        bg_wrt = cv2.VideoWriter(str(bg_dst), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        bg_sub = cv2.createBackgroundSubtractorMOG2()
        
        with mon.get_progress_bar() as pbar:
            for i in pbar.track(
                sequence    = range(frame_count),
                total       = frame_count,
                description = f"[bright_yellow] Processing {src.name}"
            ):
                ret, image = cap.read()
                if not ret:
                    continue
                if i < from_index or i > to_index:
                    continue

                fg = bg_sub.apply(image)
                bg = bg_sub.getBackgroundImage()
                fg_wrt.write(fg)
                bg_wrt.write(bg)
                if verbose:
                    cv2.imshow("Foreground", fg)
                    cv2.imshow("Background", bg)
                    if cv2.waitKey(1) == ord("q"):
                        break
            
# endregion


# region Main

if __name__ == "__main__":
    convert_video()

# endregion
