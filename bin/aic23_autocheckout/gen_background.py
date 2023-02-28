#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script performs background subtraction."""

from __future__ import annotations

import click
import cv2

import mon


# region Function

@click.command()
@click.option("--source",      default=mon.DATA_DIR/"aic23-autocheckout/testA/inpainting/", type=click.Path(exists=True), help="Video path or directory.")
@click.option("--destination", default=mon.DATA_DIR/"aic23-autocheckout/testA/background/", type=click.Path(exists=False), help="Output video filepath or directory.")
@click.option("--start-frame", default=None, type=int, help="Start/end frame.")
@click.option("--end-frame",   default=None, type=int, help="Start/end frame.")
@click.option("--verbose",     is_flag=True)
def convert_video(
    source     : mon.Path,
    destination: mon.Path,
    start_frame: int,
    end_frame  : int,
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
    
    for src, dst in zip(source, destination):
        dst.parent.mkdir(parents=True, exist_ok=True)
        cap         = cv2.VideoCapture(str(src))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps         = cap.get(cv2.CAP_PROP_FPS)
        w           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        start_frame  = start_frame or 0
        end_frame    = end_frame or frame_count
        if not end_frame >= start_frame:
            raise ValueError(
                f"start_frame must >= end_frame, but got {start_frame} and "
                f"{end_frame}."
            )
        
        # fg_dst = dst.parent / f"{dst.stem}-fg.mp4"
        bg_dst = dst.parent / f"{dst.stem}-bg.mp4"
        # fg_wrt = cv2.VideoWriter(str(fg_dst), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        bg_wrt = cv2.VideoWriter(str(bg_dst), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        bg_sub = cv2.createBackgroundSubtractorMOG2()
        
        with mon.get_progress_bar() as pbar:
            for i in pbar.track(
                sequence    = range(frame_count),
                total       = frame_count,
                description = f"[bright_yellow] Background subtracting {src.name}"
            ):
                ret, image = cap.read()
                if not ret:
                    continue
                if i < start_frame or i > end_frame:
                    continue

                fg = bg_sub.apply(image)
                bg = bg_sub.getBackgroundImage()
                # fg_wrt.write(fg)
                bg_wrt.write(bg)
                if verbose:
                    # cv2.imshow("Foreground", fg)
                    cv2.imshow("Background", bg)
                    if cv2.waitKey(1) == ord("q"):
                        break
            
# endregion


# region Main

if __name__ == "__main__":
    convert_video()

# endregion
