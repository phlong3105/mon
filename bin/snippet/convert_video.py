#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script converts video."""

from __future__ import annotations

import click
import cv2

import mon
import mon.vision.core.image


# region Function

@click.command()
@click.option("--source",      default=mon.DATA_DIR/"aic23-autocheckout"/"testA"/"inpainting", type=click.Path(exists=True), help="Video filepath or directory.")
@click.option("--destination", default=mon.DATA_DIR/"aic23-autocheckout"/"train"/"tray", type=click.Path(exists=False), help="Output video filepath or directory.")
@click.option("--start-frame", default=None, type=int, help="Start/end frame.")
@click.option("--end-frame",   default=None, type=int, help="Start/end frame.")
@click.option("--size",        default=None, type=int, nargs="+", help="Output images/video size.")
@click.option("--skip",        default=60, type=int, help="Skip n frames.")
@click.option("--save-image",  default=True, is_flag=True, help="Save images.")
@click.option("--extension",   default="png", type=click.Choice(["jpg", "png"], case_sensitive=False), help="Image extension.")
@click.option("--verbose",     is_flag=True)
def convert_video(
    source     : mon.Path,
    destination: mon.Path,
    start_frame: int,
    end_frame  : int,
    size       : int | list[int],
    skip       : int,
    save_image : bool,
    extension  : str,
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
    if save_image:
        destination = [d.parent / d.stem for d in destination]
    
    if size is not None:
        size = mon.vision.core.image.get_hw(size=size)
    
    skip = skip or 1
    
    for src, dst in zip(source, destination):
        cap         = cv2.VideoCapture(str(src))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps         = cap.get(cv2.CAP_PROP_FPS)
        w           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        start_frame = start_frame or 0
        end_frame   = end_frame or frame_count
        if not end_frame >= start_frame:
            raise ValueError(
                f"end_frame must >= start_frame, but got {start_frame} and "
                f"{end_frame}."
            )
        
        if not save_image:
            dst.parent.mkdir(parents=True, exist_ok=True)
            wrt = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        else:
            dst.mkdir(parents=True, exist_ok=True)
            
        with mon.get_progress_bar() as pbar:
            for i in pbar.track(
                sequence    = range(frame_count),
                total       = frame_count,
                description = f"[bright_yellow] Converting {src.name}"
            ):
                success, image = cap.read()
                if not success or not start_frame <= i <= end_frame or i % skip != 0:
                    continue
                
                if size is not None:
                    image = cv2.resize(image, size)
                    
                if save_image:
                    cv2.imwrite(str(dst/f"{i:06}.{extension}"), image)
                else:
                    wrt.write(image)
               
                if verbose:
                    cv2.imshow("Image", image)
                    if cv2.waitKey(1) == ord("q"):
                        break
            
# endregion


# region Main

if __name__ == "__main__":
    convert_video()

# endregion
