#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script converts video."""

from __future__ import annotations

import click
import cv2

import mon


# region Function

@click.command()
@click.option("--source",      default=mon.DATA_DIR, type=click.Path(exists=True), help="Video filepath or directory.")
@click.option("--destination", default=mon.DATA_DIR, type=click.Path(exists=False), help="Output video filepath or directory.")
@click.option("--from-index",  default=None, type=int, help="From/to frame index.")
@click.option("--to-index",    default=None, type=int, help="From/to frame index.")
@click.option("--size",        default=None, type=int, nargs="+", help="Output images/video size.")
@click.option("--save-image",  is_flag=True, help="Save images.")
@click.option("--extension",   default="png", type=click.Choice(["jpg", "png"], case_sensitive=False), help="Image extension.")
@click.option("--verbose",     is_flag=True)
def convert_video(
    source     : mon.Path,
    destination: mon.Path,
    from_index : int,
    to_index   : int,
    size       : int | list[int],
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
        size = mon.get_hw(size=size)
    
    for src, dst in zip(source, destination):
        cap         = cv2.VideoCapture(str(src))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps         = cap.get(cv2.CAP_PROP_FPS)
        w           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_index     = from_index or 0
        t_index     = to_index   or frame_count
        if not t_index >= f_index:
            raise ValueError(
                f"t_index must >= f_index, but got {f_index} and {t_index}."
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
                if not success or not f_index <= i <= t_index:
                    continue
                
                if size is not None:
                    image = cv2.resize(image, size)
                    
                if save_image:
                    cv2.imwrite(dst/f"{i:06}.{extension}", image)
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
