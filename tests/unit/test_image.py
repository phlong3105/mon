#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test image operations.
"""

from __future__ import annotations

import cv2

from one.vision.acquisition import to_image, read_image
from one.vision.acquisition import VideoLoaderFFmpeg
from one.vision.acquisition import VideoWriterFFmpeg


def test_image_loader():
    
    pass


def test_video_loader_ffmpeg():
    video_loader = VideoLoaderFFmpeg(data="data/demo.mp4")
    for images, indexes, files, rel_paths in video_loader:
        for img in images:
            img = to_image(image=img, keep_dims=False, denormalize=True)
            cv2.imshow("Image", img)
            if cv2.waitKey(1) == 27:
                video_loader.close()
                cv2.destroyAllWindows()
                break


def test_video_writer_ffmpeg():
    video_loader = VideoLoaderFFmpeg(data="data/demo.mp4")
    video_writer = VideoWriterFFmpeg(
        dst        = "data/results.mp4",
        shape      = video_loader.shape,
        frame_rate = 30,
        pix_fmt    = "yuv420p",
        save_image = False,
        save_video = True,
        verbose    = False,
    )
    for images, indexes, files, rel_paths in video_loader:
        for img in images:
            video_writer.write(img)
            img = to_image(image=img, keep_dims=False, denormalize=True)
            cv2.imshow("Image", img)
            if cv2.waitKey(1) == 27:
                video_loader.close()
                video_writer.close()
                cv2.destroyAllWindows()
                break
