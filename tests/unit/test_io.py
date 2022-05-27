#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test CV IO operations.
"""

import cv2

from one import FFmpegVideoLoader
from one import FFmpegVideoWriter
from one import load_file
from one import merge_files


# MARK: - Test Video IO

def test_ffmpeg_video_loader():
	video_loader = FFmpegVideoLoader(data="../../data/demo.mp4")
	for imgs, idxes, files, rel_paths in video_loader:
		for img in imgs:
			cv2.imshow("Image", img)
			if cv2.waitKey(1) == 27:
				video_loader.close()
				cv2.destroyAllWindows()
				break


def test_ffmpeg_video_writer():
	video_loader = FFmpegVideoLoader(data="../../data/demo.mp4")
	video_writer = FFmpegVideoWriter(
		dst        ="../../data/results.mp4",
		shape      = video_loader.shape,
		frame_rate = 30,
		pix_fmt    = "yuv420p",
		save_image = False,
		save_video = True,
		verbose    = False,
	)
	for imgs, idxes, files, rel_paths in video_loader:
		for img in imgs:
			cv2.imshow("Image", img)
			video_writer.write(img)
			if cv2.waitKey(1) == 27:
				video_loader.close()
				video_writer.close()
				cv2.destroyAllWindows()
				break


# MARK: - Test File IO

def test_merge_files():
	file1 = "../../data/predictions0.pkl"
	file2 = "../../data/predictions1.pkl"
	merge_files(
		in_paths    = [file1, file2],
		out_path    = "../../data/predictions.pkl",
		file_format = "pkl"
	)
	

def test_pickle_loader():
	data = load_file("../../data/sample_val_predictions.pkl")
	print(data["May"])
