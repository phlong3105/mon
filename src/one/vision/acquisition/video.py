#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import glob
import inspect
import os
import subprocess
import sys
from abc import ABCMeta
from abc import abstractmethod
from pathlib import Path
from typing import Union

import cv2
import ffmpeg
import numpy as np
import torch
from torch import Tensor

from one.core import create_dirs
from one.core import error_console
from one.core import Ints
from one.core import is_image_file
from one.core import is_video_file
from one.core import is_video_stream
from one.core import Tensors
from one.core import to_size
from one.vision.acquisition.image import to_channel_last
from one.vision.acquisition.image import to_image
from one.vision.acquisition.image import to_tensor


# MARK: - Functional

def ffmpeg_read_frame(process, height: int, width: int) -> Tensor:
	"""Read raw bytes from video stream using ffmpeg and return a Tensor.
	
	Arguments:
		process:
			Subprocess that manages ffmpeg.
		height (int):
			Frame height.
		width (int):
			Frame width.
		
	Returns:
		frame (Tensor[1, C, H, W]):
			Tensor frame/image.
	"""
	# Note: RGB24 == 3 bytes per pixel.
	frame_size = height * width * 3
	in_bytes   = process.stdout.read(frame_size)
	if len(in_bytes) == 0:
		frame = None
	else:
		if len(in_bytes) != frame_size:
			raise ValueError()
		frame = (
			np
				.frombuffer(in_bytes, np.uint8)
				.reshape([height, width, 3])
		)  # Numpy
		frame = to_tensor(image=frame, keep_dims=False)
	return frame


def ffmpeg_write_frame(process, frame: Union[Tensor, None]):
	"""Write a `Tensor[1, C, H, W]` to video file using ffmpeg.

	Arguments:
		process:
			Subprocess that manages ffmpeg.
		frame (Tensor[1, C, H, W]):
			`Tensor` frame/image.
	"""
	if isinstance(frame, Tensor):
		frame = to_image(image=frame, keep_dims=False, denormalize=True)
		process.stdin.write(
			frame
				.astype(np.uint8)
				.tobytes()
		)
	else:
		error_console(f"`frame` must be `Tensor`. But got: {type(frame)}.")


# MARK: - Modules

class BaseVideoLoader(metaclass=ABCMeta):
	"""A baseclass/interface for all VideoLoader classes.
	
	Args:
		data (str):
			Data source. Can be a path to an image file, a directory, a video,
			or a stream. It can also be a pathname pattern to images.
		batch_size (int):
			Number of samples in one forward & backward pass. Default: `1`.
		verbose (bool):
			Verbosity mode of video loader backend. Default: `False`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		data      : str,
		batch_size: int = 1,
		verbose   : bool = False,
		*args, **kwargs
	):
		super().__init__()
		self.data       = data
		self.batch_size = batch_size
		self.verbose    = verbose
		self.index      = 0
		
	def __len__(self) -> int:
		"""Return the number of frames in the video.
			>0: if the offline video.
			-1: if the online video.
		"""
		return self.frame_count
	
	def __iter__(self):
		"""Returns an iterator starting at index 0."""
		self.reset()
		return self
	
	@abstractmethod
	def __next__(self) -> tuple[Tensor, list, list, list]:
		"""Load next batch of images.
		
		Returns:
			images (Tensor[B, C, H, W]):
				Images tensor.
			indexes (list):
				List of image indexes.
			files (list):
				List of image files.
			rel_paths (list):
				List of images' relative paths corresponding to data.
		"""
	
	def __del__(self):
		"""Close."""
		self.close()

	# MARK: Properties
	
	def batch_len(self) -> int:
		"""Return the total batches calculated from `batch_size`."""
		return int(self.__len__() / self.batch_size)
	
	@property
	@abstractmethod
	def fourcc(self) -> str:
		"""Return 4-character code of codec."""
		pass
	
	@property
	@abstractmethod
	def fps(self) -> int:
		"""Return frame rate."""
		pass
	
	@property
	@abstractmethod
	def frame_count(self) -> int:
		"""Return number of frames in the video file."""
		pass

	@property
	@abstractmethod
	def frame_height(self) -> int:
		"""Return height of the frames in the video stream."""
		pass
	
	@property
	@abstractmethod
	def frame_width(self) -> int:
		"""Return width of the frames in the video stream."""
		pass
	
	@property
	def is_stream(self) -> bool:
		"""Return `True` if it is a video stream, i.e, unknown `frame_count`.
		"""
		return self.frame_count == -1
	
	@property
	def shape(self) -> Ints:
		"""Return shape of the frames in the video stream in [C, H, W] format.
		"""
		return self.frame_height, self.frame_width
	
	# MARK: Configure
	
	@abstractmethod
	def init(self):
		"""Initialize input."""
		pass
	
	@abstractmethod
	def reset(self):
		"""Reset and start over."""
		pass
	
	@abstractmethod
	def close(self):
		"""Stop and release."""
		pass


class BaseVideoWriter(metaclass=ABCMeta):
	"""A baseclass/interface for all VideoWriter classes.
	
	Args:
		dst (str):
			Output video file or a directory.
		shape (Ints):
			Output size as [C, H, W]. This is also used to reshape the input.
			Default: `(3, 480, 640)`.
		frame_rate (float):
			Frame rate of the video. Default: `10`.
		save_image (bool):
			Should write individual image? Default: `False`.
		save_video (bool):
			Should write video? Default: `True`.
		verbose (bool):
			Verbosity mode of video writer backend. Default: `False`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		dst		  : str,
		shape     : Ints  = (3, 480, 640),
		frame_rate: float = 10,
		save_image: bool  = False,
		save_video: bool  = True,
		verbose   : bool  = False,
		*args, **kwargs
	):
		super().__init__()
		self.dst		= dst
		self.shape      = shape
		self.image_size = to_size(shape)
		self.frame_rate = frame_rate
		self.save_image = save_image
		self.save_video = save_video
		self.verbose    = verbose
		self.index		= 0
	
	def __len__(self) -> int:
		"""Return the number of already written frames."""
		return self.index
	
	def __del__(self):
		"""Close the `video_writer`."""
		self.close()

	# MARK: Configure
	
	@abstractmethod
	def init(self):
		"""Initialize output."""
		pass
	
	@abstractmethod
	def close(self):
		"""Close the `video_writer`."""
		pass
	
	# MARK: Write
	
	@abstractmethod
	def write(self, image: Tensor, image_file: Union[str, None] = None):
		"""Add a frame to writing video.

		Args:
			image (Tensor):
				Image.
			image_file (str, None):
				Image file. Default: `None`.
		"""
		pass
	
	@abstractmethod
	def write_batch(
		self, images: Tensors, image_files: Union[list[str], None] = None
	):
		"""Add batch of frames to video.

		Args:
			images (Tensors):
				List of images.
			image_files (list[str], None):
				Image files. Default: `None`.
		"""
		pass
	

class CVVideoLoader(BaseVideoLoader):
	"""Loads frame(s) from a filepath, a pathname pattern, a directory, a video,
	or a stream using OpenCV.

	Args:
		data (str):
			Data source. Can be a path to an image file, a directory, a video,
			or a stream. It can also be a pathname pattern to images.
		batch_size (int):
			Number of samples in one forward & backward pass. Default: `1`.
		api_preference (int):
			Preferred Capture API backends to use. Can be used to enforce a
			specific reader implementation. Two most used options are:
			[cv2.CAP_ANY=0, cv2.CAP_FFMPEG=1900].
			See more: https://docs.opencv.org/4.5.5/d4/d15/group__videoio__flags__base.html#ggaeb8dd9c89c10a5c63c139bf7c4f5704da7b235a04f50a444bc2dc72f5ae394aaf
			Default: `cv2.CAP_FFMPEG`.
		verbose (bool):
			Verbosity mode of video loader backend. Default: `False`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		data          : str,
		batch_size    : int  = 1,
		api_preference: int  = cv2.CAP_FFMPEG,
		verbose       : bool = False,
		*args, **kwargs
	):
		super().__init__(
			data       = data,
			batch_size = batch_size,
			verbose    = verbose,
			*args, **kwargs
		)
		self.api_preference = api_preference
		self.video_capture  = None
		self.init()
		
	def __next__(self) -> tuple[Tensor, list, list, list]:
		"""Load next batch of images.
		
		Returns:
			images (Tensor[B, C, H, W]):
				Image Tensor.
			indexes (list):
				List of image indexes.
			files (list):
				List of image files.
			rel_paths (list):
				List of images' relative paths corresponding to data.
		"""
		if not self.is_stream and self.index >= self.frame_count:
			self.close()
			raise StopIteration
		else:
			images    = []
			indexes   = []
			files     = []
			rel_paths = []
			
			for i in range(self.batch_size):
				if not self.is_stream and self.index >= self.frame_count:
					break
				
				if isinstance(self.video_capture, cv2.VideoCapture):
					ret_val, image = self.video_capture.read()
					rel_path       = os.path.basename(self.data)
				elif isinstance(self.video_capture, list):
					image    = cv2.imread(self.video_capture[self.index])
					file     = self.video_capture[self.index]
					rel_path = file.replace(self.data, "")
				else:
					raise RuntimeError(
						f"`video_capture` has not been initialized."
					)
				
				if image is None:
					continue
				image = image[:, :, ::-1]  # BGR to RGB
				image = to_tensor(image=image, keep_dims=False)
				
				images.append(image)
				indexes.append(self.index)
				files.append(self.data)
				rel_paths.append(rel_path)
				
				self.index += 1
			
			# return np.array(images), indexes, files, rel_paths
			return torch.stack(images), indexes, files, rel_paths

	# MARK: Properties
	
	@property
	def format(self):  # Flag=8
		"""Return format of the Mat objects (see Mat::type()) returned
		by VideoCapture::retrieve().
		Set value -1 to fetch undecoded RAW video streams (as Mat 8UC1).
		"""
		return self.video_capture.get(cv2.CAP_PROP_FORMAT)
	
	@property
	def fourcc(self) -> str:  # Flag=6
		"""Return 4-character code of codec."""
		return str(self.video_capture.get(cv2.CAP_PROP_FOURCC))
	
	@property
	def fps(self) -> int:  # Flag=5
		"""Return frame rate."""
		return int(self.video_capture.get(cv2.CAP_PROP_FPS))
	
	@property
	def frame_count(self) -> int:  # Flag=7
		"""Return number of frames in the video file."""
		if isinstance(self.video_capture, cv2.VideoCapture):
			return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
		elif isinstance(self.video_capture, list):
			return len(self.video_capture)
		else:
			return -1
	
	@property
	def frame_height(self) -> int:  # Flag=4
		"""Return height of the frames in the video stream."""
		return int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	
	@property
	def frame_width(self) -> int:  # Flag=3
		"""Return width of the frames in the video stream."""
		return int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	
	@property
	def mode(self):  # Flag=10
		"""Return backend-specific value indicating the current capture mode.
		"""
		return self.video_capture.get(cv2.CAP_PROP_MODE)
	
	@property
	def pos_avi_ratio(self) -> int:  # Flag=2
		"""Return relative position of the video file:
		0=start of the film, 1=end of the film.
		"""
		return int(self.video_capture.get(cv2.CAP_PROP_POS_AVI_RATIO))
	
	@property
	def pos_msec(self) -> int:  # Flag=0
		"""Return current position of the video file in milliseconds."""
		return int(self.video_capture.get(cv2.CAP_PROP_POS_MSEC))
	
	@property
	def pos_frames(self) -> int:  # Flag=1
		"""Return 0-based index of the frame to be decoded/captured next."""
		return int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
	
	# MARK: Configure
	
	def init(self):
		"""Initialize `video_capture` object."""
		if is_video_file(self.data) or is_video_stream(self.data):
			self.video_capture = cv2.VideoCapture(self.data, self.api_preference)
			self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, self.batch_size)  # Set buffer (batch) size
		elif is_image_file(self.data):
			self.video_capture = [self.data]
		elif os.path.isdir(self.data):
			self.video_capture = [
				i for i
				in glob.glob(os.path.join(self.data, "**/*"), recursive=True)
				if is_image_file(i)
			]
		elif isinstance(self.data, str):
			self.video_capture = [
				i for i in glob.glob(self.data) if is_image_file(i)
			]
		else:
			raise IOError(f"Do not support data of type: {self.data}.")
	
	# if is_video_file(data):
	# self.video_capture = cv2.VideoCapture(data, api_preference)
	# self.num_frames    = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
	# self.frame_rate    = int(self.video_capture.get(cv2.CAP_PROP_FPS))
	# elif is_video_stream(data):
	# self.video_capture = cv2.VideoCapture(data, api_preference)  # stream
	# Set buffer (batch) size
	# self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, self.batch_size)
	# self.num_frames    = -1
	# if self.video_capture is None:
	# 	raise IOError("Error when reading input stream or video file!")
	
	def reset(self):
		"""Reset and start over."""
		self.index = 0
		if isinstance(self.video_capture, cv2.VideoCapture):
			self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.index)
	
	def close(self):
		"""Stop and release the current `video_capture` object."""
		if isinstance(self.video_capture, cv2.VideoCapture) \
			and self.video_capture:
			self.video_capture.release()
			

class CVVideoWriter(BaseVideoWriter):
	"""Saves frames to individual image files or appends all to a video file.

	Args:
		dst (str):
			Output video file or a directory.
		shape (Ints):
			Output size as [C, H, W]. This is also used to reshape the input.
			Default: `(3, 480, 640)`.
		frame_rate (int):
			Frame rate of the video. Default: `10`.
		fourcc (str):
			Video codec. One of: ["mp4v", "xvid", "mjpg", "wmv"].
			Default: `mp4v`.
		save_image (bool):
			Should write individual image? Default: `False`.
		save_video (bool):
			Should write video? Default: `True`.
		verbose (bool):
			Verbosity mode of video writer backend. Default: `False`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		dst		  : str,
		shape     : Ints  = (3, 480, 640),
		frame_rate: float = 10,
		fourcc    : str   = "mp4v",
		save_image: bool  = False,
		save_video: bool  = True,
		verbose   : bool  = False,
	):
		super().__init__(
			dst        = dst,
			shape      = shape,
			frame_rate = frame_rate,
			save_image = save_image,
			save_video = save_video,
			verbose    = verbose,
		)
		self.fourcc       = fourcc
		self.video_writer = None
		if self.save_video:
			self.init()

	# MARK: Configure

	def init(self):
		"""Initialize output."""
		if os.path.isdir(self.dst):
			parent_dir = self.dst
			video_file = os.path.join(parent_dir, f"result.mp4")
		else:
			parent_dir = str(Path(self.dst).parent)
			stem       = str(Path(self.dst).stem)
			video_file = os.path.join(parent_dir, f"{stem}.mp4")
		create_dirs(paths=[parent_dir])

		fourcc			  = cv2.VideoWriter_fourcc(*self.fourcc)
		self.video_writer = cv2.VideoWriter(
			video_file, fourcc, self.frame_rate, self.image_size[::-1]  # Must be in [W, H]
		)

		if self.video_writer is None:
			raise FileNotFoundError(
				f"Cannot create video file at: {video_file}."
			)

	def close(self):
		"""Close the `video_writer`."""
		if self.video_writer:
			self.video_writer.release()

	# MARK: Write

	def write(self, image: Tensor, image_file: Union[str, None] = None):
		"""Add a frame to writing video.

		Args:
			image (Tensor):
				Image.
			image_file (str, None):
				Image file. Default: `None`.
		"""
		image = to_image(image=image, keep_dims=False, denormalize=True)
		
		if self.save_image:
			if image_file is not None:
				image_file = (image_file[1:] if image_file.startswith("\\")
				              else image_file)
				image_name = os.path.splitext(image_file)[0]
			else:
				image_name = f"{self.index}"
			parent_dir  = self.dst.split(".")[0]
			output_file = os.path.join(parent_dir, f"{image_name}.png")
			create_dirs(paths=[parent_dir])
			cv2.imwrite(output_file, image)
		if self.save_video:
			self.video_writer.write(image)

		self.index += 1

	def write_batch(
		self, images: Tensors, image_files: Union[list[str], None] = None
	):
		"""Add batch of frames to video.

		Args:
			images (Tensors):
				Images.
			image_files (list[str], None):
				Image files. Default: `None`.
		"""
		if image_files is None:
			image_files = [None for _ in range(len(images))]

		for image, image_file in zip(images, image_files):
			self.write(image=image, image_file=image_file)


class FFmpegVideoLoader(BaseVideoLoader):
	"""Loads frame(s) from a filepath, a pathname pattern, a directory, a video,
	or a stream using FFmpeg.
	
	Args:
		data (str):
			Data source. Can be a path to an image file, pathname pattern to 
			image files, a directory, a video, or a stream.
		batch_size (int):
			Number of samples in one forward & backward pass. Default: `1`.
		verbose (bool):
			Verbosity mode of video loader backend. Default: `False`.
		kwargs:
			Any supplied kwargs are passed to ffmpeg verbatim.
		
	References:
		https://github.com/kkroening/ffmpeg-python/tree/master/examples
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		data      : str,
		batch_size: int  = 1,
		verbose   : bool = False,
		*args, **kwargs
	):
		super().__init__(
			data       = data,
			batch_size = batch_size,
			verbose    = verbose,
			*args, **kwargs
		)
		self.ffmpeg_cmd     = None
		self.ffmpeg_process = None
		self.ffmpeg_kwargs  = kwargs
		self.video_info     = None
		self.init()
	
	def __next__(self) -> tuple[Tensor, list, list, list]:
		"""Load next batch of images.
	
		Returns:
			images (Tensor[B, C, H, W]):
				Image Tensor.
			indexes (list):
				List of image indexes.
			files (list):
				List of image files.
			rel_paths (list):
				List of images' relative paths corresponding to data.
		"""
		if not self.is_stream and self.index >= self.frame_count:
			self.close()
			raise StopIteration
		else:
			images    = []
			indexes   = []
			files     = []
			rel_paths = []
			
			for i in range(self.batch_size):
				if not self.is_stream and self.index >= self.frame_count:
					break
				
				if self.ffmpeg_process:
					image = ffmpeg_read_frame(
						process = self.ffmpeg_process,
						width   = self.frame_width,
						height  = self.frame_height
					)  # Already in RGB
					rel_path = os.path.basename(self.data)
				else:
					raise RuntimeError(
						f"`video_capture` has not been initialized."
					)
				
				images.append(image)
				indexes.append(self.index)
				files.append(self.data)
				rel_paths.append(rel_path)
				
				self.index += 1
			
			# return np.array(images), indexes, files, rel_paths
			return torch.stack(images), indexes, files, rel_paths
		
	# MARK: Properties

	@property
	def fourcc(self) -> str:
		"""Return 4-character code of codec."""
		return self.video_info["codec_name"]
	
	@property
	def fps(self) -> int:
		"""Return frame rate."""
		return int(self.video_info["avg_frame_rate"].split("/")[0])
	
	@property
	def frame_count(self) -> int:
		"""Return number of frames in the video file."""
		if is_video_file(self.data):
			return int(self.video_info["nb_frames"])
		else:
			return -1
	
	@property
	def frame_width(self) -> int:
		"""Return width of the frames in the video stream."""
		return int(self.video_info["width"])
	
	@property
	def frame_height(self) -> int:
		"""Return height of the frames in the video stream."""
		return int(self.video_info["height"])
	
	# MARK: Configure
	
	def init(self):
		"""Initialize ffmpeg cmd."""
		probe           = ffmpeg.probe(self.data, **self.ffmpeg_kwargs)
		self.video_info = next(
			s for s in probe["streams"] if s["codec_type"] == "video"
		)
		if self.verbose:
			self.ffmpeg_cmd = (
				ffmpeg
					.input(self.data, **self.ffmpeg_kwargs)
					.output("pipe:", format="rawvideo", pix_fmt="rgb24")
					.compile()
			)
		else:
			self.ffmpeg_cmd = (
				ffmpeg
					.input(self.data, **self.ffmpeg_kwargs)
					.output("pipe:", format="rawvideo", pix_fmt="rgb24")
					.global_args("-loglevel", "quiet")
					.compile()
			)
	
	def reset(self):
		"""Reset and start over."""
		self.close()
		self.index = 0
		if self.ffmpeg_cmd:
			self.ffmpeg_process = subprocess.Popen(
				self.ffmpeg_cmd,
				stdout  = subprocess.PIPE,
				bufsize = 10**8
			)
	
	def close(self):
		"""Stop and release the current `ffmpeg_process`."""
		if self.ffmpeg_process and self.ffmpeg_process.poll() is not None:
			# os.killpg(os.getpgid(self.ffmpeg_process.pid), signal.SIGTERM)
			self.ffmpeg_process.terminate()
			self.ffmpeg_process.wait()
			self.ffmpeg_process = None
			# raise StopIteration


class FFmpegVideoWriter(BaseVideoWriter):
	"""Saves frames to individual image files or appends all to a video file.

	Args:
		dst (str):
			Output video file or a directory.
		shape (Int3T):
			Output size as [C, H, W]. This is also used to reshape the input.
			Default: `(3, 480, 640)`.
		frame_rate (int):
			Frame rate of the video. Default: `10`.
		pix_fmt (str):
			Video codec. Default: `yuv420p`.
		save_image (bool):
			Should write individual image? Default: `False`.
		save_video (bool):
			Should write video? Default: `True`.
		verbose (bool):
			Verbosity mode of video loader backend. Default: `False`.
		ffmpeg_process (subprocess.Popen):
			Subprocess that manages ffmpeg.
		index (int):
			Current index.
		kwargs:
			Any supplied kwargs are passed to ffmpeg verbatim.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		dst		  : str,
		shape     : Ints  = (3, 480, 640),
		frame_rate: float = 10,
		pix_fmt   : str   = "yuv420p",
		save_image: bool  = False,
		save_video: bool  = True,
		verbose   : bool  = False,
		*args, **kwargs
	):
		super().__init__(
			dst        = dst,
			shape      = shape,
			frame_rate = frame_rate,
			save_image = save_image,
			save_video = save_video,
			verbose    = verbose,
			*args, **kwargs
		)
		self.pix_fmt        = pix_fmt
		self.ffmpeg_process = None
		self.ffmpeg_kwargs  = kwargs
		if self.save_video:
			self.init()

	# MARK: Configure

	def init(self):
		"""Initialize output."""
		if os.path.isdir(self.dst):
			parent_dir = self.dst
			video_file = os.path.join(parent_dir, f"result.mp4")
		else:
			parent_dir = str(Path(self.dst).parent)
			stem       = str(Path(self.dst).stem)
			video_file = os.path.join(parent_dir, f"{stem}.mp4")
		create_dirs(paths=[parent_dir])
		
		if self.verbose:
			self.ffmpeg_process = (
				ffmpeg
					.input("pipe:", format="rawvideo", pix_fmt="rgb24", s="{}x{}".format(self.image_size[1], self.image_size[0]))
					.output(video_file, pix_fmt=self.pix_fmt, **self.ffmpeg_kwargs)
					.overwrite_output()
					.run_async(pipe_stdin=True)
			)
		else:
			self.ffmpeg_process = (
				ffmpeg
					.input("pipe:", format="rawvideo", pix_fmt="rgb24", s="{}x{}".format(self.image_size[1], self.image_size[0]))
					.output(video_file, pix_fmt=self.pix_fmt, **self.ffmpeg_kwargs)
					.global_args("-loglevel", "quiet")
					.overwrite_output()
					.run_async(pipe_stdin=True)
			)
		
	def close(self):
		"""Stop and release the current `ffmpeg_process`."""
		if self.ffmpeg_process:
			self.ffmpeg_process.stdin.close()
			self.ffmpeg_process.terminate()
			self.ffmpeg_process.wait()
			self.ffmpeg_process = None
	
	# MARK: Write

	def write(self, image: Tensor, image_file: Union[str, None] = None):
		"""Add a frame to writing video.

		Args:
			image (Tensor[C, H, W]):
				Image.
			image_file (str, None):
				Image file. Default: `None`.
		"""
		image = to_channel_last(image)

		if self.save_image:
			if image_file is not None:
				image_file = (image_file[1:] if image_file.startswith("\\") else image_file)
				image_name = os.path.splitext(image_file)[0]
			else:
				image_name = f"{self.index}"
			parent_dir  = self.dst.split(".")[0]
			output_file = os.path.join(parent_dir, f"{image_name}.png")
			create_dirs(paths=[parent_dir])
			cv2.imwrite(output_file, image)
		
		if self.save_video and self.ffmpeg_process:
			ffmpeg_write_frame(process=self.ffmpeg_process, frame=image)

		self.index += 1

	def write_batch(
		self, images: Tensors, image_files: Union[list[str], None] = None
	):
		"""Add batch of frames to video.

		Args:
			images (Tensors):
				Images.
			image_files (list[str], None):
				Image files. Default: `None`.
		"""
		if image_files is None:
			image_files = [None for _ in range(len(images))]

		for image, image_file in zip(images, image_files):
			self.write(image=image, image_file=image_file)


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
