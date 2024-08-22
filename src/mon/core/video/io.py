#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Video I/O.

This module implements the basic I/O functionalities of video data.
"""

from __future__ import annotations

__all__ = [
	"VideoWriter",
	"VideoWriterCV",
	"VideoWriterFFmpeg",
	"read_video_ffmpeg",
	"write_video_ffmpeg",
]

import abc
from abc import abstractmethod

import cv2
import ffmpeg
import numpy as np
import torch

from mon.core import image as ci, pathlib
from mon.core.typing import _size_2_t


# region Read

def read_video_ffmpeg(
	process,
	height   : int,
	width    : int,
	to_tensor: bool = False,
	normalize: bool = False,
) -> torch.Tensor | np.ndarray:
	"""Read raw bytes from a video stream using :mod`ffmpeg`. Optionally,
	convert it to :obj:`torch.Tensor` type of shape `[1, C, H, W]`.
	
	Args:
		process: The subprocess that manages :obj:`ffmpeg` instance.
		height: The height of the video frame.
		width: The width of the video.
		to_tensor: If ``True`` convert the image from :obj:`numpy.ndarray` to
			:obj:`torch.Tensor`. Default: ``False``.
		normalize: If ``True``, normalize the image to ``[0.0, 1.0]``.
			Default: ``False``.
	
	Return:
		A :obj:`numpy.ndarray` image of shape `[H, W, C]` with value in
		range ``[0, 255]`` or a :obj:`torch.Tensor` image of shape
		`[1, C, H, W]` with value in range ``[0.0, 1.0]``.
	"""
	# RGB24 == 3 bytes per pixel.
	img_size = height * width * 3
	in_bytes = process.stdout.read(img_size)
	if len(in_bytes) == 0:
		image = None
	else:
		if len(in_bytes) != img_size:
			raise ValueError()
		image = (
			np
			.frombuffer(in_bytes, np.uint8)
			.reshape([height, width, 3])
		)  # Numpy
		if to_tensor:
			image = ci.to_image_tensor(
				image     = image,
				keepdim   = False,
				normalize = normalize
			)
	return image

# endregion


# region Write

def write_video_ffmpeg(
	process,
	image      : torch.Tensor | np.ndarray,
	denormalize: bool = False
):
	"""Write an image to a video file using :obj:`ffmpeg`.

	Args:
		process: A subprocess that manages :obj:``ffmpeg``.
		image: A frame/image of shape `[1, C, H, W]`.
		denormalize: If ``True``, convert image to ``[0, 255]``. Default: ``False``.
	"""
	if isinstance(image, np.ndarray):
		if ci.is_normalized_image(image=image):
			image = ci.denormalize_image(image=image)
		if ci.is_channel_first_image(image=image):
			image = ci.to_channel_last_image(image=image)
	elif isinstance(image, torch.Tensor):
		image = ci.to_image_nparray(
			image= image,
			keepdim     = False,
			denormalize = denormalize
		)
	else:
		raise ValueError(
			f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
			f"but got {type(image)}."
		)
	process.stdin.write(
		image
		.astype(np.uint8)
		.tobytes()
	)


class VideoWriter(abc.ABC):
	"""The base class for all video writers.

	Args:
		dst: A directory to save images.
		image_size: A desired output size of shape `[H, W]`. This is used to
			reshape the input. Default: ``[480, 640]``.
		frame_rate: A frame rate of the output video. Default: ``10``.
		save_image: If ``True`` save each image separately. Default: ``False``.
		denormalize: If ``True``, convert image to ``[0, 255]``.
			Default: ``False``.
		verbose: Verbosity. Default: ``False``.
	"""
	
	def __init__(
		self,
		dst		   : pathlib.Path,
		image_size : _size_2_t = [480, 640],
		frame_rate : float 	   = 10,
		save_image : bool      = False,
		denormalize: bool      = False,
		verbose    : bool      = False,
		*args, **kwargs
	):
		self.dst         = pathlib.Path(dst)
		self.denormalize = denormalize
		self.index       = 0
		self.image_size  = ci.parse_hw(size=image_size)
		self.frame_rate  = frame_rate
		self.save_image  = save_image
		self.verbose     = verbose
		self.init()
		
	def __len__(self) -> int:
		"""Return the number frames of already written frames."""
		return self.index
	
	def __del__(self):
		"""Close."""
		self.close()
	
	@abstractmethod
	def init(self):
		"""Initialize the output handler."""
		pass
	
	@abstractmethod
	def close(self):
		"""Close."""
		pass
	
	@abstractmethod
	def write(
		self,
		frame      : torch.Tensor | np.ndarray,
		path       : pathlib.Path = None,
		denormalize: bool 		  = False
	):
		"""Write an image to :obj:`dst`.

		Args:
			frame: A video frame.
			path: An image file path with an extension. Default: ``None``.
			denormalize: If ``True``, convert image to ``[0, 255]``.
				Default: ``False``.
		"""
		pass
	
	@abstractmethod
	def write_batch(
		self,
		frames     : list[torch.Tensor | np.ndarray],
		paths      : list[pathlib.Path] = None,
		denormalize: bool			 	= False
	):
		"""Write a batch of images to :obj:`dst`.

		Args:
			frames: A :obj:`list` of video frames.
			paths: A :obj:`list` of image file paths with extensions.
				Default: ``None``.
			denormalize: If ``True``, convert image to ``[0, 255]``.
				Default: ``False``.
		"""
		pass
	

class VideoWriterCV(VideoWriter):
	"""A video writer that writes images to a video file using :obj:`cv2`.

	Args:
		dst: A destination directory to save images.
		image_size: A desired output size of shape `[H, W]`. This is used
			to reshape the input. Default: `[480, 640]`.
		frame_rate: A frame rate of the output video. Default: ``10``.
		fourcc: Video codec. One of ``'mp4v'``, ``'xvid'``, ``'mjpg'``, or
			``'wmv'``. Default: ``'.mp4v'``.
		save_image: If ``True``, save each image separately. Default: ``False``.
		denormalize: If ``True``, convert image to ``[0, 255]``.
			Default: ``False``.
		verbose: Verbosity. Default: ``False``.
	"""
	
	def __init__(
		self,
		dst		   : pathlib.Path,
		image_size : _size_2_t = [480, 640],
		frame_rate : float     = 30,
		fourcc     : str       = "mp4v",
		save_image : bool      = False,
		denormalize: bool      = False,
		verbose    : bool      = False,
		*args, **kwargs
	):
		self.fourcc       = fourcc
		self.video_writer = None
		super().__init__(
			dst			= dst,
			image_size  = image_size,
			frame_rate  = frame_rate,
			save_image  = save_image,
			denormalize = denormalize,
			verbose     = verbose,
			*args, **kwargs
		)
	
	def init(self):
		"""Initialize the output destination."""
		if self.dst.is_dir():
			video_file = self.dst / f"result.mp4"
		else:
			video_file = self.dst.parent / f"{self.dst.stem}.mp4"
		video_file.parent.mkdir(parents=True, exist_ok=True)
		
		fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
		self.video_writer = cv2.VideoWriter(
			filename  = str(video_file),
			fourcc    = fourcc,
			fps       = float(self.frame_rate),
			frameSize =self.image_size[::-1],  # Must be in [W, H]
			isColor   = True
		)
		
		if self.video_writer is None:
			raise FileNotFoundError(f"Cannot create video file at {video_file}.")
	
	def close(self):
		"""Close the :obj:`video_writer`."""
		if self.video_writer:
			self.video_writer.release()
	
	def write(
		self,
		frame      : torch.Tensor | np.ndarray,
		path       : pathlib.Path = None,
		denormalize: bool 		  = False
	):
		"""Write an image to :obj:`dst`.

		Args:
			frame: An image.
			path: An image file path with an extension. Default: ``None``.
			denormalize: If ``True``, convert image to ``[0, 255]``.
				Default: ``False``.
		"""
		if self.save_image:
			ci.write_image_cv(
				image       = frame,
				dir_path    = self.dst,
				name        = f"{pathlib.Path(path).stem}.png",
				prefix      = "",
				extension   = ".png",
				denormalize = denormalize or self.denormalize
			)
		
		image = ci.to_image_nparray(
			image       = frame,
			keepdim     = True,
			denormalize = denormalize or self.denormalize,
		)
		# IMPORTANT: Image must be in a BGR format
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		
		self.video_writer.write(image)
		self.index += 1
	
	def write_batch(
		self,
		frames     : list[torch.Tensor | np.ndarray],
		paths      : list[pathlib.Path] = None,
		denormalize: bool 				= False
	):
		"""Write a batch of images to :obj:`dst`.

		Args:
			frames: A :obj:`list` of images.
			paths: A :obj:`list` of image file paths with extensions.
				Default: ``None``.
			denormalize: If ``True``, convert image to ``[0, 255]``.
				Default: ``False``.
		"""
		if paths is None:
			paths = [None for _ in range(len(frames))]
		for image, file in zip(frames, paths):
			self.write(frame=image, path=file, denormalize=denormalize)


class VideoWriterFFmpeg(VideoWriter):
	"""A video writer that writes images to a video file using :obj:`ffmpeg`.

	Args:
		dst: A destination directory to save images.
		image_size: A desired output size of shape `[H, W]`. This is used
			to reshape the input. Default: `[480, 640]`.
		frame_rate: A frame rate of the output video. Default: ``10``.
		pix_fmt: A video codec. Default: ``'yuv420p'``.
		save_image: If ``True`` save each image separately. Default: ``False``.
		denormalize: If ``True``, convert image to ``[0, 255]``.
			Default: ``False``.
		verbose: Verbosity. Default: ``False``.
		kwargs: Any supplied kwargs are passed to :obj:`ffmpeg` verbatim.
	"""
	
	def __init__(
		self,
		dst		   : pathlib.Path,
		image_size : _size_2_t = [480, 640],
		frame_rate : float     = 10,
		pix_fmt    : str       = "yuv420p",
		save_image : bool      = False,
		denormalize: bool      = False,
		verbose    : bool      = False,
		*args, **kwargs
	):
		self.pix_fmt        = pix_fmt
		self.ffmpeg_process = None
		self.ffmpeg_kwargs  = kwargs
		super().__init__(
			dst			= dst,
			image_size  = image_size,
			frame_rate  = frame_rate,
			save_image  = save_image,
			denormalize = denormalize,
			verbose     = verbose,
			*args, **kwargs
		)
	
	def init(self):
		"""Initialize the output destination."""
		if self.dst.is_dir():
			video_file = self.dst / f"result.mp4"
		else:
			video_file = self.dst.parent / f"{self.dst.stem}.mp4"
		video_file.parent.mkdir(parents=True, exist_ok=True)
		
		if self.verbose:
			self.ffmpeg_process = (
				ffmpeg
				.input(
					filename = "pipe:",
					format   = "rawvideo",
					pix_fmt  = "rgb24",
					s        = "{}x{}".format(self.image_size[1], self.image_size[0])
				)
				.output(
					filename = str(video_file),
					pix_fmt  = self.pix_fmt,
					**self.ffmpeg_kwargs
				)
				.overwrite_output()
				.run_async(pipe_stdin=True)
			)
		else:
			self.ffmpeg_process = (
				ffmpeg
				.input(
					filename ="pipe:",
					format   = "rawvideo",
					pix_fmt  = "rgb24",
					s        = "{}x{}".format(self.image_size[1], self.image_size[0])
				)
				.output(
					filename = str(video_file),
					pix_fmt  = self.pix_fmt,
					**self.ffmpeg_kwargs
				)
				.global_args("-loglevel", "quiet")
				.overwrite_output()
				.run_async(pipe_stdin=True)
			)
	
	def close(self):
		"""Stop and release the current :obj:`ffmpeg_process`."""
		if self.ffmpeg_process:
			self.ffmpeg_process.stdin.close()
			self.ffmpeg_process.terminate()
			self.ffmpeg_process.wait()
			self.ffmpeg_process = None
	
	def write(
		self,
		frame      : torch.Tensor | np.ndarray,
		path       : pathlib.Path = None,
		denormalize: bool 		  = False
	):
		"""Write an image to :obj:`dst`.

		Args:
			frame: An image.
			path: An image file path with an extension. Default: ``None``.
			denormalize: If ``True``, convert image to ``[0, 255]``.
				Default: ``False``.
		"""
		if self.save_image:
			assert isinstance(path, pathlib.Path)
			ci.write_image_cv(
				image       = frame,
				dir_path	= self.dst,
				name        = f"{pathlib.Path(path).stem}.png",
				prefix      = "",
				extension   = ".png",
				denormalize = denormalize or self.denormalize
			)
		
		write_video_ffmpeg(
			process     = self.ffmpeg_process,
			image       = frame,
			denormalize = denormalize or self.denormalize
		)
		self.index += 1
	
	def write_batch(
		self,
		frames     : list[torch.Tensor | np.ndarray],
		paths      : list[pathlib.Path] = None,
		denormalize: bool 			    = False,
	):
		"""Write a batch of images to :obj:`dst`.

		Args:
			frames: A :obj:`list` of images.
			paths: A :obj:`list` of image file paths with extensions.
				Default: ``None``.
			denormalize: If ``True``, convert image to ``[0, 255]``.
				Default: ``False``.
		"""
		if paths is None:
			paths = [None for _ in range(len(frames))]
		for image, file in zip(frames, paths):
			self.write(frame=image, path=file, denormalize=denormalize)
			
# endregion
