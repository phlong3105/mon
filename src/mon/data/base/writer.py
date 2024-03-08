#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements data writers."""

from __future__ import annotations

__all__ = [
	"DataWriter",
	"VideoWriter",
	"VideoWriterCV",
	"VideoWriterFFmpeg",
]

from abc import ABC, abstractmethod

import cv2
import ffmpeg
import numpy as np
import torch

from mon import core
from mon.core import _size_2_t


# region Base

class DataWriter(ABC):
	"""The base class for all data writers.

	Args:
		dst: A destination.
		denormalize: If ``True``, convert image to :math:`[0, 255]`. Default: ``False``.
		verbose: Verbosity. Default: ``False``.
	"""
	
	def __init__(
		self,
		dst		   : core.Path,
		denormalize: bool = False,
		verbose    : bool = False,
		*args, **kwargs
	):
		self._dst         = core.Path(dst)
		self._denormalize = denormalize
		self.verbose      = verbose
		self._index       = 0
		self._init()
	
	def __len__(self) -> int:
		"""Return the amount of already written result."""
		return self._index
	
	def __del__(self):
		"""Close."""
		self.close()
	
	@abstractmethod
	def _init(self):
		"""Initialize the data writer."""
		pass
	
	@abstractmethod
	def close(self):
		"""Close."""
		pass
	
	@abstractmethod
	def write(self, data: torch.Tensor | np.ndarray, denormalize: bool = False):
		"""Write data to :attr:`dst`.

		Args:
			data: A data sample.
			denormalize: If ``True``, convert image to :math:`[0, 255]`.
				Default: ``False``.
		"""
		pass
	
	@abstractmethod
	def write_batch(self, data: list[torch.Tensor | np.ndarray], denormalize: bool = False):
		"""Write a batch of images to :attr:`dst`.

		Args:
			data: A :class:`list` of images.
			denormalize: If ``True``, convert image to :math:`[0, 255]`.
				Default: ``False``.
		"""
		pass

# endregion


# region Video

class VideoWriter(DataWriter):
	"""The base class for all video writers.

	Args:
		dst: A directory to save images.
		image_size: A desired output size of shape :math:`[H, W]`. This is used
			to reshape the input. Default: :math:`[480, 640]`.
		frame_rate: A frame rate of the output video. Default: ``10``.
		save_image: If ``True`` save each image separately. Default: ``False``.
		denormalize: If ``True``, convert image to :math:`[0, 255]`.
			Default: ``False``.
		verbose: Verbosity. Default: ``False``.
	"""
	
	def __init__(
		self,
		dst		   : core.Path,
		image_size : _size_2_t = [480, 640],
		frame_rate : float 	   = 10,
		save_image : bool      = False,
		denormalize: bool      = False,
		verbose    : bool      = False,
		*args, **kwargs
	):
		self._image_size = core.parse_hw(size=image_size)
		self._frame_rate = frame_rate
		self._save_image = save_image
		super().__init__(
			dst		    = dst,
			denormalize = denormalize,
			verbose     = verbose,
		)
	
	def __len__(self) -> int:
		"""Return the number frames of already written frames."""
		return self._index
	
	def __del__(self):
		"""Close."""
		self.close()
	
	@abstractmethod
	def _init(self):
		"""Initialize the output handler."""
		pass
	
	@abstractmethod
	def close(self):
		"""Close."""
		pass
	
	@abstractmethod
	def write(
		self,
		data       : torch.Tensor | np.ndarray,
		path       : core.Path 	  | None = None,
		denormalize: bool 				 = False
	):
		"""Write an image to :attr:`dst`.

		Args:
			data: An image.
			path: An image file path with an extension. Default: ``None``.
			denormalize: If ``True``, convert image to :math:`[0, 255]`.
				Default: ``False``.
		"""
		pass
	
	@abstractmethod
	def write_batch(
		self,
		data       : list[torch.Tensor | np.ndarray],
		paths      : list[core.Path]   | None = None,
		denormalize: bool			 		  = False
	):
		"""Write a batch of images to :attr:`dst`.

		Args:
			data: A :class:`list` of images.
			paths: A :class:`list` of image file paths with extensions.
				Default: ``None``.
			denormalize: If ``True``, convert image to :math:`[0, 255]`.
				Default: ``False``.
		"""
		pass


class VideoWriterCV(VideoWriter):
	"""A video writer that writes images to a video file using :mod:`cv2`.

	Args:
		dst: A destination directory to save images.
		image_size: A desired output size of shape :math:`[H, W]`. This is used
			to reshape the input. Default: :math:`[480, 640]`.
		frame_rate: A frame rate of the output video. Default: ``10``.
		fourcc: Video codec. One of ``'mp4v'``, ``'xvid'``, ``'mjpg'``, or
		``'wmv'``. Default: ``'.mp4v'``.
		save_image: If ``True``, save each image separately. Default: ``False``.
		denormalize: If ``True``, convert image to :math:`[0, 255]`.
			Default: ``False``.
		verbose: Verbosity. Default: ``False``.
	"""
	
	def __init__(
		self,
		dst		   : core.Path,
		image_size : _size_2_t = [480, 640],
		frame_rate : float     = 30,
		fourcc     : str       = "mp4v",
		save_image : bool      = False,
		denormalize: bool      = False,
		verbose    : bool      = False,
		*args, **kwargs
	):
		self._fourcc       = fourcc
		self._video_writer = None
		super().__init__(
			dst			= dst,
			image_size  = image_size,
			frame_rate  = frame_rate,
			save_image  = save_image,
			denormalize = denormalize,
			verbose     = verbose,
			*args, **kwargs
		)
	
	def _init(self):
		"""Initialize the output destination."""
		if self._dst.is_dir():
			video_file = self._dst / f"result.mp4"
		else:
			video_file = self._dst.parent / f"{self._dst.stem}.mp4"
		video_file.parent.mkdir(parents=True, exist_ok=True)
		
		fourcc = cv2.VideoWriter_fourcc(*self._fourcc)
		self._video_writer = cv2.VideoWriter(
			filename  = str(video_file),
			fourcc    = fourcc,
			fps       = float(self._frame_rate),
			frameSize = self._image_size[::-1],  # Must be in [W, H]
			isColor   = True
		)
		
		if self._video_writer is None:
			raise FileNotFoundError(f"Cannot create video file at {video_file}.")
	
	def close(self):
		"""Close the :attr:`video_writer`."""
		if self._video_writer:
			self._video_writer.release()
	
	def write(
		self,
		data      : torch.Tensor | np.ndarray,
		path       : core.Path    | None = None,
		denormalize: bool 				 = False
	):
		"""Write an image to :attr:`dst`.

		Args:
			data: An image.
			path: An image file path with an extension. Default: ``None``.
			denormalize: If ``True``, convert image to :math:`[0, 255]`.
				Default: ``False``.
		"""
		if self._save_image:
			core.write_image_cv(
				image       = data,
				dir_path    = self._dst,
				name        = f"{core.Path(path).stem}.png",
				prefix      = "",
				extension   = ".png",
				denormalize = denormalize or self._denormalize
			)
		
		image = core.to_image_nparray(
			input       = data,
			keepdim     = True,
			denormalize = denormalize or self._denormalize,
		)
		# IMPORTANT: Image must be in a BGR format
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		
		self._video_writer.write(image)
		self._index += 1
	
	def write_batch(
		self,
		data       : list[torch.Tensor | np.ndarray],
		paths      : list[core.Path]   | None = None,
		denormalize: bool 					  = False
	):
		"""Write a batch of images to :attr:`dst`.

		Args:
			data: A :class:`list` of images.
			paths: A :class:`list` of image file paths with extensions.
				Default: ``None``.
			denormalize: If ``True``, convert image to :math:`[0, 255]`.
				Default: ``False``.
		"""
		if paths is None:
			paths = [None for _ in range(len(data))]
		for image, file in zip(data, paths):
			self.write(data=image, path=file, denormalize=denormalize)


class VideoWriterFFmpeg(VideoWriter):
	"""A video writer that writes images to a video file using :mod:`ffmpeg`.

	Args:
		dst: A destination directory to save images.
		image_size: A desired output size of shape :math:`[H, W]`. This is used
			to reshape the input. Default: :math:`[480, 640]`.
		frame_rate: A frame rate of the output video. Default: ``10``.
		pix_fmt: A video codec. Default: ``'yuv420p'``.
		save_image: If ``True`` save each image separately. Default: ``False``.
		denormalize: If ``True``, convert image to :math:`[0, 255]`.
			Default: ``False``.
		verbose: Verbosity. Default: ``False``.
		kwargs: Any supplied kwargs are passed to :mod:`ffmpeg` verbatim.
	"""
	
	def __init__(
		self,
		dst		   : core.Path,
		image_size : _size_2_t = [480, 640],
		frame_rate : float     = 10,
		pix_fmt    : str       = "yuv420p",
		save_image : bool      = False,
		denormalize: bool      = False,
		verbose    : bool      = False,
		*args, **kwargs
	):
		self._pix_fmt        = pix_fmt
		self._ffmpeg_process = None
		self._ffmpeg_kwargs  = kwargs
		super().__init__(
			dst			= dst,
			image_size  = image_size,
			frame_rate  = frame_rate,
			save_image  = save_image,
			denormalize = denormalize,
			verbose     = verbose,
			*args, **kwargs
		)
	
	def _init(self):
		"""Initialize the output destination."""
		if self._dst.is_dir():
			video_file = self._dst / f"result.mp4"
		else:
			video_file = self._dst.parent / f"{self._dst.stem}.mp4"
		video_file.parent.mkdir(parents=True, exist_ok=True)
		
		if self.verbose:
			self._ffmpeg_process = (
				ffmpeg
				.input(
					filename = "pipe:",
					format   = "rawvideo",
					pix_fmt  = "rgb24",
					s        = "{}x{}".format(self._image_size[1], self._image_size[0])
				)
				.output(
					filename = str(video_file),
					pix_fmt  = self._pix_fmt,
					**self._ffmpeg_kwargs
				)
				.overwrite_output()
				.run_async(pipe_stdin=True)
			)
		else:
			self._ffmpeg_process = (
				ffmpeg
				.input(
					filename ="pipe:",
					format   = "rawvideo",
					pix_fmt  = "rgb24",
					s        = "{}x{}".format(self._image_size[1], self._image_size[0])
				)
				.output(
					filename = str(video_file),
					pix_fmt  = self._pix_fmt,
					**self._ffmpeg_kwargs
				)
				.global_args("-loglevel", "quiet")
				.overwrite_output()
				.run_async(pipe_stdin=True)
			)
	
	def close(self):
		"""Stop and release the current :attr:`ffmpeg_process`."""
		if self._ffmpeg_process:
			self._ffmpeg_process.stdin.close()
			self._ffmpeg_process.terminate()
			self._ffmpeg_process.wait()
			self._ffmpeg_process = None
	
	def write(
		self,
		data       : torch.Tensor | np.ndarray,
		path       : core.Path    | None = None,
		denormalize: bool 				 = False
	):
		"""Write an image to :attr:`dst`.

		Args:
			data: An image.
			path: An image file path with an extension. Default: ``None``.
			denormalize: If ``True``, convert image to :math:`[0, 255]`.
				Default: ``False``.
		"""
		if self._save_image:
			assert isinstance(path, core.Path)
			core.write_image_cv(
				image       = data,
				dir_path	= self._dst,
				name        = f"{core.Path(path).stem}.png",
				prefix      = "",
				extension   =".png",
				denormalize = denormalize or self._denormalize
			)
		
		core.write_video_ffmpeg(
			process     = self._ffmpeg_process,
			image       = data,
			denormalize = denormalize or self._denormalize
		)
		self._index += 1
	
	def write_batch(
		self,
		data       : list[torch.Tensor | np.ndarray],
		paths      : list[core.Path]   | None = None,
		denormalize: bool 					  = False,
	):
		"""Write a batch of images to :attr:`dst`.

		Args:
			data: A :class:`list` of images.
			paths: A :class:`list` of image file paths with extensions.
				Default: ``None``.
			denormalize: If ``True``, convert image to :math:`[0, 255]`.
				Default: ``False``.
		"""
		if paths is None:
			paths = [None for _ in range(len(data))]
		for image, file in zip(data, paths):
			self.write(data=image, path=file, denormalize=denormalize)
			
# endregion
