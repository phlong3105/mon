#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements video-only datasets."""

from __future__ import annotations

__all__ = [
	"UnlabeledVideoDataset",
	"VideoLoaderCV",
	"VideoLoaderFFmpeg",
]

import subprocess
from abc import ABC, abstractmethod

import albumentations as A
import cv2
import ffmpeg
import numpy as np
import torch

from mon import core
from mon.data.datastruct import annotation as anno
from mon.data.datastruct.dataset import base
from mon.globals import Split

console     = core.console
ClassLabels = anno.ClassLabels
FrameLabel  = anno.FrameAnnotation


# region Unlabeled Video Dataset

class UnlabeledVideoDataset(base.UnlabeledDataset, ABC):
	"""The base class for datasets that represent an unlabeled video. This is
	mainly used for unsupervised learning tasks.
	
	Args:
		root: A data source. It can be a path to a single video file or a
			stream.
		split: The data split to use. One of: [``'train'``, ``'val'``,
			``'test'``, ``'predict'``]. Default: ``'train'``.
		classlabels: :class:`ClassLabels` object. Default: ``None``.
		transform: Transformations performed on both the input and target. We
			use `albumentations <https://albumentations.ai/docs/api_reference/full_reference>`__
		to_tensor: If True, convert input and target to :class:`torch.Tensor`.
			Default: ``False``.
		cache_data: If ``True``, cache data to disk for faster loading next
			time. Default: ``False``.
		verbose: Verbosity. Default: ``True``.
	
	See Also: :class:`UnlabeledDataset`.
	"""
	
	def __init__(
		self,
		root       : core.Path,
		split      : Split          	= Split.PREDICT,
		classlabels: ClassLabels | None = None,
		transform  : A.Compose   | None = None,
		to_tensor  : bool               = False,
		cache_data : bool               = False,
		verbose    : bool               = True,
		*args, **kwargs
	):
		super().__init__(
			root        = root,
			split       = split,
			classlabels = classlabels,
			transform   = transform,
			to_tensor   = to_tensor,
			verbose     = verbose,
			*args, **kwargs
		)
		self.num_frames = 0
		self.init_video()
	
	def __len__(self) -> int:
		return self.num_frames
	
	@abstractmethod
	def __getitem__(self, item: int) -> tuple[
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray | None,
		dict | None
	]:
		pass
	
	@property
	@abstractmethod
	def fourcc(self) -> str:
		"""Return the 4-character code of codec."""
		pass
	
	@property
	@abstractmethod
	def fps(self) -> int:
		"""Return the frame rate."""
		pass
	
	@property
	@abstractmethod
	def frame_height(self) -> int:
		"""Return the height of the frames in the video stream."""
		pass
	
	@property
	@abstractmethod
	def frame_width(self) -> int:
		"""Return the width of the frames in the video stream."""
		pass
	
	@property
	def is_stream(self) -> bool:
		"""Return ``True`` if it is a video stream, i.e., unknown :attr:`frame_count`. """
		return self.num_frames == -1
	
	@property
	def shape(self) -> list[int]:
		"""Return the shape of the frames in the video stream in
		:math:`[H, W, C]` format.
		"""
		return [self.frame_height, self.frame_width, 3]
	
	@property
	def image_size(self) -> list[int]:
		"""Return the shape of the frames in the video stream in
		:math:`[H, W]` format.
		"""
		return [self.frame_height, self.frame_width]
	
	@property
	def imgsz(self) -> list[int]:
		"""Return the shape of the frames in the video stream in
		:math:`[H, W]` format.
		"""
		return self.image_size
	
	@abstractmethod
	def init_video(self):
		"""Initialize the video capture object."""
		pass
	
	@abstractmethod
	def reset(self):
		"""Reset and start over."""
		pass
	
	@abstractmethod
	def close(self):
		"""Stop and release."""
		pass


class VideoLoaderCV(UnlabeledVideoDataset):
	"""A video loader that retrieves and loads frame(s) from a video or a stream
	using :mod:`cv2`.
	
	See Also: :class:`UnlabeledVideoDataset`.
	"""
	
	def __init__(
		self,
		root       : core.Path,
		split      : Split      	    = Split.TRAIN,
		classlabels: ClassLabels | None = None,
		transform  : A.Compose   | None = None,
		to_tensor  : bool               = False,
		cache_data : bool               = False,
		verbose    : bool               = True,
		*args, **kwargs
	):
		self.video_capture = None
		super().__init__(
			root        = root,
			split       = split,
			classlabels = classlabels,
			transform   = transform,
			to_tensor   = to_tensor,
			cache_data  = cache_data,
			verbose     = verbose,
			*args, **kwargs
		)
	
	def __getitem__(self, item: int) -> tuple[
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray | None,
		dict | None
	]:
		pass
	
	def __next__(self) -> tuple[
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray | None,
		dict | None
	]:
		if not self.is_stream and self.index >= self.num_frames:
			self.close()
			raise StopIteration
		else:
			# Read the next frame
			if isinstance(self.video_capture, cv2.VideoCapture):
				ret_val, frame = self.video_capture.read()
			else:
				raise RuntimeError(f":attr`_video_capture` has not been initialized.")
			if frame is not None:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				frame = FrameLabel(index=self.index, path=self.root, frame=frame)
			self.index += 1
			
			# Get data
			image = frame.data
			meta  = frame.meta
			if self.transform is not None:
				transformed = self.transform(image=image)
				image	    = transformed["image"]
			if self.to_tensor:
				image = core.to_image_tensor(input=image, keepdim=False, normalize=True)
			return image, None, meta
			
	@property
	def format(self):  # Flag=8
		"""Return the format of the Mat objects (see Mat::type()) returned by
		VideoCapture::retrieve(). Set value -1 to fetch undecoded RAW video
		streams (as Mat 8UC1).
		"""
		return self.video_capture.get(cv2.CAP_PROP_FORMAT)
	
	@property
	def fourcc(self) -> str:  # Flag=6
		"""Return the 4-character code of codec."""
		return str(self.video_capture.get(cv2.CAP_PROP_FOURCC))
	
	@property
	def fps(self) -> int:  # Flag=5
		"""Return the frame rate."""
		return int(self.video_capture.get(cv2.CAP_PROP_FPS))
	
	@property
	def frame_count(self) -> int:  # Flag=7
		"""Return the number of frames in the video file."""
		if isinstance(self.video_capture, cv2.VideoCapture):
			return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
		elif isinstance(self.video_capture, list):
			return len(self.video_capture)
		else:
			return -1
	
	@property
	def frame_height(self) -> int:  # Flag=4
		"""Return the height of the frames in the video stream."""
		return int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	
	@property
	def frame_width(self) -> int:  # Flag=3
		"""Return the width of the frames in the video stream."""
		return int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	
	@property
	def mode(self):  # Flag=10
		"""Return the backend-specific value indicating the current capture mode."""
		return self.video_capture.get(cv2.CAP_PROP_MODE)
	
	@property
	def pos_avi_ratio(self) -> int:  # Flag=2
		"""Return the relative position of the video file: ``0``=start of the
		film, ``1``=end of the film.
		"""
		return int(self.video_capture.get(cv2.CAP_PROP_POS_AVI_RATIO))
	
	@property
	def pos_msec(self) -> int:  # Flag=0
		"""Return the current position of the video file in milliseconds."""
		return int(self.video_capture.get(cv2.CAP_PROP_POS_MSEC))
	
	@property
	def pos_frames(self) -> int:  # Flag=1
		"""Return the 0-based index of the frame to be decoded/captured next."""
		return int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
	
	def init_video(self):
		root = core.Path(self.root)
		if root.is_video_file():
			self.video_capture = cv2.VideoCapture(str(root), cv2.CAP_FFMPEG)
			num_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
		elif root.is_video_stream():
			self.video_capture = cv2.VideoCapture(str(root), cv2.CAP_FFMPEG)  # stream
			num_frames = -1
		else:
			raise IOError(f"Error when reading input stream or video file!")
		
		if self.num_frames == 0:
			self.num_frames = num_frames
		
	def reset(self):
		"""Reset and start over."""
		self.index = 0
		if isinstance(self.video_capture, cv2.VideoCapture):
			self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.index)
	
	def close(self):
		"""Stop and release the current attr:`_video_capture` object."""
		if isinstance(self.video_capture, cv2.VideoCapture):
			self.video_capture.release()


class VideoLoaderFFmpeg(UnlabeledVideoDataset):
	"""A video loader that retrieves and loads frame(s) from a video or a stream
	using :mod:`ffmpeg`.
	
	References:
		`<https://github.com/kkroening/ffmpeg-python/tree/master/examples>`__
	
	See Also: :class:`UnlabeledVideoDataset`.
	"""
	
	def __init__(
		self,
		root       : core.Path,
		split      : Split          	= Split.TRAIN,
		classlabels: ClassLabels | None = None,
		transform  : A.Compose   | None = None,
		to_tensor  : bool               = False,
		cache_data : bool               = False,
		verbose    : bool               = True,
		*args, **kwargs
	):
		self.ffmpeg_cmd     = None
		self.ffmpeg_process = None
		self.ffmpeg_kwargs  = kwargs
		self.video_info     = None
		super().__init__(
			root        = root,
			split       = split,
			classlabels = classlabels,
			transform   = transform,
			to_tensor   = to_tensor,
			cache_data  = cache_data,
			verbose     = verbose,
			*args, **kwargs
		)
	
	def __getitem__(self, item: int) -> tuple[
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray | None,
		dict | None
	]:
		pass
	
	def __next__(self) -> tuple[
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray | None,
		dict | None
	]:
		if not self.is_stream and self.index >= self.frame_count:
			self.close()
			raise StopIteration
		else:
			# Read the next frame
			if self.ffmpeg_process:
				frame = core.read_video_ffmpeg(
					process = self.ffmpeg_process,
					width   = self.frame_width,
					height  = self.frame_height
				)  # Already in RGB
			else:
				raise RuntimeError(f":attr`_video_capture` has not been initialized.")
			if frame is not None:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				frame = FrameLabel(index=self.index, path=self.root, frame=frame)
			self.index += 1
			
			# Get data
			image = frame.data
			meta  = frame.meta
			if self.transform is not None:
				transformed = self.transform(image=image)
				image	    = transformed["image"]
			if self.to_tensor:
				image = core.to_image_tensor(input=image, keepdim=False, normalize=True)
			
			return image, None, meta
	
	@property
	def fourcc(self) -> str:
		"""Return the 4-character code of codec."""
		return self.video_info["codec_name"]
	
	@property
	def fps(self) -> int:
		"""Return the frame rate."""
		return int(self.video_info["avg_frame_rate"].split("/")[0])
	
	@property
	def frame_count(self) -> int:
		"""Return the number of frames in the video file."""
		if self.root.is_video_file():
			return int(self.video_info["nb_frames"])
		else:
			return -1
	
	@property
	def frame_width(self) -> int:
		"""Return the width of the frames in the video stream."""
		return int(self.video_info["width"])
	
	@property
	def frame_height(self) -> int:
		"""Return the height of the frames in the video stream."""
		return int(self.video_info["height"])
	
	def init_video(self):
		"""Initialize ``ffmpeg`` cmd."""
		source = str(self.root)
		probe  = ffmpeg.probe(source, **self.ffmpeg_kwargs)
		self.video_info = next(
			s for s in probe["streams"] if s["codec_type"] == "video"
		)
		if self.verbose:
			self.ffmpeg_cmd = (
				ffmpeg
				.input(source, **self.ffmpeg_kwargs)
				.output("pipe:", format="rawvideo", pix_fmt="rgb24")
				.compile()
			)
		else:
			self.ffmpeg_cmd = (
				ffmpeg
				.input(source, **self.ffmpeg_kwargs)
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
				bufsize = 10 ** 8
			)
	
	def close(self):
		"""Stop and release the current :attr:`ffmpeg_process`."""
		if self.ffmpeg_process and self.ffmpeg_process.poll() is not None:
			# os.killpg(os.getpgid(self.ffmpeg_process.pid), signal.SIGTERM)
			self.ffmpeg_process.terminate()
			self.ffmpeg_process.wait()
			self.ffmpeg_process = None

# endregion
