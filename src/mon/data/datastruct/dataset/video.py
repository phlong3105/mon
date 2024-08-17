#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements video-only datasets."""

from __future__ import annotations

__all__ = [
	"VideoDataset",
	"VideoLoaderCV",
]

from abc import ABC
from typing import Any

import albumentations as A
import cv2

from mon import core
from mon.data.datastruct import annotation
from mon.data.datastruct.dataset import base
from mon.globals import Split

console             = core.console
ClassLabels         = annotation.ClassLabels
DatapointAttributes = annotation.DatapointAttributes
FrameAnnotation     = annotation.FrameAnnotation
ImageAnnotation     = annotation.ImageAnnotation


# region Video Dataset

class VideoDataset(base.Dataset, ABC):
	"""The base class for all video-based datasets.
	
	Attributes:
		datapoint_attrs: A :class:`dict` of datapoint attributes with the keys
			are the attribute names and the values are the attribute types.
			Must contain: {``'frame'``: :class:`FrameAnnotation`}. Note that to
			comply with :class:`albumentations.Compose`, we will treat the first
			key as the main image attribute.
			
	Args:
		root: A data source. It can be a path to a single video file or a stream.
		split: The data split to use. Default: ``'Split.PREDICT'``.
		classlabels: :class:`ClassLabels` object. Default: ``None``.
		transform: Transformations performed on both the input and target. We
			use `albumentations <https://albumentations.ai/docs/api_reference/full_reference>`__
		to_tensor: If ``True``, convert input and target to :class:`torch.Tensor`.
            Default: ``False``.
        cache_data: If ``True``, cache data to disk for faster loading next
            time. Default: ``False``.
        verbose: Verbosity. Default: ``True``.
	
	See Also: :class:`mon.data.datastruct.dataset.base.Dataset`.
	"""
	
	datapoint_attrs = DatapointAttributes({
		"frame": FrameAnnotation,
	})
	
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
		self.num_frames = 0
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
		
	def __getitem__(self, index: int) -> dict:
		"""Returns a dictionary containing the datapoint and metadata at the
		given :param:`index`.
		"""
		# Get datapoint at the given index
		datapoint = self.get_datapoint(index=index)
		meta      = self.get_meta(index=index)
		# Transform
		if self.transform:
			main_attr      = self.main_attribute
			args           = {k: v for k, v in datapoint.items() if v is not None}
			args["image"]  = args.pop(main_attr)
			transformed    = self.transform(**args)
			transformed[main_attr] = transformed.pop("image")
			datapoint     |= transformed
		if self.to_tensor:
			for k, v in datapoint.items():
				to_tensor_fn = self.datapoint_attrs.get_tensor_fn(k)
				if to_tensor_fn and v is not None:
					datapoint[k] = to_tensor_fn(data=v, keepdim=False, normalize=True)
		# Return
		return datapoint | {"meta": meta}
	
	def __len__(self) -> int:
		"""Return the total number of frames in the video."""
		return self.num_frames
	
	def init_transform(self, transform: A.Compose | Any = None):
		super().init_transform(transform=transform)
		# Add additional targets
		if self.transform:
			additional_targets = self.datapoint_attrs.albumentation_target_types()
			additional_targets.pop(self.main_attribute, None)
			additional_targets.pop("meta", None)
			self.transform.add_targets(additional_targets)
	
	def filter_data(self):
		"""Filter unwanted datapoints."""
		pass
	
	def verify_data(self):
		"""Verify dataset."""
		if self.__len__() <= 0:
			raise RuntimeError(f"No datapoints in the dataset.")
		if self.verbose:
			console.log(f"Number of {self.split_str} datapoints: {self.__len__()}.")
	

class VideoLoaderCV(VideoDataset):
	"""A video loader that retrieves and loads frame(s) from a video or a stream
	using :mod:`cv2`.
	
	See Also: :class:`VideoDataset`.
	"""
	
	def __init__(
		self,
		root       : core.Path,
		split      : Split      	    = Split.PREDICT,
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
	
	@property
	def is_stream(self) -> bool:
		"""Return ``True`` if the input is a video stream."""
		return self.root.is_video_stream() or self.num_frames == -1
	
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
	def frame_height(self) -> int:  # Flag=4
		"""Return the height of the frames in the video stream."""
		return int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	
	@property
	def frame_width(self) -> int:  # Flag=3
		"""Return the width of the frames in the video stream."""
		return int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	
	@property
	def shape(self) -> tuple[int, int, int]:
		"""Return the shape of the frames in the video stream."""
		return self.frame_height, self.frame_width, 3
	
	@property
	def imgsz(self) -> tuple[int, int]:
		"""Return the image size of the frames in the video stream."""
		return self.frame_height, self.frame_width
	
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
	
	def get_data(self):
		root = core.Path(self.root)
		if root.is_video_file():
			self.video_capture = cv2.VideoCapture(str(root), cv2.CAP_FFMPEG)
			num_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
		elif root.is_video_stream():
			self.video_capture = cv2.VideoCapture(str(root), cv2.CAP_FFMPEG)  # stream
			num_frames = -1
		else:
			raise IOError(f"Error when reading input stream or video file!")
		
		if self.num_frames != num_frames:
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
	
	def get_datapoint(self, index: int) -> dict:
		"""Get a datapoint at the given :param:`index`."""
		if not self.is_stream and self.index >= self.num_frames:
			self.close()
			raise StopIteration
		else:
			# Read the next frame
			if isinstance(self.video_capture, cv2.VideoCapture):
				ret_val, frame = self.video_capture.read()
			else:
				raise RuntimeError(f":attr:`video_capture` has not been initialized.")
			if frame:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				frame = FrameAnnotation(index=self.index, frame=frame, path=self.root)
			self.index += 1
			
			# Get data
			datapoint = self.new_datapoint
			for k, v in self.datapoints.items():
				if k == self.main_attribute:
					datapoint[k] = frame.data
				elif v and v[index] and hasattr(v[index], "data"):
					datapoint[k] = v[index].data
			return datapoint
	
	def get_meta(self, index: int = 0) -> dict:
		"""Get metadata at the given :param:`index`. By default, we will use
		the first attribute in :attr:`datapoint_attrs` as the main image attribute.
		"""
		return {
			"format"       : self.format,
			"fourcc"       : self.fourcc,
			"fps"          : self.fps,
			"frame_height" : self.frame_height,
			"frame_width"  : self.frame_width,
			"hash"         : self.root.stat().st_size if isinstance(self.root, core.Path) else None,
			"image_size"   : (self.frame_height, self.frame_width),
			"imgsz" 	   : (self.frame_height, self.frame_width),
			"index"        : index,
			"mode"         : self.mode,
			"name"         : str(self.root.name),
			"num_frames"   : self.num_frames,
			"path"         : self.root,
			"pos_avi_ratio": self.pos_avi_ratio,
			"pos_frames"   : self.pos_frames,
			"pos_msec"     : self.pos_msec,
			"shape" 	   : (self.frame_height, self.frame_width, 3),
			"split"        : self.split_str,
			"stem"         : str(self.root.stem),
		}
	
# endregion
