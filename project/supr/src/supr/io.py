#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements IO functions."""

from __future__ import annotations

__all__ = [
	"ProductCountingWriter",
]

import os
from operator import itemgetter
from timeit import default_timer as timer
from typing import Sequence

import mon
from supr import data
from supr.typing import PathType


# region Writer

class ProductCountingWriter:
	"""Save product counting results.
	
	Args:
		output: A path to the counting results file.
		camera_name: A camera name.
		start_time: The moment when the TexIO is initialized.
		subset: A subset name. One of: ['testA', 'testB'].
		exclude: A list of class ID to exclude from writing. Defaults to [116].
	"""
	
	video_map = {
		"testA": {
			"testA_1": 1,
			"testA_2": 2,
			"testA_3": 3,
			"testA_4": 4,
			"testA_5": 5,
		},
		"testB": {
			"testB_1": 1,
			"testB_2": 2,
			"testB_3": 3,
			"testB_4": 4,
			"testB_5": 5,
		},
	}
	
	def __init__(
		self,
		output 	   : PathType,
		camera_name: str,
		start_time : float         = timer(),
		subset     : str           = "testA",
		exclude    : Sequence[int] = [116]
	):
		super().__init__()
		assert subset in self.video_map
		assert camera_name in self.video_map[subset]
		self.output		 = mon.Path(output)
		self.camera_name = camera_name
		self.video_id 	 = self.video_map[subset][camera_name]
		self.start_time  = start_time
		self.exclude     = exclude
		self.lines 		 = []
		
	def __del__(self):
		""" Close the writer object."""
		pass

	def init_writer(self, output: PathType | None = None):
		"""Initialize the writer object.

		Args:
			output: A path to the counting results file.
		"""
		output = output or self.output
		output = mon.Path(output)
		if output.is_stem():
			output = f"{output}.txt"
		elif output.is_dir():
			output = output / f"{self.camera_name}.txt"
		mon.create_dirs(paths=[str(output.parent)])
		self.output = output
	
	def write(self, products: list[data.Product]):
		"""Write counting results.

		Args:
			products: A list of tracking :class:`data.Product` objects.
		"""
		for p in products:
			class_id = p.majority_label_id
			if class_id in self.exclude:
				continue
			line = f"{self.video_id} {class_id + 1} {int(p.timestamp)}\n"
			self.lines.append(line)
	
	def dump(self):
		"""Dump all content in :attr:`lines` to :attr:`output` file."""
		if not self.output.is_txt_file():
			self.init_writer()
		
		with open(self.output, "w") as f:
			for line in self.lines:
				f.write(line)
	
	@classmethod
	def merge_results(
		cls,
		output_dir : PathType | None = None,
		output_name: str      | None = "track4.txt",
		subset     : str             = "testA"
	):
		"""Merge all cameras' result files into one file.
		
		Args:
			output_dir: A directory to store the :attr:`output_name`.
			output_name: A result file name. Defaults to 'track4.txt'.
			subset: A subset name. One of: ['testA', 'testB'].
		"""
		assert subset in cls.video_map
		
		output_dir 		= output_dir  or mon.Path().absolute()
		output_dir      = mon.Path(output_dir)
		output_name 	= output_name or "track4"
		output_name     = mon.Path(output_name).stem
		output_name 	= output_dir / f"{output_name}.txt"
		compress_writer = open(output_name, "w")
		
		# NOTE: Get results from each file
		for v_name, v_id in cls.video_map[subset].items():
			video_result_file = os.path.join(output_dir, f"{v_name}.txt")
	
			if not os.path.exists(video_result_file):
				mon.console.log(f"Result of {video_result_file} does not exist!")
				continue
	
			# NOTE: Read result
			results = []
			with open(video_result_file) as f:
				line = f.readline()
				while line:
					words  = line.split(" ")
					result = {
						"video_id" : int(words[0]),
						"class_id" : int(words[1]),
						"timestamp": int(words[2]),
					}
					if result["class_id"] != 116:
						results.append(result)
					line = f.readline()
	
			# NOTE: Sort result
			results = sorted(results, key=itemgetter("video_id"))
	
			# NOTE: write result
			for result in results:
				compress_writer.write(f"{result['video_id']} ")
				compress_writer.write(f"{result['class_id']} ")
				compress_writer.write(f"{result['timestamp']} ")
				compress_writer.write("\n")
	
		compress_writer.close()

# endregion
