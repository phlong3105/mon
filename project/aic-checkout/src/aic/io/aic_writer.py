#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""IO classes especially for AI City Challenge tasks.
"""

from __future__ import annotations

import os
from operator import itemgetter
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional

from aic.objects import Product
from onevision import console
from onevision import create_dirs
from onevision import is_stem

__all__ = [
	"AIC22RetailCheckoutWriter",
]


# MARK: - AIC22RetailCheckoutWriter

class AIC22RetailCheckoutWriter:
	"""AIC Retail Checkout Writer periodically saves the measurement results in ONE
	camera.
	
	Attributes:
		dst (str):
			Path to the file to save the counting results.
		camera_name (str):
			Camera name.
		video_id (int):
			Numeric identifier of input camera stream.
		start_time (float):
			Moment when the TexIO is initialized.
		subset (str):
            Subset name. One of: [`dataset_a`, `dataset_b`].
		writer (io stream):
			File writer to export the counting results.
	"""
	
	video_map_a = {
		"testA_1": 1,
		"testA_2": 2,
		"testA_3": 3,
		"testA_4": 4,
		"testA_5": 5,
	}
	
	video_map_b = {
		"testB_1": 1,
		"testB_2": 2,
		"testB_3": 3,
		"testB_4": 4,
		"testB_5": 5,
	}
	
	# MARK: Magic Function

	def __init__(
		self,
		dst 	   : str,
		camera_name: str,
		start_time : float = timer(),
		subset     : str = "test_a",
		*args, **kwargs
	):
		super().__init__()
		if subset not in ["test_a", "test_b"]:
			raise ValueError(f"`subset` must be one of ['test_a', 'test_b']. But got: {subset}.")
		if subset == "test_a":
			video_map = self.video_map_a
		else:
			video_map = self.video_map_b
		if camera_name not in video_map:
			raise ValueError(
				f"The given `camera_name` has not been defined in AIC camera "
			    f"list. Please check again!"
			)

		self.dst		 = dst
		self.camera_name = camera_name
		self.video_id 	 = video_map[camera_name]
		self.start_time  = start_time
		self.lines 		 = []
		
	def __del__(self):
		""" Close the writer object."""
		pass

	# MARK: Configure

	def init_writer(self, dst: str):
		"""Initialize writer object.

		Args:
			dst (str):
				Path to the file to save the counting results.
		"""
		if is_stem(dst):
			dst = f"{dst}.txt"
		elif os.path.isdir(dst):
			dst = os.path.join(dst, f"{self.camera_name}.txt")
		parent_dir = str(Path(dst).parent)
		create_dirs(paths=[parent_dir])

	# MARK: Write
	
	def write(self, moving_objects: list[Product]):
		"""Write counting result from a list of tracked moving objects.

		Args:
			moving_objects (list):
				List of tracked moving objects.
		"""
		for obj in moving_objects:
			class_id = obj.label_id_by_majority
			if class_id != 116:
				line = f"{self.video_id} {class_id + 1} {int(obj.timestamp)}\n"
				self.lines.append(line)
	
	def dump(self):
		dst = self.dst
		if is_stem(dst):
			dst = f"{dst}.txt"
		elif os.path.isdir(dst):
			dst = os.path.join(dst, f"{self.camera_name}.txt")
		parent_dir = str(Path(dst).parent)
		create_dirs(paths=[parent_dir])
		
		with open(dst, "w") as f:
			for line in self.lines:
				f.write(line)
	
	@classmethod
	def compress_all_results(
		cls,
		output_dir : Optional[str] = None,
		output_name: Optional[str] = None,
		subset     : str           = "test_a"
	):
		"""Compress all result of video into one file
		
		Args:
			output_dir (str):
				Directory of output track1.txt will be written
			output_name (str):
				Final compress result name. Example "track1.txt".
			subset (str):
				Subset name. One of: [`test_a`, `test_b`].
		"""
		if subset not in ["test_a", "test_b"]:
			raise ValueError(f"`subset` must be one of ['test_a', 'test_b']. "
			                 f"But got: {subset}.")
		if subset == "test_a":
			video_map = cls.video_map_a
		else:
			video_map = cls.video_map_b
		
		output_dir 		= output_dir  if (output_dir is not None)  else ""
		output_name 	= output_name if (output_name is not None) else "track4"
		output_name 	= os.path.join(output_dir, f"{output_name}.txt")
		compress_writer = open(output_name, "w")
		
		# NOTE: Get result from each file
		for video_name, video_id in video_map.items():
			video_result_file = os.path.join(output_dir, f"{video_name}.txt")
	
			if not os.path.exists(video_result_file):
				console.log(f"Result of {video_result_file} is not exist")
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
