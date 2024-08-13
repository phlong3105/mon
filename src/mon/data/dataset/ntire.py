#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements datasets and datamodules for
`NTIRE challenge <https://cvlai.net/ntire/2024/>`__.
"""

from __future__ import annotations

__all__ = [
	"NTIRE24LLIE",
	"NTIRE24LLIEDataModule",
]

from typing import Literal

from mon import core
from mon.data.datastruct import annotation, datamodule, dataset
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "ntire"
DatapointAttributes = annotation.DatapointAttributes
ImageAnnotation     = annotation.ImageAnnotation
ImageDataset        = dataset.ImageDataset


# region Dataset

@DATASETS.register(name="ntire24_llie")
class NTIRE24LLIE(ImageDataset):
	"""NTIRE24-LLIE dataset consists of 300 low-light and normal-light image
	pairs. They are divided into 230 training pairs and 35 validation pairs,
	and 35 testing pairs.
	
	See Also: :class:`mon.data.datastruct.dataset.image.ImageDataset`.
	"""
	
	tasks : list[Task]  = [Task.LLIE]
	splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
	datapoint_attrs     = DatapointAttributes({
		"lq_image": ImageAnnotation,
		"hq_image": ImageAnnotation,
	})
	has_test_annotations: bool = False
	
	def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
		super().__init__(root=root, *args, **kwargs)
	
	def get_data(self):
		# patterns = [
		# 	self.root / "ntire24-llie" / self.split / "low",
		# ]
		if self.split in [Split.TRAIN]:
			patterns = [
				self.root / "train" / "ntire24_llie" / "lq",
			]
		elif self.split in [Split.VAL]:
			patterns = [
				self.root / "train" / "ntire24_llie" / "lq",
			]
		elif self.split in [Split.TEST]:
			patterns = [
				self.root / "val"   / "ntire24_llie" / "lq",
			]
		else:
			raise ValueError
		
		# LQ images
		lq_images: list[ImageAnnotation] = []
		with core.get_progress_bar(disable=self.disable_pbar) as pbar:
			for pattern in patterns:
				for path in pbar.track(
					sorted(list(pattern.rglob("*"))),
					description=f"Listing {self.__class__.__name__} {self.split_str} images"
				):
					if path.is_image_file():
						lq_images.append(ImageAnnotation(path=path))
		
		# HQ images
		hq_images: list[ImageAnnotation] = []
		with core.get_progress_bar(disable=self.disable_pbar) as pbar:
			for img in pbar.track(
				lq_images,
				description=f"Listing {self.__class__.__name__} {self.split_str} ground-truths"
			):
				path = img.path.replace("/lq/", "/hq/")
				hq_images.append(ImageAnnotation(path=path.image_file()))
		
		self.datapoints["lq_image"] = lq_images
		self.datapoints["hq_image"] = hq_images
		
# endregion


# region DataModule

@DATAMODULES.register(name="ntire24_llie")
class NTIRE24LLIEDataModule(datamodule.DataModule):
	"""NTIRE24-LLIE datamodule used in NTIRE 2024 Challenge
	`<https://cvlai.net/ntire/2024/>`__
	
	See Also: :class:`mon.data.datastruct.datamodule.DataModule`.
	"""
	
	tasks: list[Task] = [Task.LLIE]
	
	def prepare_data(self, *args, **kwargs):
		if self.classlabels is None:
			self.get_classlabels()
	
	def setup(self, stage: Literal["train", "test", "predict", None] = None):
		if self.can_log:
			console.log(f"Setup [red]{self.__class__.__name__}[/red].")
		
		if stage in [None, "train"]:
			self.train = NTIRE24LLIE(split=Split.TRAIN, **self.dataset_kwargs)
			self.val   = NTIRE24LLIE(split=Split.VAL,   **self.dataset_kwargs)
		if stage in [None, "test"]:
			self.test  = NTIRE24LLIE(split=Split.TEST,  **self.dataset_kwargs)
		
		if self.classlabels is None:
			self.get_classlabels()
		
		if self.can_log:
			self.summarize()
	
	def get_classlabels(self):
		pass
	
# endregion
