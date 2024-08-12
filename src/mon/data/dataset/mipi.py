#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements datasets and datamodules for
`MIPI challenge <https://mipi-challenge.org/MIPI2024/index.html>`__.
"""

from __future__ import annotations

__all__ = [
	"MIPI24Flare",
	"MIPI24FlareDataModule",
]

from typing import Literal

from mon import core
from mon.data.datastruct import annotation as anno, datamodule, dataset
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console          = core.console
default_root_dir = DATA_DIR / "mipi"


# region Dataset

@DATASETS.register(name="mipi24_flare")
class MIPI24Flare(dataset.ImageEnhancementDataset):
	"""Nighttime Flare Removal dataset used in MIPI 2024 Challenge
	`<https://mipi-challenge.org/MIPI2024/index.html>`__
	
	See Also: :class:`base.ImageEnhancementDataset`.
	"""
	
	tasks  = [Task.LES]
	splits = [Split.TRAIN, Split.VAL, Split.TEST]
	has_test_annotations = False
	
	def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
		super().__init__(root=root, *args, **kwargs)
	
	def get_images(self):
		if self.split in [Split.TRAIN]:
			patterns = [
				self.root / "train" / "mipi24_flare" / "lq",
			]
		elif self.split in [Split.VAL]:
			patterns = [
				self.root / "val" / "mipi24_flare" / "lq",
			]
		elif self.split in [Split.TEST]:
			patterns = [
				self.root / "test" / "mipi24_flare" / "lq",
			]
		else:
			raise ValueError
		self.images: list[anno.ImageAnnotation] = []
		with core.get_progress_bar(disable=self.disable_pbar) as pbar:
			for pattern in patterns:
				for path in pbar.track(
					sorted(list(pattern.rglob("*"))),
					description=f"Listing {self.__class__.__name__} {self.split_str} images"
				):
					if path.is_image_file():
						image = anno.ImageAnnotation(path=path)
						self.images.append(image)
	
	def get_annotations(self):
		self.annotations: list[anno.ImageAnnotation] = []
		with core.get_progress_bar(disable=self.disable_pbar) as pbar:
			for img in pbar.track(
				self.images,
				description=f"Listing {self.__class__.__name__} {self.split_str} labels"
			):
				path = img.path.replace("/lq/", "/hq/")
				ann  = anno.ImageAnnotation(path=path.image_file())
				self.annotations.append(ann)


# region DataModule

@DATAMODULES.register(name="mipi24_flare")
class MIPI24FlareDataModule(datamodule.DataModule):
	"""Nighttime Flare Removal datamodule used in MIPI 2024 Challenge
	`<https://mipi-challenge.org/MIPI2024/index.html>`__
	
	See Also: :class:`base.DataModule`.
	"""
	
	tasks = [Task.LES]
	
	def prepare_data(self, *args, **kwargs):
		if self.classlabels is None:
			self.get_classlabels()
	
	def setup(self, stage: Literal["train", "test", "predict", None] = None):
		if self.can_log:
			console.log(f"Setup [red]{self.__class__.__name__}[/red].")
		
		if stage in [None, "train"]:
			self.train = MIPI24Flare(split=Split.TRAIN, **self.dataset_kwargs)
			self.val   = MIPI24Flare(split=Split.VAL, **self.dataset_kwargs)
		if stage in [None, "test"]:
			self.test  = MIPI24Flare(split=Split.VAL, **self.dataset_kwargs)
		
		if self.classlabels is None:
			self.get_classlabels()
		
		if self.can_log:
			self.summarize()
	
	def get_classlabels(self):
		pass
	
# endregion
