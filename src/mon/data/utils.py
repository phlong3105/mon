#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from __future__ import annotations

__all__ = [
	"parse_io_worker",
]

from mon import core
from mon.data import base
from mon.globals import DATA_DIR, DATASETS, Split


# region Utils

def parse_io_worker(
	src        : core.Path | str,
	dst        : core.Path | str,
	denormalize: bool = False,
	data_root  : core.Path | str = None
) -> tuple[str, base.Dataset, base.DataWriter]:
	"""Parse the :param:`src` and :param:`dst` to get the correct I/O worker.
	
	Args:
		src: The source of the input data.
		dst: The destination path.
		denormalize: If ``True``, denormalize the image to :math:`[0, 255]`.
			Default: ``False``.
		data_root: The root directory of all datasets, i.e., :file:`data/`.
		
	Return:
		A input loader and an output writer
	"""
	data_name  : str             = ""
	data_loader: base.Dataset 	 = None
	data_writer: base.DataWriter = None
	
	if isinstance(src, str) and src in DATASETS:
		dataset_defaults = DATASETS[src].__init__.__defaults__
		root = dataset_defaults.get("root", DATA_DIR)
		if (
			root      not in [None, "None", ""] and
			data_root not in [None, "None", ""] and
			core.Path(data_root).is_dir() and
			str(root) != str(data_root)
		):
			root = data_root
		
		config = {
			"name"     : src,
			"root"     : root,
			"split"	   : Split.TEST,
			"to_tensor": True,
			"verbose"  : False,
		}
		data_name   = src
		data_loader = DATASETS.build(config=config)
	elif core.Path(src).is_dir() and core.Path(src).exists():
		data_name   = core.Path(src).name
		data_loader = base.ImageLoader(root=src, to_tensor=True, verbose=False)
	elif core.Path(src).is_video_file():
		data_name   = core.Path(src).name
		data_loader = base.VideoLoaderCV(root=src, to_tensor=True, verbose=False)
		data_writer = base.VideoWriterCV(
			dst         = core.Path(dst),
			image_size  = data_loader.imgsz,
			frame_rate  = data_loader.fps,
			fourcc      = "mp4v",
			save_image  = False,
			denormalize = denormalize,
			verbose     = False,
		)
	else:
		raise ValueError(f"Invalid input source: {src}")
	return data_name, data_loader, data_writer
	
# endregion
