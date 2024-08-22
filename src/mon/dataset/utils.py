#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dataset Utilities.

This module implements data i/o classes and functions.
"""

from __future__ import annotations

__all__ = [
	"parse_io_worker",
]

from mon import core
from mon.globals import DATA_DIR, DATASETS, Split


# region Parsing

def parse_io_worker(
	src        : core.Path | str,
	dst        : core.Path | str,
	to_tensor  : bool            = False,
	denormalize: bool            = False,
	data_root  : core.Path | str = None,
	verbose    : bool            = False,
) -> tuple[str, core.Dataset, core.VideoWriterCV]:
	"""Parse the :obj:`src` and :obj:`dst` to get the correct I/O worker.
	
	Args:
		src: The source of the input data.
		dst: The destination path.
		to_tensor: If ``True``, convert the image to a tensor. Default: ``False``.
		denormalize: If ``True``, denormalize the image to ``[0, 255]``.
			Default: ``False``.
		data_root: The root directory of all datasets, i.e., :file:`data/`.
		verbose: Verbosity. Default: ``False``.
		
	Return:
		A input loader and an output writer
	"""
	data_name  : str                = ""
	data_loader: core.Dataset       = None
	data_writer: core.VideoWriterCV = None
	
	if isinstance(src, str) and src in DATASETS:
		defaults_dict = dict(
			zip(DATASETS[src].__init__.__code__.co_varnames[1:], DATASETS[src].__init__.__defaults__)
		)
		root = defaults_dict.get("root", DATA_DIR)
		if (
			         root not in [None, "None", ""]
			and data_root not in [None, "None", ""]
			and core.Path(data_root).is_dir()
			and str(root) != str(data_root)
		):
			root = data_root
		config      = {
			"name"     : src,
			"root"     : root,
			"split"	   : Split.TEST,
			"to_tensor": to_tensor,
			"verbose"  : verbose,
		}
		data_name   = src
		data_loader = DATASETS.build(config=config)
	elif core.Path(src).is_dir() and core.Path(src).exists():
		data_name   = core.Path(src).name
		data_loader = core.ImageLoader(
			root      = src,
			to_tensor = to_tensor,
			verbose   = verbose,
		)
	elif core.Path(src).is_video_file():
		data_name   = core.Path(src).name
		data_loader = core.VideoLoaderCV(
			root      = src,
			to_tensor = to_tensor,
			verbose   = verbose,
		)
		data_writer = core.VideoWriterCV(
			dst         = core.Path(dst),
			image_size  = data_loader.imgsz,
			frame_rate  = data_loader.fps,
			fourcc      = "mp4v",
			save_image  = False,
			denormalize = denormalize,
			verbose     = verbose,
		)
	else:
		raise ValueError(f"Invalid input source: {src}")
	return data_name, data_loader, data_writer
	
# endregion
