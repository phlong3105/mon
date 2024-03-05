#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from __future__ import annotations

__all__ = [
	"parse_io_worker",
]

from typing import Any

from mon import core
from mon.data import base
from mon.globals import DATASETS, Split


# region Utils

def parse_io_worker(
	src        : Any,
	dst        : core.Path | str,
	denormalize: bool = False,
) -> tuple[str, base.Dataset, base.DataWriter]:
	"""Parse the :param:`src` and :param:`dst` to get the correct I/O worker.
	
	Args:
		src: The source of the input data.
		dst: The destination path.
		denormalize: If ``True``, denormalize the image to :math:`[0, 255]`.
			Default: ``False``.
		
	Return:
		A input loader and an output writer
	"""
	data_name  : str             = ""
	data_loader: base.Dataset 	 = None
	data_writer: base.DataWriter = None
	
	if isinstance(src, str) and src in DATASETS:
		config = {
			"name"     : src,
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
