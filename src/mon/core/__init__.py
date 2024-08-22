#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Core Package.

This package implements the basic functionalities of Python operations. This is
achieved by extending `Python <https://www.python.org/>`__ built-in
functions, including:
	- Data types and structures.
	- File I/O.
	- Filesystem handling.
	- Logging.
	- Managing devices.
	- Parsing.
	- Path handling.
	- etc.

Design Principle:
	- All submodules must be ATOMIC and self-contained.
	- Each submodule should extend a module and keep the same name.
"""

from __future__ import annotations

import mon.core.audio
import mon.core.data
import mon.core.dtype
import mon.core.factory
import mon.core.file
import mon.core.humps
import mon.core.image
import mon.core.logging
import mon.core.pathlib
import mon.core.pointcloud
import mon.core.rich
import mon.core.thop
import mon.core.transform
import mon.core.typing
import mon.core.utils
import mon.core.video
from mon.core.audio import *
from mon.core.data import *
from mon.core.dtype import *
from mon.core.factory import *
from mon.core.file import *
from mon.core.humps import *
from mon.core.image import *
from mon.core.logging import *
from mon.core.pathlib import *
from mon.core.pointcloud import *
from mon.core.rich import (
	console,
	error_console,
	get_download_bar,
	get_progress_bar,
	get_terminal_size,
	print_dict,
	print_table,
	set_terminal_size,
)
from mon.core.thop import *
from mon.core.transform import albumentation
from mon.core.typing import (
	_callable, _ratio_2_t, _ratio_3_t, _ratio_any_t, _scalar_or_tuple_1_t,
	_scalar_or_tuple_2_t, _scalar_or_tuple_3_t, _scalar_or_tuple_4_t,
	_scalar_or_tuple_5_t, _scalar_or_tuple_6_t, _scalar_or_tuple_any_t,
	_size_1_t, _size_2_opt_t, _size_2_t, _size_3_opt_t, _size_3_t, _size_4_t,
	_size_5_t, _size_6_t, _size_any_opt_t, _size_any_t,
)
from mon.core.utils import *
from mon.core.video import *
