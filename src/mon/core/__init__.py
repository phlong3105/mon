#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The :mod:`core` package implements basic functionality Python operations.

This is achieved by extending `Python <https://www.python.org/>`__ built-in
functions, including:

- Managing devices.
- Filesystem handling.
- Basic file I/O.
- Logging.

Design Principle:

- All submodules must be ATOMIC and self-contained.
- Each submodule should extend a Python module.
- The import statement for each submodule should look like this:
    - Before: `import math`
    - Now: `from mon.core import math`
"""

from __future__ import annotations

import mon.core.builtins
import mon.core.config
import mon.core.enum
import mon.core.factory
import mon.core.file
import mon.core.logging
import mon.core.math
import mon.core.pathlib
import mon.core.pynvml
import mon.core.rich
from mon.core.builtins import *
from mon.core.config import *
from mon.core.enum import *
from mon.core.factory import *
from mon.core.file import *
from mon.core.logging import *
from mon.core.pathlib import *
from mon.core.rich import (
    console, error_console, get_progress_bar, print_dict, print_table,
)
