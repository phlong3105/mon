#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The `foundation` package provides a base layer of functionality by extending
`Python <https://www.python.org/>`_ built-in functions.

The low-level operating system operations include:

- Managing devices.
- Filesystem handling.
- Basic file I/O.
- Logging.

Design Principle:

- All submodules must be ATOMIC and self-contained.
- Each submodule should extend a Python module.
- The import statement for each submodule should look like this:
    - Before: `import math`
    - Now: `from mon.foundation import math`
"""

from __future__ import annotations

import mon.foundation.builtins
import mon.foundation.config
import mon.foundation.enum
import mon.foundation.factory
import mon.foundation.file
import mon.foundation.logging
import mon.foundation.math
import mon.foundation.pathlib
import mon.foundation.pynvml
import mon.foundation.rich
from mon.foundation.builtins import *
from mon.foundation.config import *
from mon.foundation.enum import *
from mon.foundation.factory import *
from mon.foundation.file import *
from mon.foundation.logging import *
from mon.foundation.pathlib import *
from mon.foundation.rich import (
    console, error_console, get_progress_bar, print_dict, print_table,
)
