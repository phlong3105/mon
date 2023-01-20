#!/usr/bin/env python
# -*- coding: utf-8 -*-

r""":mod:`test.test_mon_coreimage` perform unit testings for
:mod:`mon.coreimage` package.
"""

from __future__ import annotations

import unittest

from mon.foundation import pathlib


class TestPathLib(unittest.TestCase):
    
    def test_path(self):
        path = pathlib.Path("lenna.png")
        print(path)
        self.assertIsNotNone(path)


if __name__ == "__main__":
    unittest.main()
