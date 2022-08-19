#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Loading Models
"""

from __future__ import annotations

import unittest

from one.vision.enhancement import *


class TestModel(unittest.TestCase):
    
    def test_ffanet(self):
        ffanet = FFANet(verbose=True)
        self.assertIsNotNone(ffanet)
        
    def test_mbllen(self):
        mbllen = MBLLEN(verbose=True)
        self.assertIsNotNone(mbllen)
        
    def test_zerodce(self):
        zerodce = ZeroDCE(verbose=True)
        self.assertIsNotNone(zerodce)

    def test_zerodcepp(self):
        zerodcepp = ZeroDCEPP(verbose=True)
        self.assertIsNotNone(zerodcepp)
