#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Loading Models
"""

from __future__ import annotations

import unittest

from one.vision.classification.alexnet import AlexNet
from one.vision.enhancement.ffanet import FFANet
from one.vision.enhancement.hinet import HINet
from one.vision.enhancement.mbllen import MBLLEN
from one.vision.enhancement.zerodce import ZeroDCE
from one.vision.enhancement.zerodcepp import ZeroDCEPP


class TestModel(unittest.TestCase):
    
    def test_alexnet(self):
        alexnet = AlexNet(pretrained="imagenet", verbose=True)
        self.assertIsNotNone(alexnet)
        
    def test_ffanet(self):
        ffanet = FFANet(verbose=True)
        self.assertIsNotNone(ffanet)
    
    def test_hinet(self):
        hinet = HINet(verbose=True)
        self.assertIsNotNone(hinet)
        
    def test_mbllen(self):
        mbllen = MBLLEN(verbose=True)
        self.assertIsNotNone(mbllen)
        
    def test_zerodce(self):
        zerodce = ZeroDCE(verbose=True)
        self.assertIsNotNone(zerodce)

    def test_zerodcepp(self):
        zerodcepp = ZeroDCEPP(verbose=True)
        self.assertIsNotNone(zerodcepp)
