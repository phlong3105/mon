#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from __future__ import annotations

import unittest

import mon


# region TestCase

class TestModel(unittest.TestCase):
    
    def test_alexnet(self):
        model = mon.AlexNet(
            config      = "alexnet.yaml",
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = "imagenet",
            fullname    = "alexnet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_googlenet(self):
        model = mon.GoogleNet(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = False,
            fullname    = "googlenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
        
    def test_finet(self):
        model = mon.FINet(
            config      = "finet-a.yaml",
            hparams     = None,
            channels    = 3,
            num_classes = None,
            classlabels = None,
            weights     = False,
            fullname    = "finet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
        
    def test_hinet(self):
        model = mon.HINet(
            config      = "hinet.yaml",
            hparams     = None,
            channels    = 3,
            num_classes = None,
            classlabels = None,
            weights     = False,
            fullname    = "hinet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_lenet(self):
        model = mon.LeNet(
            config      = "lenet.yaml",
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = None,
            fullname    = "lenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_vgg11(self):
        model = mon.VGG11(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = None,
            fullname    = "vgg11",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_vgg13(self):
        model = mon.VGG13(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = None,
            fullname    = "vgg13",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_vgg16(self):
        model = mon.VGG16(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = None,
            fullname    = "vgg16",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_vgg19(self):
        model = mon.VGG19(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = None,
            fullname    = "vgg19",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_zeroadce(self):
        model = mon.ZeroADCE(
            config      = "zeroadce-b.yaml",
            hparams     = None,
            channels    = 3,
            num_classes = None,
            classlabels = None,
            weights     = False,
            fullname    = "zeroadce-b",
            verbose     = True,
        )
        self.assertIsNotNone(model)
        
    def test_zerodce(self):
        model = mon.ZeroDCE(
            config      = "zerodce.yaml",
            hparams     = None,
            channels    = 3,
            num_classes = None,
            classlabels = None,
            weights     = False,
            fullname    = "zerodce",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_zerodce_tiny(self):
        model = mon.ZeroDCETiny(
            config      = "zerodce-tiny.yaml",
            hparams     = None,
            channels    = 3,
            num_classes = None,
            classlabels = None,
            weights     = False,
            fullname    = "zerodce-tiny",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_zerodcepp(self):
        model = mon.ZeroDCEPP(
            config      = "zerodce++.yaml",
            hparams     = None,
            channels    = 3,
            num_classes = None,
            classlabels = None,
            weights     = False,
            fullname    = "zerodce++",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
# endregion


# region Main
        
if __name__ == "__main__":
    unittest.main()
    
# endregion
