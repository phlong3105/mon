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
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = "imagenet",
            fullname    = "alexnet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_densenet121(self):
        model = mon.DenseNet121(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = "imagenet",
            fullname    = "densenet121",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_densenet161(self):
        model = mon.DenseNet161(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = "imagenet",
            fullname    = "densenet161",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_densenet169(self):
        model = mon.DenseNet169(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = "imagenet",
            fullname    = "densenet169",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_densenet201(self):
        model = mon.DenseNet201(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = "imagenet",
            fullname    = "densenet201",
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
    
    def test_inception(self):
        model = mon.Inception(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = False,
            fullname    = "inception3",
            verbose     = True,
        )
        self.assertIsNotNone(model)
        
    def test_lenet(self):
        model = mon.LeNet(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = None,
            fullname    = "lenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_resnet18(self):
        model = mon.ResNet18(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = "imagenet",
            fullname    = "resnet18",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_resnet34(self):
        model = mon.ResNet34(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = "imagenet",
            fullname    = "resnet34",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_resnet50(self):
        model = mon.ResNet50(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = "imagenet",
            fullname    = "resnet50",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_resnet101(self):
        model = mon.ResNet101(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = "imagenet",
            fullname    = "resnet101",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_resnet152(self):
        model = mon.ResNet152(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = "imagenet",
            fullname    = "resnet152",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_resnext50_32x4d(self):
        model = mon.ResNeXt50_32X4D(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = "imagenet",
            fullname    = "resnext50-32x4d",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_resnext101_32x8d(self):
        model = mon.ResNeXt101_32X8D(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = "imagenet",
            fullname    = "resnext101-32x8d",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_resnext101_64x4d(self):
        model = mon.ResNeXt101_64X4D(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = "imagenet",
            fullname    = "resnext101-64x4d",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_shufflenetv2_x0_5(self):
        model = mon.ShuffleNetV2_x0_5(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = None,
            fullname    = "shufflenet-v2-x0.5",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_shufflenetv2_x1_0(self):
        model = mon.ShuffleNetV2_x1_0(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = None,
            fullname    = "shufflenet-v2-x1.0",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_shufflenetv2_x1_5(self):
        model = mon.ShuffleNetV2_x1_5(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = None,
            fullname    = "shufflenet-v2-x1.5",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_shufflenetv2_x2_0(self):
        model = mon.ShuffleNetV2_x2_0(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = None,
            fullname    = "shufflenet-v2-x2.0",
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
    
    def test_wide_resnet50(self):
        model = mon.WideResNet50(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = None,
            fullname    = "wide-resnet50",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_wide_resnet101(self):
        model = mon.WideResNet101(
            hparams     = None,
            channels    = 3,
            num_classes = 10,
            classlabels = None,
            weights     = None,
            fullname    = "wide-resnet101",
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
