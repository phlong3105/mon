#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Loading Models
"""

from __future__ import annotations

import unittest

from one.vision.classification.alexnet import AlexNet
from one.vision.classification.convnext import ConvNeXtBase
from one.vision.classification.convnext import ConvNeXtLarge
from one.vision.classification.convnext import ConvNeXtSmall
from one.vision.classification.convnext import ConvNeXtTiny
from one.vision.classification.densenet import DenseNet121
from one.vision.classification.densenet import DenseNet161
from one.vision.classification.densenet import DenseNet169
from one.vision.classification.densenet import DenseNet201
from one.vision.classification.googlenet import GoogleNet
from one.vision.classification.inception import Inception3
from one.vision.classification.lenet import LeNet
from one.vision.classification.resnet import ResNet101
from one.vision.classification.resnet import ResNet152
from one.vision.classification.resnet import ResNet18
from one.vision.classification.resnet import ResNet34
from one.vision.classification.resnet import ResNet50
from one.vision.classification.resnext import ResNeXt101_32X8D
from one.vision.classification.resnext import ResNeXt101_64X4D
from one.vision.classification.resnext import ResNeXt50_32X4D
from one.vision.classification.shufflenet_v2 import ShuffleNetV2_x0_5
from one.vision.classification.shufflenet_v2 import ShuffleNetV2_x1_0
from one.vision.classification.shufflenet_v2 import ShuffleNetV2_x1_5
from one.vision.classification.shufflenet_v2 import ShuffleNetV2_x2_0
from one.vision.classification.squeezenet import SqueezeNet_1_0
from one.vision.classification.squeezenet import SqueezeNet_1_1
from one.vision.classification.vgg import VGG11
from one.vision.classification.vgg import VGG11Bn
from one.vision.classification.vgg import VGG13
from one.vision.classification.vgg import VGG13Bn
from one.vision.classification.vgg import VGG16
from one.vision.classification.vgg import VGG16Bn
from one.vision.classification.vgg import VGG19
from one.vision.classification.vgg import VGG19Bn
from one.vision.classification.wide_resnet import Wide_ResNet101
from one.vision.classification.wide_resnet import Wide_ResNet50
from one.vision.enhancement.ffanet import FFANet
from one.vision.enhancement.finet import FINet
from one.vision.enhancement.finet import FINetDeblur
from one.vision.enhancement.finet import FINetDenoise
from one.vision.enhancement.finet import FINetDerain
from one.vision.enhancement.hinet import HINet
from one.vision.enhancement.hinet import HINetDeblur
from one.vision.enhancement.hinet import HINetDenoise
from one.vision.enhancement.hinet import HINetDerain
from one.vision.enhancement.mbllen import MBLLEN
from one.vision.enhancement.retinexnet import RetinexNet
from one.vision.enhancement.zerodce import ZeroDCE
from one.vision.enhancement.zerodcepp import ZeroDCEPP
from one.vision.segmentation.unet import UNet


class TestModel(unittest.TestCase):
    
    def test_alexnet(self):
        alexnet = AlexNet(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(alexnet)
    
    def test_convnext_base(self):
        convnext_base = ConvNeXtBase(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(convnext_base)
    
    def test_convnext_tiny(self):
        convnext_tiny = ConvNeXtTiny(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(convnext_tiny)
        
    def test_convnext_small(self):
        convnext_small = ConvNeXtSmall(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(convnext_small)
    
    def test_convnext_large(self):
        convnext_large = ConvNeXtLarge(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(convnext_large)
        
    def test_densenet121(self):
        densenet121 = DenseNet121(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(densenet121)
    
    def test_densenet161(self):
        densenet161 = DenseNet161(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(densenet161)
    
    def test_densenet169(self):
        densenet169 = DenseNet169(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(densenet169)
    
    def test_densenet201(self):
        densenet201 = DenseNet201(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(densenet201)
        
    def test_ffanet(self):
        ffanet = FFANet(verbose=True)
        self.assertIsNotNone(ffanet)
    
    def test_finet(self):
        finet = FINet(verbose=True)
        self.assertIsNotNone(finet)
    
    def test_finet_deblur(self):
        finet_deblur = FINetDeblur( verbose=True)
        self.assertIsNotNone(finet_deblur)
    
    def test_finet_denoise(self):
        finet_denoise = FINetDenoise(verbose=True)
        self.assertIsNotNone(finet_denoise)
    
    def test_finet_derain(self):
        finet_derain = FINetDerain(verbose=True)
        self.assertIsNotNone(finet_derain)
    
    def test_googlenet(self):
        googlenet = GoogleNet(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(googlenet)
    
    def test_hinet(self):
        hinet = HINet(verbose=True)
        self.assertIsNotNone(hinet)
    
    def test_hinet_deblur(self):
        hinet_deblur = HINetDeblur(pretrained="gopro", verbose=True)
        self.assertIsNotNone(hinet_deblur)
    
    def test_hinet_denoise(self):
        hinet_denoise = HINetDenoise(pretrained="sidd", verbose=True)
        self.assertIsNotNone(hinet_denoise)
    
    def test_hinet_derain(self):
        hinet_derain = HINetDerain(pretrained="rain13k", verbose=True)
        self.assertIsNotNone(hinet_derain)
    
    def test_inception3(self):
        inception3 = Inception3(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(inception3)
    
    def test_lenet(self):
        lenet = LeNet(num_classes=1000, verbose=True)
        self.assertIsNotNone(lenet)
        
    def test_mbllen(self):
        mbllen = MBLLEN(verbose=True)
        self.assertIsNotNone(mbllen)
    
    def test_resnet18(self):
        resnet18 = ResNet18(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(resnet18)
        
    def test_resnet34(self):
        resnet34 = ResNet34(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(resnet34)
    
    def test_resnet50(self):
        resnet50 = ResNet50(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(resnet50)
    
    def test_resnet101(self):
        resnet101 = ResNet101(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(resnet101)
    
    def test_resnet152(self):
        resnet152 = ResNet152(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(resnet152)
    
    def test_resnext50_32x4d(self):
        resnext50_32x4d = ResNeXt50_32X4D(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(resnext50_32x4d)

    def test_resnext101_32x8d(self):
        resnext101_32x8d = ResNeXt101_32X8D(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(resnext101_32x8d)
    
    def test_resnext101_64x4d(self):
        resnext101_64x4d = ResNeXt101_64X4D(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(resnext101_64x4d)
        
    def test_retinexnet(self):
        retinexnet = RetinexNet(verbose=True)
        self.assertIsNotNone(retinexnet)
        
    def test_shufflenet_v2_x0_5(self):
        shufflenet_v2_x0_5 = ShuffleNetV2_x0_5(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(shufflenet_v2_x0_5)
    
    def test_shufflenet_v2_x1_0(self):
        shufflenet_v2_x1_0 = ShuffleNetV2_x1_0(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(shufflenet_v2_x1_0)
    
    def test_shufflenet_v2_x1_5(self):
        shufflenet_v2_x1_5 = ShuffleNetV2_x1_5(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(shufflenet_v2_x1_5)
    
    def test_shufflenet_v2_x2_0(self):
        shufflenet_v2_x2_0 = ShuffleNetV2_x2_0(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(shufflenet_v2_x2_0)
        
    def test_squeezenet_1_0(self):
        squeezenet_1_0 = SqueezeNet_1_0(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(squeezenet_1_0)
    
    def test_squeezenet_1_1(self):
        squeezenet_1_1 = SqueezeNet_1_1(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(squeezenet_1_1)
    
    def test_unet_32(self):
        unet_32 = UNet(cfg="unet-32.yaml", pretrained="lgg", verbose=True)
        self.assertIsNotNone(unet_32)
    
    def test_vgg11(self):
        vgg11 = VGG11(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(vgg11)
        
    def test_vgg11_bn(self):
        vgg11_bn = VGG11Bn(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(vgg11_bn)
    
    def test_vgg13(self):
        vgg13 = VGG13(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(vgg13)
    
    def test_vgg13_bn(self):
        vgg13_bn = VGG13Bn(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(vgg13_bn)
    
    def test_vgg16(self):
        vgg16 = VGG16(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(vgg16)
    
    def test_vgg16_bn(self):
        vgg16_bn = VGG16Bn(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(vgg16_bn)

    def test_vgg19(self):
        vgg19 = VGG19(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(vgg19)

    def test_vgg19_bn(self):
        vgg19_bn = VGG19Bn(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(vgg19_bn)
    
    def test_wide_resnet50(self):
        wide_resnet50 = Wide_ResNet50(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(wide_resnet50)
    
    def test_wide_resnet101(self):
        wide_resnet101 = Wide_ResNet101(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(wide_resnet101)
    
    def test_zerodce(self):
        zerodce = ZeroDCE(pretrained="sice", verbose=True)
        self.assertIsNotNone(zerodce)

    def test_zerodcepp(self):
        zerodcepp = ZeroDCEPP(verbose=True)
        self.assertIsNotNone(zerodcepp)
