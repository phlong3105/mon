#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from __future__ import annotations

import unittest

import mon


# region TestCase

class TestModel(unittest.TestCase):
    
    # Classify
    
    def test_alexnet(self):
        num_classes = mon.AlexNet.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.AlexNet(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_convnext_base(self):
        num_classes = mon.ConvNeXtBase.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ConvNeXtBase(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_convnext_tiny(self):
        num_classes = mon.ConvNeXtTiny.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ConvNeXtTiny(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_convnext_small(self):
        num_classes = mon.ConvNeXtSmall.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ConvNeXtSmall(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_convnext_large(self):
        num_classes = mon.ConvNeXtLarge.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ConvNeXtLarge(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_densenet121(self):
        num_classes = mon.DenseNet121.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.DenseNet121(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_densenet161(self):
        num_classes = mon.DenseNet161.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.DenseNet161(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_densenet169(self):
        num_classes = mon.DenseNet169.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.DenseNet169(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_densenet201(self):
        num_classes = mon.DenseNet201.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.DenseNet201(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_efficientnet_b0(self):
        num_classes = mon.EfficientNet_B0.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.EfficientNet_B0(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_efficientnet_b1(self):
        num_classes = mon.EfficientNet_B1.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.EfficientNet_B1(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.EfficientNet_B1.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.EfficientNet_B1(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
    
    def test_efficientnet_b2(self):
        num_classes = mon.EfficientNet_B2.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.EfficientNet_B2(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_efficientnet_b3(self):
        num_classes = mon.EfficientNet_B3.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.EfficientNet_B3(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_efficientnet_b42(self):
        num_classes = mon.EfficientNet_B4.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.EfficientNet_B4(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_efficientnet_b5(self):
        num_classes = mon.EfficientNet_B5.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.EfficientNet_B5(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_efficientnet_b6(self):
        num_classes = mon.EfficientNet_B6.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.EfficientNet_B6(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_efficientnet_b7(self):
        num_classes = mon.EfficientNet_B7.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.EfficientNet_B7(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_efficientnet_v2_s(self):
        num_classes = mon.EfficientNet_V2_S.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.EfficientNet_V2_S(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_efficientnet_v2_m(self):
        num_classes = mon.EfficientNet_V2_M.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.EfficientNet_V2_M(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_efficientnet_v2_l(self):
        num_classes = mon.EfficientNet_V2_L.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.EfficientNet_V2_L(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_googlenet(self):
        num_classes = mon.GoogleNet.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.GoogleNet(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_inception3(self):
        num_classes = mon.Inception3.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.Inception3(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_mnasnet0_5(self):
        num_classes = mon.MNASNet0_5.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.MNASNet0_5(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_mnasnet0_75(self):
        num_classes = mon.MNASNet0_75.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.MNASNet0_75(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_mnasnet1_0(self):
        num_classes = mon.MNASNet1_0.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.MNASNet1_0(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_mnasnet1_3(self):
        num_classes = mon.MNASNet1_3.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.MNASNet1_3(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_mobilenet_v2(self):
        num_classes = mon.MobileNetV2.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.MobileNetV2(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.MobileNetV2.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.MobileNetV2(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
        
    def test_mobilenetv3_large(self):
        num_classes = mon.MobileNetV3_Large.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.MobileNetV3_Large(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.MobileNetV3_Large.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.MobileNetV3_Large(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
    
    def test_mobilenetv3_small(self):
        num_classes = mon.MobileNetV3_Small.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.MobileNetV3_Small(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_regnet_y_400mf(self):
        num_classes = mon.RegNet_Y_400MF.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.RegNet_Y_400MF(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.RegNet_Y_400MF.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.RegNet_Y_400MF(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
    
    def test_regnet_y_800mf(self):
        num_classes = mon.RegNet_Y_800MF.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.RegNet_Y_800MF(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.RegNet_Y_800MF.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.RegNet_Y_800MF(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
    
    def test_regnet_y_1_6gf(self):
        num_classes = mon.RegNet_Y_1_6GF.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.RegNet_Y_1_6GF(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.RegNet_Y_1_6GF.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.RegNet_Y_1_6GF(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
    
    def test_regnet_y_3_2gf(self):
        num_classes = mon.RegNet_Y_3_2GF.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.RegNet_Y_3_2GF(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.RegNet_Y_3_2GF.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.RegNet_Y_3_2GF(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
    
    def test_regnet_y_8gf(self):
        num_classes = mon.RegNet_Y_8GF.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.RegNet_Y_8GF(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.RegNet_Y_8GF.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.RegNet_Y_8GF(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
    
    def test_regnet_y_16gf(self):
        num_classes = mon.RegNet_Y_16GF.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.RegNet_Y_16GF(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.RegNet_Y_16GF.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.RegNet_Y_16GF(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.RegNet_Y_16GF.zoo["imagenet1k-swag-e2e-v1"]["num_classes"]
        model = mon.RegNet_Y_16GF(channels=3, num_classes=num_classes, weights="imagenet1k-swag-e2e-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.RegNet_Y_16GF.zoo["imagenet1k-swag-lc-v1"]["num_classes"]
        model = mon.RegNet_Y_16GF(channels=3, num_classes=num_classes, weights="imagenet1k-swag-lc-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_regnet_y_32gf(self):
        num_classes = mon.RegNet_Y_32GF.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.RegNet_Y_32GF(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.RegNet_Y_32GF.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.RegNet_Y_32GF(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.RegNet_Y_32GF.zoo["imagenet1k-swag-e2e-v1"]["num_classes"]
        model = mon.RegNet_Y_32GF(channels=3, num_classes=num_classes, weights="imagenet1k-swag-e2e-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.RegNet_Y_32GF.zoo["imagenet1k-swag-lc-v1"]["num_classes"]
        model = mon.RegNet_Y_32GF(channels=3, num_classes=num_classes, weights="imagenet1k-swag-lc-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_regnet_y_128gf(self):
        num_classes = mon.RegNet_Y_128GF.zoo["imagenet1k-swag-e2e-v1"]["num_classes"]
        model = mon.RegNet_Y_128GF(channels=3, num_classes=num_classes, weights="imagenet1k-swag-e2e-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.RegNet_Y_128GF.zoo["imagenet1k-swag-lc-v1"]["num_classes"]
        model = mon.RegNet_Y_128GF(channels=3, num_classes=num_classes, weights="imagenet1k-swag-lc-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_regnet_x_400mf(self):
        num_classes = mon.RegNet_X_400MF.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.RegNet_X_400MF(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.RegNet_X_400MF.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.RegNet_X_400MF(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
    
    def test_regnet_x_800mf(self):
        num_classes = mon.RegNet_X_800MF.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.RegNet_X_800MF(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.RegNet_X_800MF.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.RegNet_X_800MF(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
    
    def test_regnet_x_1_6gf(self):
        num_classes = mon.RegNet_X_1_6GF.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.RegNet_X_1_6GF(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.RegNet_X_1_6GF.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.RegNet_X_1_6GF(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
    
    def test_regnet_x_3_2gf(self):
        num_classes = mon.RegNet_X_3_2GF.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.RegNet_X_3_2GF(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.RegNet_X_3_2GF.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.RegNet_X_3_2GF(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
    
    def test_regnet_x_8gf(self):
        num_classes = mon.RegNet_X_8GF.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.RegNet_Y_8GF(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.RegNet_X_8GF.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.RegNet_Y_8GF(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
    
    def test_regnet_x_16gf(self):
        num_classes = mon.RegNet_X_16GF.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.RegNet_X_16GF(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.RegNet_X_16GF.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.RegNet_X_16GF(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
    
    def test_regnet_x32gf(self):
        num_classes = mon.RegNetX_32GF.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.RegNetX_32GF(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.RegNetX_32GF.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.RegNetX_32GF(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
    
    def test_resnet18(self):
        num_classes = mon.ResNet18.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ResNet18(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
    def test_resnet34(self):
        num_classes = mon.ResNet34.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ResNet34(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
    def test_resnet50(self):
        num_classes = mon.ResNet50.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ResNet50(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.ResNet50.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.ResNet50(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
    
    def test_resnet101(self):
        num_classes = mon.ResNet101.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ResNet101(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.ResNet101.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.ResNet101(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
    
    def test_resnet152(self):
        num_classes = mon.ResNet152.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ResNet152(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.ResNet152.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.ResNet152(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
    
    def test_resnext50_32x4d(self):
        num_classes = mon.ResNeXt50_32X4D.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ResNeXt50_32X4D(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.ResNeXt50_32X4D.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.ResNeXt50_32X4D(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
        
    def test_resnext101_32x8d(self):
        num_classes = mon.ResNeXt101_32X8D.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ResNeXt101_32X8D(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.ResNeXt101_32X8D.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.ResNeXt101_32X8D(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
        
    def test_resnext101_64x4d(self):
        num_classes = mon.ResNeXt101_64X4D.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ResNeXt101_64X4D(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
    def test_wide_resnet50(self):
        num_classes = mon.WideResNet50.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.WideResNet50(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.WideResNet50.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.WideResNet50(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
    
    def test_wide_resnet101(self):
        num_classes = mon.WideResNet101.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.WideResNet101(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.WideResNet101.zoo["imagenet1k-v2"]["num_classes"]
        model = mon.WideResNet101(channels=3, num_classes=num_classes, weights="imagenet1k-v2", verbose=True)
        self.assertIsNotNone(model)
    
    def test_shufflenetv2_x0_5(self):
        num_classes = mon.ShuffleNetV2_x0_5.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ShuffleNetV2_x0_5(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
    def test_shufflenetv2_x1(self):
        num_classes = mon.ShuffleNetV2_X1_0.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ShuffleNetV2_X1_0(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
    def test_shufflenetv2_x1_5(self):
        num_classes = mon.ShuffleNetV2_X1_5.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ShuffleNetV2_X1_5(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
    def test_shufflenetv2_x2_0(self):
        num_classes = mon.ShuffleNetV2_X2_0.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ShuffleNetV2_X2_0(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_squeezenet1_0(self):
        num_classes = mon.SqueezeNet1_0.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.SqueezeNet1_0(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_squeezenet1_1(self):
        num_classes = mon.SqueezeNet1_1.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.SqueezeNet1_1(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_swin_t(self):
        num_classes = mon.SqueezeNet1_1.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.Swin_T(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_swin_s(self):
        num_classes = mon.SqueezeNet1_1.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.Swin_S(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_swin_b(self):
        num_classes = mon.Swin_B.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.Swin_B(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_swin_v2_t(self):
        num_classes = mon.Swin_V2_T.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.Swin_V2_T(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_swin_v2_s(self):
        num_classes = mon.Swin_V2_S.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.Swin_V2_S(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_swin_v2_b(self):
        num_classes = mon.Swin_V2_B.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.Swin_V2_B(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_vgg11(self):
        num_classes = mon.VGG11.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.VGG11(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_vgg11_bn(self):
        num_classes = mon.VGG11_BN.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.VGG11_BN(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_vgg13(self):
        num_classes = mon.VGG13.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.VGG13(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_vgg13_bn(self):
        num_classes = mon.VGG13_BN.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.VGG13_BN(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_vgg16(self):
        num_classes = mon.VGG16.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.VGG16(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
        num_classes = mon.VGG16.zoo["imagenet1k-features"]["num_classes"]
        model = mon.VGG16(channels=3, num_classes=num_classes, weights="imagenet1k-features", verbose=True)
        self.assertIsNotNone(model)
    
    def test_vgg16_bn(self):
        num_classes = mon.VGG16_BN.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.VGG16_BN(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_vgg19(self):
        num_classes = mon.VGG19.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.VGG19(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_vgg19_bn(self):
        num_classes = mon.VGG19_BN.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.VGG19_BN(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_vit_b_16(self):
        num_classes = mon.ViT_B_16.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ViT_B_16(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
        
    def test_vit_b_32(self):
        num_classes = mon.ViT_B_32.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ViT_B_32(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_vit_l_16(self):
        num_classes = mon.ViT_L_16.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ViT_L_16(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    def test_vit_l_32(self):
        num_classes = mon.ViT_L_32.zoo["imagenet1k-v1"]["num_classes"]
        model = mon.ViT_L_32(channels=3, num_classes=num_classes, weights="imagenet1k-v1", verbose=True)
        self.assertIsNotNone(model)
    
    # Enhance
    
    def test_hinet(self):
        model = mon.HINet(weights="gopro", verbose=True)
        self.assertIsNotNone(model)
        model = mon.HINet(weights="reds", verbose=True)
        self.assertIsNotNone(model)
        model = mon.HINet(weights="sidd", verbose=True)
        self.assertIsNotNone(model)
        model = mon.HINet(weights="rain13k", verbose=True)
        self.assertIsNotNone(model)
    
    def test_zerodce(self):
        model = mon.ZeroDCE(weights="sice-zerodce", verbose=True)
        self.assertIsNotNone(model)
    
    def test_zerodcepp(self):
        model = mon.ZeroDCEPP(weights="sice-zerodce", verbose=True)
        self.assertIsNotNone(model)
    
# endregion


# region Main
        
if __name__ == "__main__":
    unittest.main()
    
# endregion
