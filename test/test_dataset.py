#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from __future__ import annotations

import unittest

import albumentations as A

from mon.globals import DATA_DIR, DATAMODULES
from mon.vision import visualize


# region Helper Function

def load_image_enhancement_dataset(cfg):
    dm = DATAMODULES.build(config=cfg)
    dm.setup()
    if dm.classlabels:
        dm.classlabels.print()
    data_iter = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    result = {"image": input, "target": target}
    label  = [(m["name"]) for m in meta]
    visualize.imshow_enhancement(
        winname     = "image",
        image       = result,
        label       = label,
        denormalize = cfg["to_tensor"],
    )
    visualize.plt.show(block=True)

# endregion


# region TestCase

class TestDataset(unittest.TestCase):
    
    def test_a2i2_haze(self):
        cfg = {
            "name"        : "a2i2-haze",
            "root"        : DATA_DIR / "a2i2-haze",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_cifar_10(self):
        cfg = {
            "name"        : "cifar-10",
            "root"        : DATA_DIR / "cifar" / "cifar-10",
            "image_size"  : 32,
            "transform"   : A.Compose(
                [
                    A.Resize(width=32, height=32),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_cifar_100(self):
        cfg = {
            "name"        : "cifar-100",
            "root"        : DATA_DIR / "cifar" / "cifar-100",
            "image_size"  : 32,
            "transform"   : A.Compose(
                [
                    A.Resize(width=32, height=32),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
        
    def test_dcim(self):
        cfg = {
            "name"        : "dcim",
            "root"        : DATA_DIR / "lol",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_densehaze(self):
        cfg = {
            "name"        : "dense-haze",
            "root"        : DATA_DIR / "ntire" / "dense-haze",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
        
    def test_fashion_mnist(self):
        cfg = {
            "name"        : "fashion-mnist",
            "root"        : DATA_DIR / "mnist" / "fashion-mnist",
            "image_size"  : 32,
            "transform"   : A.Compose(
                [
                    A.Resize(width=32, height=32),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
        
    def test_gt_rain(self):
        cfg = {
            "name"        : "gt-rain",
            "root"        : DATA_DIR / "gt-rain",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_ihaze(self):
        cfg = {
            "name"        : "i-haze",
            "root"        : DATA_DIR / "ntire" / "i-haze",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_kodas_lol19(self):
        cfg = {
            "name"        : "kodas-lol19",
            "root"        : DATA_DIR / "kodas" / "lol19",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_lime(self):
        cfg = {
            "name"        : "lime",
            "root"        : DATA_DIR / "lol",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_lol(self):
        cfg = {
            "name"        : "lol",
            "root"        : DATA_DIR / "lol",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_lol226(self):
        cfg = {
            "name"        : "lol226",
            "root"        : DATA_DIR / "lol",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_lol4k(self):
        cfg = {
            "name"        : "lol4k",
            "root"        : DATA_DIR / "lol",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)

    def test_mef(self):
        cfg = {
            "name"        : "mef",
            "root"        : DATA_DIR / "lol",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_mnist(self):
        cfg = {
            "name"        : "mnist",
            "root"        : DATA_DIR / "mnist" / "mnist",
            "image_size"  : 32,
            "transform"   : A.Compose(
                [
                    A.Resize(width=32, height=32),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_nhhaze(self):
        cfg = {
            "name"        : "nh-haze",
            "root"        : DATA_DIR / "ntire" / "nh-haze",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_npe(self):
        cfg = {
            "name"        : "npe",
            "root"        : DATA_DIR / "lol",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_ohaze(self):
        cfg = {
            "name"        : "o-haze",
            "root"        : DATA_DIR / "ntire" / "o-haze",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_rain100(self):
        cfg = {
            "name"        : "rain100",
            "root"        : DATA_DIR / "rain13k" / "rain100",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_rain100h(self):
        cfg = {
            "name"        : "rain100h",
            "root"        : DATA_DIR / "rain13k" / "rain100h",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_rain100l(self):
        cfg = {
            "name"        : "rain100l",
            "root"        : DATA_DIR / "rain13k" / "rain100l",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_rain12(self):
        cfg = {
            "name"        : "rain12",
            "root"        : DATA_DIR / "rain13k" / "rain12",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_rain1200(self):
        cfg = {
            "name"        : "rain1200",
            "root"        : DATA_DIR / "rain13k" / "rain1200",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_rain13k(self):
        cfg = {
            "name"        : "rain1200",
            "root"        : DATA_DIR / "rain13k",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_rain1400(self):
        cfg = {
            "name"        : "rain1400",
            "root"        : DATA_DIR / "rain13k" / "rain1400",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_rain2800(self):
        cfg = {
            "name"        : "rain2800",
            "root"        : DATA_DIR / "rain13k" / "rain2800",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_rain800(self):
        cfg = {
            "name"        : "rain800",
            "root"        : DATA_DIR / "rain13k" / "rain800",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_satehaze1k(self):
        cfg = {
            "name"        : "satehaze1k",
            "root"        : DATA_DIR / "satehaze1k",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_satehaze1k_moderate(self):
        cfg = {
            "name"        : "satehaze1k-moderate",
            "root"        : DATA_DIR / "satehaze1k",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_satehaze1k_thick(self):
        cfg = {
            "name"        : "satehaze1k-thick",
            "root"        : DATA_DIR / "satehaze1k",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_satehaze1k_thin(self):
        cfg = {
            "name"        : "satehaze1k-thin",
            "root"        : DATA_DIR / "satehaze1k",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_sice(self):
        cfg = {
            "name"        : "sice",
            "root"        : DATA_DIR / "sice",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_sice_unsupervised(self):
        cfg = {
            "name"        : "sice-unsupervised",
            "root"        : DATA_DIR / "sice",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
        
    def test_snow100k(self):
        cfg = {
            "name"        : "snow100k",
            "root"        : DATA_DIR / "snow100k",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_snow100k_small(self):
        cfg = {
            "name"        : "snow100k-small",
            "root"        : DATA_DIR / "snow100k",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_snow100k_medium(self):
        cfg = {
            "name"        : "snow100k-medium",
            "root"        : DATA_DIR / "snow100k",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_snow100k_large(self):
        cfg = {
            "name"        : "snow100k-large",
            "root"        : DATA_DIR / "snow100k",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
        
    def test_vip(self):
        cfg = {
            "name"        : "vip",
            "root"        : DATA_DIR / "lol",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
    
    def test_vv(self):
        cfg = {
            "name"        : "vv",
            "root"        : DATA_DIR / "lol",
            "image_size"  : 256,
            "transform"   : A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.8),
                ]
            ),
            "to_tensor"   : False,
            "cache_data"  : False,
            "cache_images": False,
            "batch_size"  : 8,
            "devices"     : 0,
            "shuffle"     : True,
            "verbose"     : True,
        }
        load_image_enhancement_dataset(cfg=cfg)
        
# endregion


# region Main
        
if __name__ == "__main__":
    unittest.main()
    
# endregion
