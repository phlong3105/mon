#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from __future__ import annotations

import unittest

import albumentations as A

from mon.globals import DATA_DIR, DATAMODULES
from mon.vision import visualize


# region Helper Function

def load_image_enhancement_dataset(config):
    dm = DATAMODULES.build(config=config)
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
        denormalize = config["to_tensor"],
    )
    visualize.plt.show(block=True)

# endregion


# region TestCase

class TestDataset(unittest.TestCase):
    
    '''
    def test_a2i2_haze(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_cifar_10(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_cifar_100(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
        
    def test_dcim(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_densehaze(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
        
    def test_fashion_mnist(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
        
    def test_gt_rain(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_ihaze(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_kodas_lol19(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_lime(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_lol(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    '''
    
    def test_lol226(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    '''
    def test_lol4k(self):
        config = {
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
        load_image_enhancement_dataset(config=config)

    def test_mef(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_mnist(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_nhhaze(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_npe(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_ohaze(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_rain100(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_rain100h(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_rain100l(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_rain12(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_rain1200(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_rain13k(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_rain1400(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_rain2800(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_rain800(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_satehaze1k(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_satehaze1k_moderate(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_satehaze1k_thick(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_satehaze1k_thin(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_sice(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_sice_unsupervised(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
        
    def test_snow100k(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_snow100k_small(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_snow100k_medium(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_snow100k_large(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
        
    def test_vip(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    
    def test_vv(self):
        config = {
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
        load_image_enhancement_dataset(config=config)
    '''
    
# endregion


# region Main
        
if __name__ == "__main__":
    unittest.main()
    
# endregion
