#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements light effect suppression (les) datasets and
datamodules.
"""

from __future__ import annotations

__all__ = [
    "Flare7KPPReal",
    "Flare7KPPRealDataModule",
    "Flare7KPPSyn",
    "Flare7KPPSynDataModule",
    "FlareReal800",
    "FlareReal800DataModule",
    "LEDLight",
    "LEDLightDataModule",
    "LightEffect",
    "LightEffectDataModule",
]

import random
from typing import Literal

import albumentations as A
import numpy as np
import torch
import torchvision.transforms as TT

from mon import core
from mon.data.datastruct import annotation as anno, datamodule, dataset
from mon.data.transform import transform
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console          = core.console
ClassLabels      = anno.ClassLabels
default_root_dir = DATA_DIR / "les"


# region Dataset

@DATASETS.register(name="flare7k++")
class Flare7KPP(dataset.UnlabeledImageDataset):
    """Flare7K++-Real dataset consists of 100 flare/clear image pairs.
    
    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    tasks  = [Task.LES]
    splits = [Split.TRAIN]
    has_test_annotations = False
    
    def __init__(
        self,
        root           : core.Path = default_root_dir,
        split          : Split              = Split.TRAIN,
        image_size     : int                = 256,
        classlabels    : ClassLabels | None = None,
        use_reflective : bool               = False,
        transform      : A.Compose   | None = None,
        transform_flare: A.Compose   | None = None,
        to_tensor      : bool               = False,
        cache_data     : bool               = False,
        verbose        : bool               = True,
        *args, **kwargs
    ):
        self.scattering_flare: list[anno.ImageAnnotation] = []
        self.reflective_flare: list[anno.ImageAnnotation] = []
        super().__init__(
            root        = root,
            split       = split,
            image_size  = image_size,
            classlabels = classlabels,
            transform   = transform,
            to_tensor   = to_tensor,
            cache_data  = cache_data,
            verbose     = verbose,
            *args, **kwargs
        )
        self._transform_flare = transform_flare
        self.use_reflective   = use_reflective
    
    def __getitem__(self, index: int) -> tuple[
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray | None,
        dict | None
    ]:
        image = self.images[index].data
        meta  = self.images[index].meta
        
        gamma                = np.random.uniform(1.8, 2.2)
        adjust_gamma         = transform.RandomGammaCorrection(gamma)
        adjust_gamma_reverse = transform.RandomGammaCorrection(1 / gamma)
        
        if self.transform is not None:
            transformed = self.transform(image=image)
            image       = transformed["image"]
        image     = core.to_image_tensor(input=image, keepdim=False, normalize=True)
        image     = adjust_gamma(image)
        sigma_chi = 0.01 * np.random.chisquare(df=1)
        image     = torch.distributions.Normal(image, sigma_chi).sample()
        gain      = np.random.uniform(0.5, 1.2)
        image     = gain * image
        image     = torch.clamp(image, min=0, max=1)
        
        flare = random.choice(self.scattering_flare)
        flare = core.to_image_tensor(input=flare, keepdim=False, normalize=True)
        flare = adjust_gamma(flare)
        if self.use_reflective:
            reflective_flare = random.choice(self.reflective_flare)
            reflective_flare = core.to_image_tensor(input=reflective_flare, keepdim=False, normalize=True)
            reflective_flare = adjust_gamma(reflective_flare)
            flare 	         = torch.clamp(flare + reflective_flare, min=0, max=1)
        flare = self._remove_background(flare)
        
        if self._transform_flare is not None:
            flare = core.to_image_nparray(flare, keepdim=False, denormalize=True)
            flare = self._transform_flare(flare)
            flare = core.to_image_tensor(input=flare, keepdim=False, normalize=True)
        
        # Change color
        color_jitter = TT.ColorJitter(brightness=(0.8, 3), hue=0.0)
        flare        = color_jitter(flare)
        
        # Flare blur
        blur_transform  = TT.GaussianBlur(21, sigma=(0.1, 3.0))
        flare_dc_offset = np.random.uniform(-0.02, 0.02)
        flare = blur_transform(flare)
        flare = flare + flare_dc_offset
        flare = torch.clamp(flare, min=0, max=1)
        
        return adjust_gamma_reverse(flare), adjust_gamma_reverse(image), meta
    
    def get_data(self):
        patterns = [
            self.root / "flare7k++" / self.split_str / "lq"
        ]
        self.images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    list(pattern.rglob("*")),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self.images.append(image)
        
        self._get_reflective_flare()
        self._get_scattering_flare()
        
    def _get_reflective_flare(self):
        patterns = [
            self.root / "flare7k++" / self.split_str / "pattern" / "reflective_flare"
        ]
        self.reflective_flare: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    list(pattern.rglob("*")),
                    description=f"Listing {self.__class__.__name__} {self.split_str} reflective flare"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self.reflective_flare.append(image)
    
    def _get_scattering_flare(self):
        patterns = [
            self.root / "flare7k++" / self.split_str / "pattern" / "scattering_flare" / "compound_flare"
        ]
        self.scattering_flare: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    list(pattern.rglob("*")),
                    description=f"Listing {self.__class__.__name__} {self.split_str} reflective flare"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self.reflective_flare.append(image)
    
    def cache_data(self, path: core.Path):
        cache = {
            "images"    : self.images,
            "reflective": self.reflective_flare,
            "scattering": self.scattering_flare,
        }
        torch.save(cache, str(path))
    
    def load_cache(self, path: core.Path):
        cache                 = torch.load(path)
        self.images           = cache["images"]
        self.reflective_flare = cache["reflective"]
        self.scattering_flare = cache["scattering"]
    
    def _remove_background(self, image: np.ndarray) -> np.ndarray:
        # The input of the image is PIL.Image form with [H, W, C]
        image   = np.float32(np.array(image))
        rgb_max = np.max(image, (0, 1))
        rgb_min = np.min(image, (0, 1))
        image   = (image - rgb_min) * rgb_max / (rgb_max - rgb_min + 1e-7)
        return image
    

@DATASETS.register(name="flare7k++_real")
class Flare7KPPReal(dataset.ImageEnhancementDataset):
    """Flare7K++-Real dataset consists of 100 flare/clear image pairs.
    
    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    tasks  = [Task.LES]
    splits = [Split.TEST]
    has_test_annotations = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "flare7k++_real" / self.split_str / "lq"
        ]
        self.images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self.images.append(image)

    def get_annotations(self):
        self.annotations: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path = img.path.replace("/lq/", "/hq/")
                ann  = anno.ImageAnnotation(path=path.image_file())
                self.annotations.append(ann)


@DATASETS.register(name="flare7k++_syn")
class Flare7KPPSyn(dataset.ImageEnhancementDataset):
    """Flare7K++-Syn dataset consists of 100 flare/clear image pairs.

    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    tasks  = [Task.LES]
    splits = [Split.TEST]
    has_test_annotations = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "flare7k++_syn" / self.split_str / "lq"
        ]
        self.images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self.images.append(image)

    def get_annotations(self):
        self.annotations: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path = img.path.replace("/lq/", "/hq/")
                ann  = anno.ImageAnnotation(path=path.image_file())
                self.annotations.append(ann)


@DATASETS.register(name="flarereal800")
class FlareReal800(dataset.ImageEnhancementDataset):
    """FlareReal800 dataset consists of 800 flare/clear image pairs.
    
    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    tasks  = [Task.LES]
    splits = [Split.TRAIN, Split.VAL]
    has_test_annotations = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "flarereal800" / self.split_str / "lq"
        ]
        self.images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self.images.append(image)
    
    def get_annotations(self):
        self.annotations: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path = img.path.replace("/lq/", "/hq/")
                ann  = anno.ImageAnnotation(path=path.image_file())
                self.annotations.append(ann)


@DATASETS.register(name="ledlight")
class LEDLight(dataset.ImageEnhancementDataset):
    """LEDLight dataset consists of 100 flare/clear image pairs.

    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    tasks  = [Task.LES]
    splits = [Split.TEST]
    has_test_annotations = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "ledlight" / self.split_str / "lq"
        ]
        self.images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self.images.append(image)

    def get_annotations(self):
        self.annotations: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path = img.path.replace("/lq/", "/hq/")
                ann  = anno.ImageAnnotation(path=path.image_file())
                self.annotations.append(ann)


@DATASETS.register(name="lighteffect")
class LightEffect(dataset.UnlabeledImageDataset):
    """LightEffect dataset consists 961 flare images.

    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    tasks  = [Task.LES]
    splits = [Split.TRAIN]
    has_test_annotations = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            # self.root / self.split / "light-effect" / "clear",
            self.root / "lighteffect" / self.split_str / "lq",
        ]
        self.images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self.images.append(image)

# endregion


# region Datamodule

@DATAMODULES.register(name="flare7k++_real")
class Flare7KPPRealDataModule(datamodule.DataModule):
    """Flare7K++-Real datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.LES]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "train"]:
            self.train = Flare7KPPReal(split=Split.TEST, **self.dataset_kwargs)
            self.val   = Flare7KPPReal(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Flare7KPPReal(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="flare7k++_syn")
class Flare7KPPSynDataModule(datamodule.DataModule):
    """Flare7K++-Syn datamodule.

    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.LES]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Flare7KPPSyn(split=Split.TEST, **self.dataset_kwargs)
            self.val   = Flare7KPPSyn(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Flare7KPPSyn(split=Split.TEST, **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        pass


@DATAMODULES.register(name="flarereal800")
class FlareReal800DataModule(datamodule.DataModule):
    """FlareReal800 datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.LES]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = FlareReal800(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = FlareReal800(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FlareReal800(split=Split.VAL,   **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="ledlight")
class LEDLightDataModule(datamodule.DataModule):
    """LEDLight datamodule.

    See Also: :class:`base.DataModule`.
    """

    tasks = [Task.LES]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = LEDLight(split=Split.TEST, **self.dataset_kwargs)
            self.val   = LEDLight(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = LEDLight(split=Split.TEST, **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        pass


@DATAMODULES.register(name="lighteffect")
class LightEffectDataModule(datamodule.DataModule):
    """LightEffect datamodule.

    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.LES]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = LightEffect(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LightEffect(split=Split.TRAIN, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = LightEffect(split=Split.TRAIN, **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        pass

# endregion
