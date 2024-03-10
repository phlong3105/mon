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
from mon.data import base
from mon.data.augment import transform
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console           = core.console
ClassLabels       = base.ClassLabels
_default_root_dir = DATA_DIR / "les"


# region Dataset

@DATASETS.register(name="flare7k++")
class Flare7KPP(base.UnlabeledImageDataset):
    """Flare7K++-Real dataset consists of 100 flare/clear image pairs.
    
    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    _tasks          = [Task.LES]
    _splits         = [Split.TRAIN]
    _has_test_label = False
    
    def __init__(
        self,
        root           : core.Path = _default_root_dir,
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
        self.scattering_flare: list[base.ImageLabel] = []
        self.reflective_flare: list[base.ImageLabel] = []
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
        image = self._images[index].data
        meta  = self._images[index].meta
        
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
    
    def _get_images(self):
        patterns = [
            self.root / "flare7k++" / self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    list(pattern.rglob("*")),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)
        
        self._get_reflective_flare()
        self._get_scattering_flare()
        
    def _get_reflective_flare(self):
        patterns = [
            self.root / "flare7k++" / self.split / "pattern" / "reflective_flare"
        ]
        self.reflective_flare: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    list(pattern.rglob("*")),
                    description=f"Listing {self.__class__.__name__} {self.split} reflective flare"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self.reflective_flare.append(image)
    
    def _get_scattering_flare(self):
        patterns = [
            self.root / "flare7k++" / self.split / "pattern" / "scattering_flare" / "compound_flare"
        ]
        self.scattering_flare: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    list(pattern.rglob("*")),
                    description=f"Listing {self.__class__.__name__} {self.split} reflective flare"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self.reflective_flare.append(image)
    
    def cache_data(self, path: core.Path):
        cache = {
            "images"    : self._images,
            "reflective": self.reflective_flare,
            "scattering": self.scattering_flare,
        }
        torch.save(cache, str(path))
    
    def load_cache(self, path: core.Path):
        cache                 = torch.load(path)
        self._images           = cache["images"]
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
class Flare7KPPReal(base.ImageEnhancementDataset):
    """Flare7K++-Real dataset consists of 100 flare/clear image pairs.
    
    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    _tasks          = [Task.LES]
    _splits         = [Split.TEST]
    _has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "flare7k++_real" / self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = base.ImageLabel(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="flare7k++_syn")
class Flare7KPPSyn(base.ImageEnhancementDataset):
    """Flare7K++-Syn dataset consists of 100 flare/clear image pairs.

    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    _tasks          = [Task.LES]
    _splits         = [Split.TEST]
    _has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "flare7k++_syn" / self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = base.ImageLabel(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="flarereal800")
class FlareReal800(base.ImageEnhancementDataset):
    """FlareReal800 dataset consists of 800 flare/clear image pairs.
    
    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    _tasks          = [Task.LES]
    _splits         = [Split.TRAIN, Split.VAL]
    _has_test_label = False
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "flarereal800" / self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)
    
    def _get_labels(self):
        self._labels: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = base.ImageLabel(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="ledlight")
class LEDLight(base.ImageEnhancementDataset):
    """LEDLight dataset consists of 100 flare/clear image pairs.

    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    _tasks          = [Task.LES]
    _splits         = [Split.TEST]
    _has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "ledlight" / self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = base.ImageLabel(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="lighteffect")
class LightEffect(base.UnlabeledImageDataset):
    """LightEffect dataset consists 961 flare images.

    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    _tasks          = [Task.LES]
    _splits         = [Split.TRAIN]
    _has_test_label = False
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            # self.root / self.split / "light-effect" / "clear",
            self.root / "lighteffect" / self.split / "lq",
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)

# endregion


# region Datamodule

@DATAMODULES.register(name="flare7k++_real")
class Flare7KPPRealDataModule(base.DataModule):
    """Flare7K++-Real datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LES]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if phase in [None, "training"]:
            self.train = Flare7KPPReal(split=Split.TEST, **self.dataset_kwargs)
            self.val   = Flare7KPPReal(split=Split.TEST, **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test  = Flare7KPPReal(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="flare7k++_syn")
class Flare7KPPSynDataModule(base.DataModule):
    """Flare7K++-Syn datamodule.

    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LES]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if phase in [None, "training"]:
            self.train = Flare7KPPSyn(split=Split.TEST, **self.dataset_kwargs)
            self.val   = Flare7KPPSyn(split=Split.TEST, **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test  = Flare7KPPSyn(split=Split.TEST, **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        pass


@DATAMODULES.register(name="flarereal800")
class FlareReal800DataModule(base.DataModule):
    """FlareReal800 datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LES]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if phase in [None, "training"]:
            self.train = FlareReal800(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = FlareReal800(split=Split.VAL,   **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test  = FlareReal800(split=Split.VAL,   **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="ledlight")
class LEDLightDataModule(base.DataModule):
    """LEDLight datamodule.

    See Also: :class:`base.DataModule`.
    """

    _tasks = [Task.LES]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if phase in [None, "training"]:
            self.train = LEDLight(split=Split.TEST, **self.dataset_kwargs)
            self.val   = LEDLight(split=Split.TEST, **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test  = LEDLight(split=Split.TEST, **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        pass


@DATAMODULES.register(name="lighteffect")
class LightEffectDataModule(base.DataModule):
    """LightEffect datamodule.

    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LES]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if phase in [None, "training"]:
            self.train = LightEffect(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LightEffect(split=Split.TRAIN, **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test  = LightEffect(split=Split.TRAIN, **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        pass

# endregion
