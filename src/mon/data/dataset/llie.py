#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements low-light image enhancement (llie) datasets and
datamodules.
"""

from __future__ import annotations

__all__ = [
    "DICM",
    "DICMDataModule",
    "DarkFace",
    "DarkFaceDataModule",
    "ExDark",
    "ExDarkDataModule",
    "FiveKC",
    "FiveKCDataModule",
    "FiveKE",
    "FiveKEDataModule",
    "Fusion",
    "LIME",
    "LIMEDataModule",
    "LOLBlur",
    "LOLBlurDataModule",
    "LOLV1",
    "LOLV1DataModule",
    "LOLV2Real",
    "LOLV2RealDataModule",
    "LOLV2Synthetic",
    "LOLV2SyntheticDataModule",
    "MEF",
    "MEFDataModule",
    "NPE",
    "NPEDataModule",
    "SICEGrad",
    "SICEGradDataModule",
    "SICEMix",
    "SICEMixDataModule",
    "SICEMixV2",
    "SICEMixV2DataModule",
    "ULOL",
    "ULOLMixDataModule",
    "VV",
    "VVDataModule",
]

from typing import Literal

from mon import core
from mon.data.datastruct import annotation as anno, datamodule, dataset
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console           = core.console
_default_root_dir = DATA_DIR / "llie"


# region Dataset

@DATASETS.register(name="darkface")
class DarkFace(dataset.UnlabeledImageDataset):
    """DarkFace dataset consists of 6490 low-light images.
    
    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    _tasks  = [Task.LLIE]
    _splits = [Split.TEST]
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        
    def _get_images(self):
        patterns = [
            self.root / "darkface" / self.split_str / "lq"
        ]

        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)


@DATASETS.register(name="dicm")
class DICM(dataset.UnlabeledImageDataset):
    """DICM dataset consists of 64 low-light images.
    
    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    _tasks  = [Task.LLIE]
    _splits = [Split.TEST]
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        
    def _get_images(self):
        patterns = [
            self.root / "dicm" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)


@DATASETS.register(name="exdark")
class ExDark(dataset.UnlabeledImageDataset):
    """ExDark dataset consists of 7363 low-light images.
    
    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    _tasks  = [Task.LLIE]
    _splits = [Split.TEST]
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        
    def _get_images(self):
        patterns = [
            self.root / "exdark" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)


@DATASETS.register(name="fivek_c")
class FiveKC(dataset.UnlabeledImageDataset):
    """MIT Adobe FiveK dataset with Expert C ground-truth. It consists of 5,000
    low/high image pairs.
    
    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    _tasks  = [Task.LLIE]
    _splits = [Split.TRAIN]
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        
    def _get_images(self):
        patterns = [
            self.root / "fivek_c" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)
    
    def _get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="fivek_e")
class FiveKE(dataset.ImageEnhancementDataset):
    """MIT Adobe FiveK dataset with Expert E ground-truth. It consists of 5,000
    low/high image pairs.
    
    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    _tasks  = [Task.LLIE]
    _splits = [Split.TRAIN]
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        
    def _get_images(self):
        patterns = [
            self.root / "fivek_e" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)
    
    def get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="fusion")
class Fusion(dataset.UnlabeledImageDataset):
    """Fusion dataset consists of 64 low-light images.
    
    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    _tasks  = [Task.LLIE]
    _splits = [Split.TEST]
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "fusion" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)


@DATASETS.register(name="lime")
class LIME(dataset.UnlabeledImageDataset):
    """LIME dataset consists of 10 low-light images.
    
    See Also: :class:`base.UnlabeledImageDataset`.
    """

    _tasks  = [Task.LLIE]
    _splits = [Split.TEST]
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "lime" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)


@DATASETS.register(name="lol_blur")
class LOLBlur(dataset.ImageEnhancementDataset):
    """LOL-Blur dataset consists of low-light + blurred and normal-light + sharp
    image pairs.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.LLIE]
    _splits         = [Split.TRAIN, Split.VAL, Split.TEST]
    _has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "lol_blur" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)
    
    def _get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="lol_v1")
class LOLV1(dataset.ImageEnhancementDataset):
    """LOL-v1 dataset consists of 500 low-light and normal-light image pairs.
    They are divided into 485 training pairs and 15 testing pairs. The low-light
    images contain noise produced during the photo capture process. Most of the
    images are indoor scenes. All the images have a resolution of 400×600.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.LLIE]
    _splits         = [Split.TRAIN, Split.TEST]
    _has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "lol_v1" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)
    
    def _get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="lol_v2_real")
class LOLV2Real(dataset.ImageEnhancementDataset):
    """LOL-v2 Real (VE-LOL) dataset consists of 500 low-light and normal-light
    image pairs. They are divided into 400 training pairs and 100 testing pairs.
    The low-light images contain noise produced during the photo capture
    process. Most of the images are indoor scenes. All the images have a
    resolution of 400×600.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.LLIE]
    _splits         = [Split.TRAIN, Split.TEST]
    _has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "lol_v2_real" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)
    
    def _get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="lol_v2_synthetic")
class LOLV2Synthetic(dataset.ImageEnhancementDataset):
    """LOL-v2 Synthetic (VE-LOL-Syn) dataset consists of 1000 low-light and
    normal-light image pairs. They are divided into 900 training pairs and 100
    testing pairs. The low-light images contain noise produced during the photo
    capture process. Most of the images are indoor scenes. All the images have a
    resolution of 400×600.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.LLIE]
    _splits         = [Split.TRAIN, Split.TEST]
    _has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "lol_v2_synthetic" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)
    
    def _get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="mef")
class MEF(dataset.UnlabeledImageDataset):
    """MEF dataset consists 17 low-light images.

    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    _tasks  = [Task.LLIE]
    _splits = [Split.TEST]
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "mef" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)
      
                        
@DATASETS.register(name="npe")
class NPE(dataset.UnlabeledImageDataset):
    """NPE dataset consists 85 low-light images.
    
    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    _tasks  = [Task.LLIE]
    _splits = [Split.TEST]
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "npe" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)


@DATASETS.register(name="sice_grad")
class SICEGrad(dataset.UnlabeledImageDataset):
    """SICE-Grad dataset.
    
    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    _tasks          = [Task.LLIE]
    _splits         = [Split.TRAIN]
    _has_test_label = False
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "sice_grad" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)
                        
                        
@DATASETS.register(name="sice_mix")
class SICEMix(dataset.UnlabeledImageDataset):
    """Custom SICE dataset for training :class:`mon.vision.enhance.llie.zerodce.ZeroDCE`
    model.
    
    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    _tasks          = [Task.LLIE]
    _splits         = [Split.TRAIN]
    _has_test_label = False
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "sice_mix" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)


@DATASETS.register(name="sice_mix_v2")
class SICEMixV2(dataset.UnlabeledImageDataset):
    """SICE-MixV2 dataset.
    
    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    _tasks          = [Task.LLIE]
    _splits         = [Split.TRAIN]
    _has_test_label = False
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "sice_mix_v2" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)
                        

@DATASETS.register(name="ulol")
class ULOL(dataset.UnlabeledImageDataset):
    """Custom ULOL (Unsupervised LOw-Light) dataset for training.
    
    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    _tasks          = [Task.LLIE]
    _splits         = [Split.TRAIN]
    _has_test_label = False
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "sice_mix"         / self.split_str / "lq",
            self.root / "sice_mix_v2"      / self.split_str / "lq",
            self.root / "lol_v1"           / self.split_str / "lq",
            self.root / "lol_v1"           / self.split_str / "hq",
            self.root / "lol_v2_real"      / self.split_str / "lq",
            self.root / "lol_v2_real"      / self.split_str / "hq",
            self.root / "lol_v2_synthetic" / self.split_str / "lq",
            self.root / "lol_v2_synthetic" / self.split_str / "hq",
            self.root / "dicm"             / "test"         / "lq",
            self.root / "fusion"           / "test"         / "lq",
            self.root / "lime"             / "test"         / "lq",
            self.root / "mef"              / "test"         / "lq",
            self.root / "npe"              / "test"         / "lq",
            self.root / "vv"               / "test"         / "lq",
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)
                        

@DATASETS.register(name="vv")
class VV(dataset.UnlabeledImageDataset):
    """VV dataset consists of 24 low-light images.
    
    See Also: :class:`base.UnlabeledImageDataset`.
    """
    
    _tasks  = [Task.LLIE]
    _splits = [Split.TEST]
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "vv" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)

# endregion


# region Datamodule

@DATAMODULES.register(name="darkface")
class DarkFaceDataModule(datamodule.DataModule):
    """DarkFace datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "training"]:
            self.train = DarkFace(split=Split.TEST, **self.dataset_kwargs)
            self.val   = DarkFace(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "testing"]:
            self.test  = DarkFace(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="dicm")
class DICMDataModule(datamodule.DataModule):
    """DICM datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "training"]:
            self.train = DICM(split=Split.TEST, **self.dataset_kwargs)
            self.val   = DICM(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "testing"]:
            self.test  = DICM(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="exdark")
class ExDarkDataModule(datamodule.DataModule):
    """ExDark datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "training"]:
            self.train = ExDark(split=Split.TEST, **self.dataset_kwargs)
            self.val   = ExDark(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "testing"]:
            self.test  = ExDark(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="fivek_c")
class FiveKCDataModule(datamodule.DataModule):
    """MIT Adobe FiveK datamodule with Expert C ground-truth.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "training"]:
            self.train = FiveKC(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LOLV1(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "testing"]:
            self.test  = LOLV1(split=Split.TEST,  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="fivek_e")
class FiveKEDataModule(datamodule.DataModule):
    """MIT Adobe FiveK datamodule with Expert E ground-truth.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "training"]:
            self.train = FiveKE(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LOLV1(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "testing"]:
            self.test  = LOLV1(split=Split.TEST,  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass
    

@DATAMODULES.register(name="fusion")
class FusionDataModule(datamodule.DataModule):
    """Fusion datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "training"]:
            self.train = Fusion(split=Split.TEST, **self.dataset_kwargs)
            self.val   = Fusion(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "testing"]:
            self.test  = Fusion(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="lime")
class LIMEDataModule(datamodule.DataModule):
    """LIME datamodule.
     
     See Also: :class:`base.DataModule`.
     """
    
    _tasks = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "training"]:
            self.train = LIME(split=Split.TEST, **self.dataset_kwargs)
            self.val   = LIME(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "testing"]:
            self.test  = LIME(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="lol_blur")
class LOLBlurDataModule(datamodule.DataModule):
    """LOL-Blur datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "training"]:
            self.train = LOLBlur(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LOLBlur(split=Split.VAL,  **self.dataset_kwargs)
        if stage in [None, "testing"]:
            self.test  = LOLBlur(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass
    

@DATAMODULES.register(name="lol_v1")
class LOLV1DataModule(datamodule.DataModule):
    """LOLV1 datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "training"]:
            self.train = LOLV1(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LOLV1(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "testing"]:
            self.test  = LOLV1(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="lol_v2_real")
class LOLV2RealDataModule(datamodule.DataModule):
    """LOLV2Real datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "training"]:
            self.train = LOLV2Real(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LOLV2Real(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "testing"]:
            self.test  = LOLV2Real(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="lol_v2_synthetic")
class LOLV2SyntheticDataModule(datamodule.DataModule):
    """LOL-v2 Synthetic datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "training"]:
            self.train = LOLV2Synthetic(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LOLV2Synthetic(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "testing"]:
            self.test  = LOLV2Synthetic(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="mef")
class MEFDataModule(datamodule.DataModule):
    """MEF datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "training"]:
            self.train = MEF(split=Split.TEST, **self.dataset_kwargs)
            self.val   = MEF(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "testing"]:
            self.test  = MEF(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="npe")
class NPEDataModule(datamodule.DataModule):
    """NPE datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Use this method to do things on every device:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Prepare train/val/test/predict splits.
            - Apply transformations.
            - Define :attr:`collate_fn` for your custom dataset.

        Args:
            stage: The model phase. One of:
                - "training" : prepares :attr:'train' and :attr:'val'.
                - "testing"  : prepares :attr:'test'.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Default: ``None``.
        """
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "training"]:
            self.train = NPE(split=Split.TEST, **self.dataset_kwargs)
            self.val   = NPE(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "testing"]:
            self.test  = NPE(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="sice_grad")
class SICEGradDataModule(datamodule.DataModule):
    """SICE-Grad datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "training"]:
            self.train = SICEGrad(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LOLV1(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "testing"]:
            self.test  = LOLV1(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="sice_mix")
class SICEMixDataModule(datamodule.DataModule):
    """SICE-Mix datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "training"]:
            self.train = SICEMix(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LOLV1(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "testing"]:
            self.test  = LOLV1(split=Split.TEST, **self.dataset_kwargs)
            
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="sice_mix_v2")
class SICEMixV2DataModule(datamodule.DataModule):
    """SICE-MixV2 datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "training"]:
            self.train = SICEMixV2(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LOLV1(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "testing"]:
            self.test  = LOLV1(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass
    

@DATAMODULES.register(name="ulol")
class ULOLMixDataModule(datamodule.DataModule):
    """Custom ULOL (Unsupervised LOw-Light) datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "training"]:
            self.train = ULOL(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LOLV1(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "testing"]:
            self.test  = LOLV1(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="vv")
class VVDataModule(datamodule.DataModule):
    """VV datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "training"]:
            self.train = VV(split=Split.TEST, **self.dataset_kwargs)
            self.val   = VV(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "testing"]:
            self.test  = VV(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass

# endregion
