#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base model class for building machine learning
models.
"""

from __future__ import annotations

__all__ = [
    "Model",
    "attempt_load",
    "extract_weight_from_checkpoint",
    "get_epoch",
    "get_global_step",
    "get_latest_checkpoint",
    "get_model_fullname",
    "intersect_state_dicts",
    "is_parallel",
    "load_state_dict",
    "load_state_dict_from_path",
    "load_weights",
    "match_state_dicts",
    "sparsity",
    "strip_optimizer",
]

import os
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse  # noqa: F401

import humps
import lightning.pytorch.utilities.types
import torch.hub
from thop.profile import *
from torch import nn
from torch.nn import parallel

from mon import core
from mon.globals import (
    LOSSES, LR_SCHEDULERS, METRICS, ModelPhase, MODELS, OPTIMIZERS, ZOO_DIR,
)
from mon.nn import data as mdata, loss as mloss, metric as mmetric, parsing

StepOutput    = lightning.pytorch.utilities.types.STEP_OUTPUT
EpochOutput   = Any  # lightning.pytorch.utilities.types.EPOCH_OUTPUT
console       = core.console
error_console = core.error_console


# region Checkpoint

def extract_weight_from_checkpoint(
    ckpt       : core.Path,
    weight_file: core.Path | None = None,
):
    """Extract and save weights from the checkpoint :attr:`ckpt`.
    
    Args:
        ckpt: A checkpoint file.
        weight_file: A path to save the extracting weights. Default: ``None``,
            which saves the weights at the same location as the :param:`ckpt`
            file.
    """
    ckpt = core.Path(ckpt)
    if not ckpt.is_ckpt_file():
        raise ValueError(
            f"ckpt must be a valid path to .ckpt file, but got {ckpt}."
        )
    
    state_dict = load_state_dict_from_path(ckpt)
    if state_dict is None:
        raise ValueError()
    
    if weight_file is None:
        weight_file = ckpt.parent / f"{ckpt.stem}.pth"
    else:
        weight_file = core.Path(weight_file)
    weight_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, str(weight_file))


def get_epoch(ckpt: core.Path | None) -> int:
    """Get an epoch value stored in a checkpoint file.

    Args:
        ckpt: A checkpoint file path.
    """
    if ckpt is None:
        return 0
    
    epoch = 0
    ckpt  = core.Path(ckpt)
    if ckpt.is_torch_file():
        ckpt  = torch.load(ckpt)
        epoch = ckpt.get("epoch", 0)
    return epoch


def get_global_step(ckpt: core.Path | None) -> int:
    """Get a global step stored in a checkpoint file.

    Args:
        ckpt: A checkpoint file path.
    """
    if ckpt is None:
        return 0
    
    global_step = 0
    ckpt        = core.Path(ckpt)
    if ckpt.is_torch_file():
        ckpt        = torch.load(ckpt)
        global_step = ckpt.get("global_step", 0)
    return global_step


def get_latest_checkpoint(dirpath: core.Path) -> str | None:
    """Get the latest checkpoint (last saved) file path in a directory.

    Args:
        dirpath: The directory that contains the checkpoints.
    """
    dirpath = core.Path(dirpath)
    ckpt    = dirpath.latest_file()
    if ckpt is None:
        error_console.log(f"[red]Cannot find checkpoint file {dirpath}.")
    return ckpt

# endregion


# region Weight/State Dict

def intersect_state_dicts(x: dict, y: dict, exclude: list = []) -> dict:
    """Find the intersection between two state :class:`dict`.
    
    Args:
        x: The first state :class:`dict`.
        y: The second state :class:`dict`.
        exclude: A :class:`list` of excluding keys.
    
    Return:
        A :class:`dict` that contains only the keys that are in both :param:`x`
        and :param:`y`, and whose values have the same shape.
    """
    return {
        k: v for k, v in x.items()
        if k in y and not any(x in k for x in exclude) and v.shape == y[k].shape
    }


def match_state_dicts(
    model_dict     : dict,
    pretrained_dict: dict,
    exclude        : list = []
) -> dict:
    """Filter out unmatched keys between the model's :attr:`state_dict`, and the
    weights' :attr:`state_dict`. Omitting :param:`exclude` keys.

    Args:
        model_dict: Model :attr:`state_dict`.
        pretrained_dict: Pretrained :attr:`state_dict`.
        exclude: A :class:`list` of excluded keys. Default: ``()``.
        
    Return:
        Filtered model's :attr:`state_dict`.
    """
    # 1. Filter out unnecessary keys
    intersect_dict = intersect_state_dicts(
        x       = pretrained_dict,
        y       = model_dict,
        exclude = exclude
    )
    # 2. Overwrite entries in the existing state dict
    model_dict.update(intersect_dict)
    return model_dict


def strip_optimizer(weight_file: str, new_file: str | None = None):
    """Strip the optimizer from a saved weights file to finalize the training
    process.
    
    Args:
        weight_file: A PyTorch saved weights file path.
        new_file: A file path to save the stripped weights. If :param:`new_file`
            is given, save the weights as a new file. Otherwise, overwrite the
            :param:`weight_file`.
    """
    if not core.Path(weight_file).is_weights_file():
        raise ValueError(
            f"``weight_file`` must be a valid path to a weights file, but got "
            f"{weight_file}."
        )
    
    x = torch.load(weight_file, map_location = torch.device("cpu"))
    x["optimizer"]        = None
    x["training_results"] = None
    x["epoch"]            = -1
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
    
    torch.save(x, new_file or weight_file)
    mb = os.path.getsize(new_file or weight_file) / 1E6  # filesize
    console.log(
        "Optimizer stripped from %s,%s %.1fMB"
        % (weight_file, (" saved as %s," % new_file) if new_file else "", mb)
    )

# endregion


# region Model Loading

def attempt_load(
    model      : nn.Module | Model,
    weights    : dict | core.Path,
    name       : str              | None = None,
    config     : dict | core.Path | None = None,
    fullname   : str              | None = None,
    num_classes: int              | None = None,
    phase      : str                     = "inference",
    strict     : bool                    = True,
    file_name  : str              | None = None,
    model_dir  : core.Path        | None = ZOO_DIR,
    debugging  : bool                    = True
):
    """Try to create a model from the given configuration and load pretrained
    weights.
    
    Args:
        model: A PyTorch :class:`nn.Module` or a :class:`Model`.
        weights: Weights :class:`dict`, or a checkpoint file path.
        name: A name of the model.
        config: A configuration :class:`dict`, or a ``.yaml`` file path
            containing the building configuration of the model.
        fullname: An optional fullname of the model, in other words,
            a model's base name + its variant + a training dataset name.
        num_classes: The number of classes, in other words, the final layer's
            output channels.
        phase: The model running phase.
        strict: Default: ``True``.
        file_name: Name for the downloaded file. Filename from :param:`path`
        model_dir: Directory in which to save the object. Default to
            :attr:`ZOO_DIR`.
        debugging: If ``True``, stop and raise errors. Otherwise, just return
            the current model.
        
    Return:
        A :class:`Model` object.
    """
    # Get model's configuration
    if config is not None:
        config = config.load_config(config=config)
    
    # Get model
    if model is None:
        if name is None and config is None:
            if debugging:
                raise RuntimeError
            else:
                return model
        else:
            model: Model = MODELS.build(
                name        = name,
                fullname    = fullname,
                config      = config,
                num_classes = num_classes,
                phase       = phase,
            )
    
    # If model is None, stop
    if model is None:
        if debugging:
            raise RuntimeError
        else:
            return model
    # Load weights for :class:`Model` from a checkpoint
    elif isinstance(model, Model) and core.Path(path=weights).is_ckpt_file():
        model = model.load_from_checkpoint(
            checkpoint_path = weights,
            name            = name,
            cfg             = config,
            num_classes     = num_classes,
            phase           = phase,
        )
    # All other cases
    else:
        if isinstance(weights, str | core.Path):
            state_dict = load_state_dict_from_path(
                path         = weights,
                model_dir    = model_dir,
                map_location = None,
                progress     = True,
                check_hash   = True,
                file_name    = file_name or fullname,
            )
        elif isinstance(weights, dict):
            state_dict = weights
        else:
            raise RuntimeError
        
        if isinstance(model, Model):
            module_dict = model.model.state_dict()
        else:
            module_dict = model.state_dict()
        
        module_dict = match_state_dicts(
            model_dict      = module_dict,
            pretrained_dict = state_dict,
        )
        
        if isinstance(model, Model):
            model.model.load_state_dict(state_dict=module_dict, strict=strict)
        else:
            model.load_state_dict(state_dict=module_dict, strict=strict)
    
    if fullname is not None:
        model.fullname = fullname
    return model


def load_state_dict_from_path(
    path        : core.Path,
    model_dir   : core.Path | None = None,
    map_location: str       | None = None,
    progress    : bool             = True,
    check_hash  : bool             = False,
    file_name   : str       | None = None,
    **_
) -> dict | None:
    """Load state dict at the given URL. If downloaded file is a ``.zip`` file,
    it will be automatically decompressed. If the object is already present in
    :param:`model_dir`, it is deserialized and returned.
    
    Args:
        path: The weights or checkpoints file to load. If it is a URL, it will
            be downloaded.
        model_dir: Directory in which to save the object. Default to ``None``.
        map_location: A function, or a :class:`dict` specifying how to remap
            storage locations (see :meth:`torch.load`). Default: ``None``.
        progress: Whether to display a progress bar to stderr. Default: ``True``.
        check_hash: If ``True``, the :param:`file_name` part of the URL should
            follow the naming convention `file_name-<sha256>.ext` where
            `<sha256>` is the first eight or more digits of the SHA256 hash of
            the contents of the file. Hash is used to ensure unique names and to
            verify the contents of the file. Default: ``False``.
        file_name: Name for the downloaded file. Filename from :param:`path`
            will be used if not set.
    """
    if path is None:
        raise ValueError()
    if model_dir:
        model_dir = core.Path(model_dir)
    
    path = core.Path(path)
    if not path.is_torch_file() \
        and (model_dir is None or not model_dir.is_dir()):
        raise ValueError(f"'model_dir' must be defined. But got: {model_dir}.")
    
    save_weight = core.Path()
    if file_name:
        save_weight = model_dir / file_name
    
    state_dict = None
    if save_weight.is_torch_file():
        state_dict = torch.load(str(save_weight), map_location=map_location)
    elif path.is_torch_file():
        state_dict = torch.load(str(path), map_location=map_location)
    elif path.is_url():
        state_dict = torch.hub.load_state_dict_from_url(
            url          = str(path),
            model_dir    = str(model_dir),
            map_location = map_location,
            progress     = progress,
            check_hash   = check_hash,
            file_name    = file_name
        )
    
    if state_dict and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    if state_dict and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict


def load_state_dict(
    model    : nn.Module,
    weights  : dict | str | core.Path,
    overwrite: bool = False,
) -> dict:
    """Load state dict from :param:`weights` file."""
    path = None
    map  = {}
    
    if isinstance(weights, dict):
        path = weights.get("path", path)
        url  = weights.get("url",  None)
        map  = weights.get("map",  map)
        if path is not None and url is not None:
            path = core.Path(path)
            core.mkdirs(paths=[path.parent], exist_ok=True)
            if core.is_url(url):
                if path.exists() and overwrite:
                    core.delete_files(regex=path.name, path=path.parent)
                    torch.hub.download_url_to_file(str(url), str(path), None, progress=True)
                elif not path.exists():
                    torch.hub.download_url_to_file(str(url), str(path), None, progress=True)
    elif isinstance(weights, str | core.Path):
        path = weights
    else:
        return {}

    path = core.Path(path)
    if not path.is_weights_file():
        error_console.log(f"{path} is not a weights file.")

    state_dict       = torch.load(str(path), map_location=model.device)
    model_state_dict = model.state_dict()
    new_state_dict   = {}

    for k, v in state_dict.items():
        replace = False
        for k1, k2 in map.items():
            kr = k.replace(k1, k2)
            if kr in model_state_dict and not replace:
                new_state_dict[kr] = v
                replace = True
        if not replace:
            new_state_dict[k] = v
    
    # print(state_dict.keys())
    # print(model_state_dict.keys())
    # print(new_state_dict.keys())
    # print([value.shape for value in model_state_dict.values()])
    # print([value.shape for value in new_state_dict.values()])
    for k, v in new_state_dict.items():
        if k in model_state_dict:
            model_state_dict[k] = v
    
    return model_state_dict


def load_weights(
    model    : nn.Module,
    weights  : dict | str | core.Path,
    overwrite: bool = False,
) -> nn.Module:
    """Load weights to model."""
    model_state_dict = load_state_dict(model=model, weights=weights)
    model.load_state_dict(model_state_dict)
    return model

# endregion


# region Model Parsing

def is_parallel(model: nn.Module) -> bool:
    """Return ``True`` if a model is in a parallel run-mode. Otherwise, return
    ``False``.
    """
    return type(model) in (
        parallel.DataParallel,
        parallel.DistributedDataParallel
    )


def sparsity(model: nn.Module) -> float:
    """Return the global model sparsity."""
    a = 0.0
    b = 0.0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a

# endregion


# region Model

def get_model_fullname(
    name   : str,
    data   : str,
    variant: str | None = None,
) -> str:
    """Get model fullname"""
    name    = name    or ""
    variant = variant or ""
    data    = data    or ""

    if name != "" and variant != "" and name in variant:
        fullname = variant
    elif name != "" and variant != "":
        fullname = f"{name}-{variant}"
    else:
        fullname = f"{name}"
    fullname = f"{fullname}-{data}"
    return fullname


class Model(lightning.LightningModule, ABC):
    """The base class for all machine learning models.
    
    Attributes:
        zoo: A :class:`dict` containing all pretrained weights of the model.
        
    Args:
        arch: Path or name of the model architecture file that's used to build
            the model. Any of the following cases:
                - Case 01: A file name, or a path to a ``.yaml`` file. Ex: 'alexnet.yaml'.
                - Case 02: ``None``, define each layer manually.
        channels: The first layer's input channel. Default: ``3``.
        num_classes: A number of classes, which is also the last layer's output
            channels. Default: ``None`` mean it will be determined during model
            parsing.
        classlabels: A :class:`mon.nn.data.label.ClassLabels` object that
            contains all labels in the dataset. Default: ``None``.
        weights: The model's weights. Any of:
            - A state :class:`dict`.
            - A key in the :attr:`zoo`. Ex: 'yolov8x-det-coco'.
            - A path to an ``.pt``, ``.pth``, or ``.ckpt`` file.
        name: The model's name. Default: ``None`` mean it will be
            :attr:`self.__class__.__name__`. .
        variant: The model's variant, WHICH IS MAINLY USED FOR EXPERIMENTING
            WITH DIFFERENT MODEL CONFIGURATIONS. For example, if we have a model
            name as ``'yolov9-0010'``, then :param:`name` is ``'yolov9'`` and
            :param:`variant` is ``'0010'``. Default: ``None`` FOR MOST OF THE TIME.
        fullname: The model's fullname to save the checkpoint or weights. It
            should have the following format:
            {name}/{variant}-{dataset}-{suffix}. Default: ``None`` mean it will
             be the same as :param:`name`.
        root: The root directory of the model. It is used to save the model
            checkpoint during training: {root}/{project}/{fullname}.
        project: A project name. Default: ``None``.
        hparams: Model's hyperparameters. They are used to change the values of
            :param:`args`. Usually used in grid search or random search during
            training. Default: ``None``.
        phase: The model's running phase. Default: ``'training'``.
        loss: Loss function for training the model. Default: ``None``.
        metrics: A list metrics for validating and testing model. Default:
            ``None``.
        optimizers: Optimizer(s) for a training model. Default: ``None``.
        debug: Debug configs. Default: ``None``.
        verbose: Verbosity. Default: ``True``.

    Example:
        DEFINING MODEL

        Case 01: Define :param:`arch` as a full path to the ``.yaml`` file.
        RECOMMENDED FOR MODELS WITH VARIOUS CONFIGURATIONS BUT IDENTICAL
        FORWARD_PASS (ex: YOLO).

            >>> # config
            >>> #   |_ vgg16.yaml
            >>> #   |_ vgg19.yaml
            >>>
            >>> model = Model(arch="vgg16")
            >>> model = Model(arch="vgg19.yaml")
            >>> model = Model(arch="/home/workspace/.../vgg19.yaml")

        Case 02: Define :param:`arch` as ``None``. Then you have to manually
        define the model in the traditional way. RECOMMENDED FOR SINGLE-PURPOSE
        MODELS.

            >>> model = Model(arch=None)

    Example:
        LOADING WEIGHTS

        Case 01: Define a state :class:`dict`.

            >>> model = Model(
            >>>     weights={
            >>>         "layer01": [],
            >>>         "layer02": [],
            >>>         "layer03": [],
            >>>     },
            >>> )

        Case 02: Pre-define the weights file in ``zoo`` directory. Pre-define
        the metadata in :attr:`zoo`. Then define :param:`weights` as a key in
        :attr:`zoo`.

            >>> zoo = {
            >>>     "imagenet": {
            >>>         "url"        : "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
            >>>         "path"       : "vgg19-imagenet.pth",  # Locate in ``zoo`` directory
            >>>         "num_classes": 1000,
            >>>         "map": {}
            >>>     },
            >>> }
            >>>
            >>> model = Model(
            >>>     arch="vgg19",
            >>>     weights="imagenet",
            >>> )

        Case 03: Define the full path to an ``.pt``, ``.pth``, or ``.ckpt`` file.

            >>> model = Model(
            >>>     arch="vgg19",
            >>>     weights="home/workspace/.../vgg19-imagenet.pth",
            >>> )
    """

    zoo = {}  # A dictionary containing all pretrained weights of the model.

    def __init__(
        self,
        # For model architecture
        config     : Any                      = None,
        channels   : int                      = 3,
        num_classes: int               | None = None,
        classlabels: mdata.ClassLabels | None = None,
        weights    : Any                      = None,
        # For saving/loading
        name       : str               | None = None,
        variant    : int | str         | None = None,
        fullname   : str               | None = None,
        root       : core.Path                = core.Path(),
        project    : str               | None = None,
        # For training
        hparams    : dict              | None = None,
        phase      : ModelPhase | str         = ModelPhase.TRAINING,
        loss       : Any                      = None,
        metrics    : Any                      = None,
        optimizers : Any                      = None,
        debug      : dict              | None = None,
        verbose    : bool                     = True,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.config        = config
        self.channels      = channels or self.channels
        self.num_classes   = num_classes
        self.classlabels   = mdata.ClassLabels.from_value(classlabels) if classlabels is not None else None
        self.name          = name    or self.name
        self.variant       = variant or None
        self.fullname      = fullname
        self.project       = project
        self.root          = root
        self.weights       = weights
        self.hyperparams   = hparams
        self.loss          = loss
        self.train_metrics = metrics
        self.val_metrics   = metrics
        self.test_metrics  = metrics
        self.optims        = optimizers
        self.debug         = debug
        self.verbose       = verbose
        self.epoch_step    = 0
        
        # Define model
        if self.config is None:
            console.log(f"No ``config`` has been provided. Model must be manually defined.")
            self.model            = None
            self.save: list[int]  = []
            self.info: list[dict] = []
        else:
            self.model, self.save, self.info = self.parse_model()
            # Load weights (WE ONLY ATTEMPT TO LOAD WEIGHTS IF WE CAN BUILD
            # MODEL FROM ``config``).
            if self.weights:
                self.load_weights()
            else:
                self.apply(self.init_weights)
            if self.verbose:
                self.print_info()
        
        # Set phase to freeze or unfreeze layers
        self.phase = phase

    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, name: str | None = None):
        if name is None or name == "":
            name = humps.kebabize(self.__class__.__name__).lower()
        self._name = name
    
    @property
    def fullname(self) -> str:
        return self._fullname
    
    @fullname.setter
    def fullname(self, fullname: str | None = None):
        if fullname is None or fullname == "":
            if self.variant is not None:
                if self.name in str(self.variant):
                    fullname = self.variant
                else:
                    fullname = f"{self.name}-{self.variant}"
            else:
                fullname = self.name
        self._fullname = fullname
    
    @property
    def root(self) -> core.Path:
        return self._root
    
    @root.setter
    def root(self, root: Any):
        if root is None:
            root = core.Path() / "run"
        else:
            root = core.Path(root)
        self._root = root
        
        if self.project is not None and self.project != "":
            self._root /= self.project
        if self._root.name != self.fullname:
            self._root /= self.fullname
        
        self._debug_dir = self._root / "debug"
        self._ckpt_dir  = self._root / "weights"
    
    @property
    @abstractmethod
    def config_dir(self) -> core.Path:
        pass
    
    @property
    def ckpt_dir(self) -> core.Path:
        if self._ckpt_dir is None:
            self._ckpt_dir = self.root / "weights"
        return self._ckpt_dir
    
    @property
    def debug_dir(self) -> core.Path:
        if self._debug_dir is None:
            self._debug_dir = self.root / "debug"
        return self._debug_dir
    
    @property
    def debug_subdir(self) -> core.Path:
        """The debug subdir path located at: <debug_dir>/<phase>_<epoch>."""
        debug_dir = self.debug_dir / f"{self.phase.value}_{(self.current_epoch + 1):03d}"
        debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir
    
    @property
    def debug_image_file_path(self) -> core.Path:
        """The debug image file path located at: <debug_dir>/"""
        save_dir = self.debug_subdir \
            if self.debug["save_to_subdir"] \
            else self.debug_dir
        
        return save_dir / f"{self.phase.value}_" \
                          f"{(self.current_epoch + 1):03d}_" \
                          f"{(self.epoch_step + 1):06}.jpg"
    
    @property
    def zoo_dir(self) -> core.Path:
        return ZOO_DIR / self.name
    
    @property
    def config(self) -> dict | None:
        return self._config
    
    @config.setter
    def config(self, config: Any = None):
        if isinstance(config, str) and ".yaml" in config:
            config += ".yaml" if ".yaml" not in config else ""
            config  = self.config_dir / config
            
        self._config = core.load_config(config=config)
        if isinstance(self._config, dict):
            self.name     = self._config.get("name",     None)
            self.variant  = self._config.get("variant",  None)
            self.channels = self._config.get("channels", None)

    @property
    def params(self) -> int:
        if self.info is not None:
            params = [i["params"] for i in self.info]
            return sum(params)
        else:
            return 0
    
    @property
    def weights(self) -> core.Path | dict:
        return self._weights
    
    @weights.setter
    def weights(self, weights: Any = None):
        if isinstance(weights, str):
            if weights in self.zoo:
                weights = self.zoo[weights]
                weights["path"] = self.zoo_dir / weights.get("path", "")
            else:
                raise ValueError(f"``'{weights}'`` has not been defined in ``zoo``.")
        elif isinstance(weights, core.Path):
            pass
        elif isinstance(weights, dict):
            pass
        self._weights = weights

    @property
    def phase(self) -> ModelPhase:
        return self._phase
    
    @phase.setter
    def phase(self, phase: ModelPhase | str = "training"):
        self._phase = ModelPhase.from_value(value=phase)
        if self._phase is ModelPhase.TRAINING:
            self.unfreeze()
            if self.config is not None:
                freeze = self.config.get("freeze", None)
                if isinstance(freeze, list):
                    for k, v in self.model.named_parameters():
                        if any(x in k for x in freeze):
                            v.requires_grad = False
        else:
            self.freeze()
    
    @property
    def loss(self) -> mloss.Loss | None:
        return self._loss
    
    @loss.setter
    def loss(self, loss: Any):
        if isinstance(loss, mloss.Loss | nn.Module):
            self._loss = loss
        elif isinstance(loss, str):
            self._loss = LOSSES.build(name=loss)
        elif isinstance(loss, dict):
            self._loss = LOSSES.build(config=loss)
        else:
            self._loss = None
        
        if self._loss:
            self._loss.requires_grad = True
            # self._loss.cuda()
    
    @property
    def train_metrics(self) -> list[mmetric.Metric] | None:
        return self._train_metrics
    
    @train_metrics.setter
    def train_metrics(self, metrics: Any):
        """Assign train metrics.
        
        Args:
            metrics: One of the 2 options:
                - Common metrics for all train_/val_/test_metrics:
                    'metrics': {'name': 'accuracy'}
                  or,
                    'metrics': [{'name': 'accuracy'}, torchmetrics.Accuracy(), ...]
                
                - Define train_/val_/test_metrics separately:
                    'metrics': {
                        'train': ['name':'accuracy', torchmetrics.Accuracy(), ...],
                        'val':   torchmetrics.Accuracy(),
                        'test':  None,
                    }
        """
        if isinstance(metrics, dict) and "train" in metrics:
            metrics = metrics.get("train", metrics)
        
        self._train_metrics = self.create_metrics(metrics=metrics)
        # This is a simple hack since LightningModule needs the
        # metric to be defined with self.<metric>. So, here we
        # dynamically add the metric attribute to the class.
        if self._train_metrics:
            for metric in self._train_metrics:
                name = f"train/{metric.name}"
                setattr(self, name, metric)
    
    @property
    def val_metrics(self) -> list[mmetric.Metric] | None:
        return self._val_metrics
    
    @val_metrics.setter
    def val_metrics(self, metrics: Any):
        """Assign val metrics. Similar to: :meth:`self.train_metrics()`."""
        if isinstance(metrics, dict) and "val" in metrics:
            metrics = metrics.get("val", metrics)
        
        self._val_metrics = self.create_metrics(metrics)
        if self._val_metrics:
            for metric in self._val_metrics:
                name = f"val/{metric.name}"
                setattr(self, name, metric)
    
    @property
    def test_metrics(self) -> list[mmetric.Metric] | None:
        return self._test_metrics
    
    @test_metrics.setter
    def test_metrics(self, metrics: Any):
        """Assign test metrics. Similar to: :meth:`self.train_metrics()`."""
        if isinstance(metrics, dict) and "test" in metrics:
            metrics = metrics.get("test", metrics)
        
        self._test_metrics = self.create_metrics(metrics)
        if self._test_metrics:
            for metric in self._test_metrics:
                name = f"test/{metric.name}"
                setattr(self, name, metric)
    
    @staticmethod
    def create_metrics(metrics: Any):
        if isinstance(metrics, mmetric.Metric):
            if getattr(metrics, "name", None) is None:
                metrics.name = humps.depascalize(
                    humps.pascalize(metrics.__class__.__name__)
                )
            return [metrics]
        elif isinstance(metrics, dict):
            return [METRICS.build(config=metrics)]
        elif isinstance(metrics, list | tuple):
            return [
                METRICS.build(config=metric)
                if isinstance(metric, dict) else metric
                for metric in metrics
            ]
        else:
            return None
    
    @property
    def debug(self) -> dict | None:
        return self._debug
    
    @debug.setter
    def debug(self, debug: dict | None):
        if debug is None:
            self._debug = None
        else:
            self._debug = debug
            if "every_best_epoch" not in self._debug:
                self._debug.every_best_epoch = True
            if "every_n_epochs" not in self._debug:
                self._debug.every_n_epochs = 1
            if "save_to_subdir" not in self._debug:
                self._debug.save_to_subdir = True
            if "image_quality" not in self._debug:
                self._debug.image_quality = 95
            if "max_n" not in self._debug:
                self._debug.max_n = 8
            if "nrow" not in self._debug:
                self._debug.nrow = 8
            if "wait_time" not in self._debug:
                self._debug.wait_time = 0.01
    
    # Initialize Model
    
    def create_dir(self):
        for path in [self.root, self.ckpt_dir, self.debug_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    def parse_model(self) -> (
        tuple[nn.Sequential, list[int], list[dict]] |
        tuple[None, None, None]
    ):
        """Build the model. You have 2 options for building a model: (1) define
        each layer manually, or (2) build model automatically from a config
        :class:`dict`.
        
        Either way, each layer should have the following attributes:
            - i: index of the layer.
            - f: from, i.e., the current layer receives output from the f-th
                 layer. For example: -1 means from the previous layer; -2 means
                 from 2 previous layers; [99, 101] means from the 99th and 101st
                 layers. This attribute is used in forward pass.
            - t: type of the layer using this script:
                 t = str(m)[8:-2].replace("__main__.", "")
            - np: number of parameters using the following script:
              np = sum([x.numel() for x in m.parameters()])
        
        Return:
            A Sequential model.
            A :class:`list` of layer index to save the features during forward
                pass.
            A :class:`list` of layer's info for debugging.
        """
        if self.config is None:
            console.log(f"No :param:`config` has been provided. Model must be manually defined.")
            return None, None, None
        if not isinstance(self.config, dict):
            raise TypeError(f"config must be a dictionary, but got {self.config}.")
        if "backbone" not in self.config and "head" not in self.config:
            raise ValueError("config must contain 'backbone' and 'head' keys.")
        
        console.log(f"Parsing model from config.")
        
        # Name
        if "name" in self.config:
            self.name = self.name or self.config["name"]
        
        # Variant
        if "variant" in self.config:
            self.variant = self.variant or self.config["variant"]
        
        # Channels
        if "channels" in self.config:
            channels = self.config["channels"]
            if channels != self.channels:
                self.config["channels"] = self.channels
                console.log(
                    f"Overriding model.yaml channels={channels} with "
                    f"channels={self.channels}."
                )
        
        # Num_classes
        num_classes = self.num_classes
        if isinstance(self.classlabels, mdata.ClassLabels):
            num_classes = num_classes or self.classlabels.num_classes()
        if isinstance(self.weights, dict) and "num_classes" in self.weights:
            num_classes = num_classes or self.weights["num_classes"]
        self.num_classes = num_classes
        
        if "num_classes" in self.config:
            num_classes = self.config["num_classes"]
            if num_classes != self.num_classes:
                self.config["num_classes"] = self.num_classes
                console.log(
                    f"Overriding model.yaml num_classes={num_classes} with "
                    f"num_classes={self.num_classes}."
                )
        else:
            self.config["num_classes"] = num_classes
        
        # Parsing
        model, save, info = parsing.parse_model(
            d       = self.config,
            ch      = [self.channels],
            hparams = self.hyperparams,
        )
        return model, save, info
    
    @abstractmethod
    def init_weights(self, model: nn.Module):
        """Initialize model's weights."""
        pass
    
    def load_weights(self, weights: Any = None, overwrite: bool = False):
        """Load weights. It only loads the intersection layers of matching keys
        and shapes between the current model and weights.
        """
        if weights is not None:
            self.weights = weights

        if self.weights is not None:
            self.zoo_dir.mkdir(parents=True, exist_ok=True)
            if self.model is not None:
                self.model = load_weights(model=self.model, weights=self.weights)
            else:
                model_state_dict = load_state_dict(model=self, weights=self.weights)
                self.load_state_dict(model_state_dict)
            if self.verbose:
                console.log(f"Load weights from: {self.weights}!")
        else:
            error_console.log(f"[yellow]Cannot load from weights: {self.weights}!")
    
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally, you’d need one, but for GANs you might have
        multiple.

        Return:
            Any of these 6 options:
                - Single optimizer.
                - :class:`list` or :class:`tuple` of optimizers.
                - Two :class:`list` - First :class:`list` has multiple
                  optimizers, and the second has multiple LR schedulers (or
                  multiple lr_scheduler_config).
                - :class:`dict`, with an ``'optimizer'`` key, and (optionally) a
                  ``'lr_scheduler'`` key whose value is a single LR scheduler or
                  lr_scheduler_config.
                - :class:`tuple` of :class:`dict` as described above, with an
                  optional ``'frequency'`` key.
                - ``None`` - Fit will run without any optimizer.
        """
        optims = self.optims
        
        if optims is None:
            console.log(
                f"[yellow]No optimizers have been defined! Consider subclassing "
                f"this function to manually define the optimizers."
            )
            return None
        if isinstance(optims, dict):
            optims = [optims]
        assert isinstance(optims, list) and all(isinstance(o, dict) for o in optims)
        
        for optim in optims:
            # Define optimizer
            optimizer = optim.get("optimizer", None)
            if optimizer is None:
                raise ValueError(f"``optimizer`` must be defined.")
            if isinstance(optimizer,  dict):
                optimizer = OPTIMIZERS.build(net=self, config=optimizer)
            optim["optimizer"] = optimizer
            
            # Define learning rate scheduler
            lr_scheduler = optim.get("lr_scheduler", None)
            if "lr_scheduler" in optim and lr_scheduler is None:
                optim.pop("lr_scheduler")
            elif lr_scheduler is not None:
                scheduler = lr_scheduler.get("scheduler", None)
                if scheduler is None:
                    raise ValueError(f"``scheduler`` must be defined.")
                if isinstance(scheduler, dict):
                    scheduler = LR_SCHEDULERS.build(
                        optimizer = optim["optimizer"],
                        config    = scheduler
                    )
                lr_scheduler["scheduler"] = scheduler
            
            # Define optimizer frequency
            frequency = optim.get("frequency", None)
            if "frequency" in optim and frequency is None:
                optim.pop("frequency")
        
        # Re-assign optims
        self.optims = optims
        return self.optims
    
    # Forward Pass
    
    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with loss value. Loss function may need more arguments
        beside the ground-truth and prediction values. For calculating the
        metrics, we only need the final predictions and ground-truth.

        Args:
            input: An input of shape :math:`[B, C, H, W]`.
            target: A ground-truth of shape :math:`[N, C, H, W]`. Default:
                ``None``.
            
        Return:
            Predictions and loss value.
        """
        pred = self.forward(input=input, *args, **kwargs)
        # features = None
        if isinstance(pred, list | tuple):
            # features = pred[0:-1]
            pred = pred[-1]
        loss = self.loss(pred, target) if self.loss else None
        return pred, loss
        
    @abstractmethod
    def forward(
        self,
        input    : torch.Tensor,
        augment  : bool = False,
        profile  : bool = False,
        out_index: int  = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        """Forward pass. This is the primary :meth:`forward` function of the
        model. It supports augmented inference. In this function, we perform
        test-time augmentation and pass the transformed input to
        :meth:`forward_once()`.

        Args:
            input: An input of shape :math:`[N, C, H, W]`.
            augment: If ``True``, perform test-time augmentation. Default:
                ``False``.
            profile: If ``True``, Measure processing time. Default: ``False``.
            out_index: Return specific layer's output from :param:`out_index`.
                Default: ``-1`` means the last layer.
            
        Return:
            Predictions.
        """
        pass
    
    def forward_once(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        """Forward pass once. Implement the logic for a single forward pass.

        Args:
            input: An input of shape :math:`[N, C, H, W]`.
            profile: Measure processing time. Default: ``False``.
            out_index: Return specific layer's output from :param:`out_index`.
                Default: ``-1`` means the last layer.
                
        Return:
            Predictions.
        """
        x     = input
        y, dt = [], []
        for m in self.model:
            # console.log(f"{m.i}")
            if m.f != -1:  # Get features from the previous layer
                if isinstance(m.f, int):
                    x = y[m.f]  # From directly previous layer
                else:
                    x = [x if j == -1 else y[j] for j in m.f]  # From earlier layers
            x = m(x)  # pass features through current layer
            y.append(x if m.i in self.save else None)
        
        if out_index > -1 and out_index in self.save:
            output = y[out_index]
        else:
            output = x
        return output
    
    # Training
    
    def on_fit_start(self):
        """Called at the beginning of fit."""
        self.create_dir()

    def on_train_epoch_start(self):
        """Called in the training loop at the beginning of the epoch."""
        self.epoch_step = 0
    
    def training_step(
        self,
        batch    : Any,
        batch_idx: int,
        *args, **kwargs
    ) -> StepOutput | None:
        """Here you compute and return the training loss, and some additional
        metrics for e.g. the progress bar or logger.

        Args:
            batch: The output of :class:`~torch.utils.data.DataLoader`. It can
                be a :class:`torch.Tensor`, :class:`tuple` or :class:`list`.
            batch_idx: An integer displaying index of this batch.
            
        Return:
            Any of:
                - The loss tensor.
                - A :class:`dict`. Can include any keys, but must include the
                  key ``'loss'``.
                - ``None``, training will skip to the next batch.
        """
        # Forward
        input, target, extra = batch[0], batch[1], batch[2:]
        pred, loss = self.forward_loss(
            input  = input,
            target = target,
            *args, **kwargs
        )
        # Loss
        self.log(
            name           = f"train/loss",
            value          = loss,
            prog_bar       = False,
            logger         = True,
            on_step        = True,
            on_epoch       = True,
            sync_dist      = True,
            rank_zero_only = True,
            batch_size     = 1,
        )
        # Metric
        if self.train_metrics:
            for i, metric in enumerate(self.train_metrics):
                value = metric(pred, target)
                self.log(
                    name           = f"train/{metric.name}",
                    value          = value,
                    prog_bar       = False,
                    logger         = True,
                    on_step        = True,
                    on_epoch       = True,
                    sync_dist      = True,
                    rank_zero_only = True,
                    batch_size     = 1,
                )
        # Debug
        self.epoch_step += 1
        return loss

    def on_train_epoch_end(self):
        """Called in the training loop at the very end of the epoch."""
        if self.train_metrics:
            for i, metric in enumerate(self.train_metrics):
                # value = metric.compute()
                metric.reset()
        """
        # Loss
        loss = torch.stack([x["loss"] for x in epoch_output]).mean()
        if self.trainer.is_global_zero:
            self.log(
                name      = f"loss/train_epoch",
                value     = loss,
                prog_bar  = False,
                on_step   = False,
                on_epoch  = True,
                sync_dist = True,
            )
            self.tb_log_scalar(f"loss/train_epoch", loss, "epoch")
        # Metrics
        if self.train_metrics:
            for i, metric in enumerate(self.train_metrics):
                value = metric.compute()
                metric.reset()
                if self.trainer.is_global_zero:
                    self.log(
                        name      = f"{metric.name}/train_epoch",
                        value     = value,
                        prog_bar  = False,
                        on_step   = False,
                        on_epoch  = True,
                        sync_dist = True,
                    )
                    self.tb_log_scalar(f"{metric.name}/train_epoch", value, "epoch")
        """

    def on_validation_epoch_start(self):
        """Called in the validation loop at the beginning of the epoch."""
        self.epoch_step = 0
    
    def validation_step(
        self,
        batch    : Any,
        batch_idx: int,
        *args, **kwargs
    ) -> StepOutput | None:
        """Operates on a single batch of data from the validation set. In this
        step, you might generate examples or calculate anything of interest like
        accuracy.
        
        Args:
            batch: The output of :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.

        Return:
            - Any object or value.
            - ``None``, validation will skip to the next batch.
        """
        input, target, extra = batch[0], batch[1], batch[2:]
        pred, loss = self.forward_loss(
            input  = input,
            target = target,
            *args, **kwargs
        )
        # Loss
        self.log(
            name           = f"val/loss",
            value          = loss,
            prog_bar       = False,
            logger         = True,
            on_step        = True,
            on_epoch       = True,
            sync_dist      = True,
            rank_zero_only = True,
            batch_size     = 1,
        )
        # Metric
        if self.val_metrics:
            for i, metric in enumerate(self.val_metrics):
                value = metric(pred, target)
                self.log(
                    name           = f"val/{metric.name}",
                    value          = value,
                    prog_bar       = False,
                    logger         = True,
                    on_step        = True,
                    on_epoch       = True,
                    sync_dist      = True,
                    rank_zero_only = True,
                    batch_size     = 1,
                )
        # Debug
        epoch = self.current_epoch + 1
        if self.debug \
            and epoch % self.debug["every_n_epochs"] == 0 \
            and self.epoch_step < self.debug["max_n"]:
            if self.trainer.is_global_zero:
                self.show_results(
                    input    = input,
                    target   = target,
                    pred     = pred,
                    file_path= self.debug_image_file_path,
                    **self.debug | {
                        "max_n": input[0],
                        "nrow" : input[0],
                    }
                )
        self.epoch_step += 1
        return loss

    def on_validation_epoch_end(self):
        """Called in the validation loop at the very end of the epoch."""
        if self.val_metrics:
            for i, metric in enumerate(self.val_metrics):
                # value = metric.compute()
                metric.reset()

    def on_test_start(self) -> None:
        """Called at the very beginning of testing."""
        self.create_dir()
    
    def on_test_epoch_start(self):
        """Called in the test loop at the very beginning of the epoch."""
        self.epoch_step = 0
    
    def test_step(
        self,
        batch    : Any,
        batch_idx: int,
        *args, **kwargs
    ) -> StepOutput | None:
        """Operates on a single batch of data from the test set. In this step
        you'd normally generate examples or calculate anything of interest such
        as accuracy.

        Args:
            batch: The output of your :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.

        Return:
            Any of:
                - Any object or value.
                - ``None``, testing will skip to the next batch.
        """
        input, target, extra = batch[0], batch[1], batch[2:]
        pred, loss = self.forward_loss(
            input  = input,
            target = target,
            *args, **kwargs
        )
        # Loss
        self.log(
            name           = f"test/loss",
            value          = loss,
            prog_bar       = False,
            logger         = True,
            on_step        = True,
            on_epoch       = True,
            sync_dist      = True,
            rank_zero_only = True,
            batch_size     = 1,
        )
        # Metric
        if self.test_metrics:
            for i, metric in enumerate(self.test_metrics):
                value = metric(pred, target)
                self.log(
                    name           = f"test/{metric.name}",
                    value          = value,
                    prog_bar       = False,
                    logger         = True,
                    on_step        = True,
                    on_epoch       = True,
                    sync_dist      = True,
                    rank_zero_only = True,
                    batch_size     = 1,
                )
        # Debug
        epoch = self.current_epoch + 1
        if self.debug \
            and epoch % self.debug["every_n_epochs"] == 0 \
            and self.epoch_step < self.debug["max_n"]:
            if self.trainer.is_global_zero:
                self.show_results(
                    input    = input,
                    target   = target,
                    pred     = pred,
                    file_path= self.debug_image_file_path,
                    **self.debug | {
                        "max_n": input[0],
                        "nrow" : input[0],
                    }
                )
        self.epoch_step += 1
        return loss
    
    def on_test_epoch_end(self):
        """Called in the test loop at the very end of the epoch."""
        if self.test_metrics:
            for i, metric in enumerate(self.test_metrics):
                # value = metric.compute()
                metric.reset()
    
    def export_to_onnx(
        self,
        input_dims   : list[int] | None = None,
        file_path    : core.Path | None = None,
        export_params: bool             = True
    ):
        """Export the model to ``onnx`` format.

        Args:
            input_dims: Input dimensions in :math:`[C, H, W]` format. Default:
                ``None``.
            file_path: Path to save the model. If ``None`` or empty, then save
                to :attr:`root`. Default: ``None``.
            export_params: Should export parameters? Default: ``True``.
        """
        # Check file_path
        if file_path in [None, ""]:
            file_path = self.root / f"{self.fullname}.onnx"
        if ".onnx" not in str(file_path):
            file_path = core.Path(str(file_path) + ".onnx")
        
        if input_dims is not None:
            input_sample = torch.randn(input_dims)
        else:
            raise ValueError(f"input_dims must be defined.")
        
        self.to_onnx(
            file_path     = file_path,
            input_sample  = input_sample,
            export_params = export_params
        )
    
    def export_to_torchscript(
        self,
        input_dims: list[int] | None = None,
        file_path : core.Path | None = None,
        method    : str              = "script"
    ):
        """Export the model to TorchScript format.

        Args:
            input_dims: Input dimensions. Default: ``None``.
            file_path: Path to save the model. If ``None`` or empty, then save to
                :attr:`root`. Default: ``None``.
            method: Whether to use TorchScript's `''script''` or ``'trace'``
                method. Default: ``'script'``.
        """
        # Check file path
        if file_path in [None, ""]:
            file_path = self.root / f"{self.fullname}.pt"
        if ".pt" not in str(file_path):
            file_path = core.Path(str(file_path) + ".pt")
        
        if input_dims is not None:
            input_sample = torch.randn(input_dims)
        else:
            raise ValueError(f"'input_dims' must be defined.")
        
        script = self.to_torchscript(method=method, example_inputs=input_sample)
        torch.jit.save(script, file_path)
    
    @abstractmethod
    def show_results(
        self,
        input        : torch.Tensor | None = None,
        target       : torch.Tensor | None = None,
        pred         : torch.Tensor | None = None,
        file_path    : core.Path    | None = None,
        image_quality: int                 = 95,
        max_n        : int          | None = 8,
        nrow         : int          | None = 8,
        wait_time    : float               = 0.01,
        save         : bool                = False,
        verbose      : bool                = False,
        *args, **kwargs
    ):
        """Show results.

        Args:
            input: An input.
            target: A ground-truth.
            pred: A prediction.
            file_path: A path to save the debug result.
            image_quality: The image quality to be saved. Default: ``95``.
            max_n: Show max ``n`` items if :param:`input` has a batch size of
                more than :param:`max_n` items. Default: ``None`` means show
                all.
            nrow: The maximum number of items to display in a row. Default:
                ``8``.
            wait_time: Wait for some time (in seconds) to display the figure
                then reset. Default: ``0.01``.
            save: Save debug image. Default: ``False``.
            verbose: If ``True`` shows the results on the screen. Default:
                ``False``.
        """
        pass
    
    def print_info(self):
        if self.verbose and self.model is not None:
            console.log(f"[red]{self.fullname}")
            core.print_table(self.info)
            console.log(f"Save indexes: {self.save}")
    
# endregion
