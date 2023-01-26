#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base model class for building machine learning
models.
"""

from __future__ import annotations

__all__ = [
    "Model", "attempt_load", "extract_weights_from_checkpoint", "get_epoch",
    "get_global_step", "get_latest_checkpoint", "intersect_state_dicts",
    "is_parallel", "load_state_dict_from_path", "match_state_dicts",
    "sparsity", "strip_optimizer",
]

import os
from abc import ABC, abstractmethod
from typing import Any, Sequence, TYPE_CHECKING

import humps
import lightning
import munch
import torch
from torch import nn
from torch.nn import parallel

from mon import core
from mon.coreml import (constant, data as d, loss as l, metric as m)

if TYPE_CHECKING:
    from mon.coreml.typing import (
        ClassLabelsType, ConfigType, DictType, EpochOutput, Ints, LossesType,
        MetricsType, ModelPhaseType, OptimizersType, PathType, PretrainedType,
        StepOutput,
    )


# region Checkpoint

def extract_weights_from_checkpoint(
    ckpt       : PathType,
    weight_file: PathType | None = None,
):
    """Extract and save weights from the checkpoint :attr:`ckpt`.
    
    Args:
        ckpt: A checkpoint file.
        weight_file: A path to save the extracting weights. Defaults to None,
            which saves the weights at the same location as the :param:`ckpt`
            file.
    """
    ckpt = PathType(ckpt)
    assert ckpt.is_ckpt_file()
    
    state_dict = load_state_dict_from_path(str(ckpt))
    if state_dict is None:
        raise ValueError()
    
    if weight_file is None:
        weight_file = ckpt.parent / f"{ckpt.stem}.pth"
    else:
        weight_file = PathType(weight_file)
    core.create_dirs([weight_file.parent])
    torch.save(state_dict, str(weight_file))


def get_epoch(ckpt: PathType | None) -> int:
    """Get an epoch value stored in a checkpoint file.

    Args:
        ckpt: A checkpoint filepath.
    """
    if ckpt is None:
        return 0
    
    epoch = 0
    ckpt  = PathType(ckpt)
    assert ckpt.is_ckpt_file()
    if ckpt.is_torch_file():
        ckpt  = torch.load(ckpt)
        epoch = ckpt.get("epoch", 0)
    return epoch


def get_global_step(ckpt: PathType | None) -> int:
    """Get a global step stored in a checkpoint file.

    Args:
        ckpt: A checkpoint filepath.
    """
    if ckpt is None:
        return 0

    global_step = 0
    ckpt        = PathType(ckpt)
    assert ckpt.is_ckpt_file()
    if ckpt.is_torch_file():
        ckpt        = torch.load(ckpt)
        global_step = ckpt.get("global_step", 0)
    return global_step


def get_latest_checkpoint(dirpath: PathType) -> str | None:
    """Get the latest checkpoint (last saved) filepath in a directory.

    Args:
        dirpath: The directory that contains the checkpoints.
    """
    dirpath  = PathType(dirpath)
    ckpt     = core.get_latest_file(dirpath)
    if ckpt is None:
        core.error_console.log(f"[red]Cannot find checkpoint file {dirpath}.")
    return ckpt

# endregion


# region Weight/State Dict

def intersect_state_dicts(da: dict, db: dict, exclude: Sequence = ()) -> dict:
    """Find the intersection between two dictionaries.
    
    Args:
        da: The first dictionary.
        db: The second dictionary.
        exclude: A list of excluding keys.
    
    Return:
        A dictionary that contains only the keys that are in both dictionaries,
        and whose values have the same shape.
    """
    return {
        k: v for k, v in da.items()
        if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape
    }


def match_state_dicts(
    model_dict	   : dict,
    pretrained_dict: dict,
    exclude		   : Sequence = ()
) -> dict:
    """Filter out unmatched keys btw the model's :attr:`state_dict` and the
    weights's :attr:`state_dict`. Omitting :param:`exclude` keys.

    Args:
        model_dict: Model's :attr:`state_dict`.
        pretrained_dict: Pretrained's :attr:`state_dict`.
        exclude: List of excluded keys. Defaults to ().
        
    Return:
        Filtered model's :attr:`state_dict`.
    """
    # 1. Filter out unnecessary keys
    intersect_dict = intersect_state_dicts(
        da      = pretrained_dict,
        db      = model_dict,
        exclude = exclude
    )
    # 2. Overwrite entries in the existing state dict
    model_dict.update(intersect_dict)
    return model_dict


def strip_optimizer(weight_file: str, new_file: str | None = None):
    """Strip the optimizer from a saved weight file to finalize the training
    process.
    
    Args:
        weight_file: A PyTorch saved weight filepath.
        new_file: A filepath to save the stripped weights. If :param:`new_file`
            is given, save the weights as a new file. Otherwise, overwrite the
            :param:`weight_file`.
    """
    assert PathType(weight_file).is_weights_file()
    
    x = torch.load(weight_file, map_location=torch.device("cpu"))
    x["optimizer"]        = None
    x["training_results"] = None
    x["epoch"]            = -1
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
        
    torch.save(x, new_file or weight_file)
    mb = os.path.getsize(new_file or weight_file) / 1E6  # filesize
    core.console.log(
        "Optimizer stripped from %s,%s %.1fMB"
        % (weight_file, (" saved as %s," % new_file) if new_file else "", mb)
    )
    
# endregion


# region Model Loading

def attempt_load(
    model      : nn.Module  | Model,
    weights    : PathType   | DictType,
    name       : str        | None = None,
    cfg        : ConfigType | None = None,
    fullname   : str        | None = None,
    num_classes: int        | None = None,
    phase      : str               = "inference",
    strict     : bool              = True,
    file_name  : str        | None = None,
    model_dir  : PathType   | None = constant.PRETRAINED_DIR,
    debugging  : bool              = True
):
    """Try to create a model from the given configuration and load a weights
    weights.
    
    Args:
        model: A PyTorch :class:`nn.Module` or a :class:`Model`.
        weights: A weight dictionary or a checkpoint filepath.
        name: A name of the model.
        cfg: A configuration dictionary or a YAML filepath containing the
            building configuration of the model.
        fullname: An optional fullname of the model, in other words,
            a model's base name + its variant + a training dataset name.
        num_classes: The number of classes, in other words, the final layer's
            output channels.
        phase: The model running phase.
        strict: Defaults to True.
        file_name: Name for the downloaded file. Filename from :param:`path`
        model_dir: Directory in which to save the object. Default to None.
        debugging: If True, stop and raise errors. Otherwise, just return the
            current model.
        
    Return:
        A :class:`Model` object.
    """
    # Get model's configuration
    if cfg is not None:
        cfg = core.load_config(cfg=cfg)
        
    # Get model
    if model is None:
        if name is None and cfg is None:
            if debugging:
                raise RuntimeError
            else:
                return model
        else:
            model: Model = constant.MODEL.build(
                name        = name,
                fullname    = fullname,
                cfg         = cfg,
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
    elif isinstance(model, Model) and core.is_ckpt_file(path=weights):
        model = model.load_from_checkpoint(
            checkpoint_path = weights,
            name            = name,
            cfg             = cfg,
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
        elif isinstance(weights, dict | munch.Munch):
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
    path  		: PathType,
    model_dir   : PathType | None = None,
    map_location: str      | None = None,
    progress	: bool 	   	      = True,
    check_hash	: bool	   	      = False,
    file_name 	: str      | None = None,
    **_
) -> dict | None:
    """Load state dict at the given URL. If downloaded file is a zip file, it
    will be automatically decompressed. If the object is already present in
    :param:`model_dir`, it is deserialized and returned.
    
    Args:
        path: The weights or checkpoints file to load. If it is a URL, it will
            be downloaded.
        model_dir: Directory in which to save the object. Default to None.
        map_location: A function, or a dict specifying how to remap storage
            locations (see torch.load). Defaults to None.
        progress: Whether to display a progress bar to stderr. Defaults to True.
        check_hash: If True, the file_name part of the URL should follow the
            naming convention `file_name-<sha256>.ext` where `<sha256>` is the
            first eight or more digits of the SHA256 hash of the contents of the
            file. Hash is used to ensure unique names and to verify the contents
            of the file. Defaults to False.
        file_name: Name for the downloaded file. Filename from :param:`path`
            will be used if not set.
    """
    if path is None:
        raise ValueError()
    if model_dir:
        model_dir = PathType(model_dir)
    
    path = PathType(path)
    if not path.is_torch_file() \
        and (model_dir is None or not model_dir.is_dir()):
        raise ValueError(
            f":param:`model_dir` must be defined. But got: {model_dir}."
        )
    
    save_weight = ""
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

# endregion


# region Model Parsing

def is_parallel(model: nn.Module) -> bool:
    """Return True if a model is in a parallel run-mode. Otherwise, return
    False.
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

class Model(lightning.LightningModule, ABC):
    """The base class for all machine learning models.
    
    Args:
        cfg: Model's layers configuration. It can be a dictionary or an external
            .yaml path. Defaults to None mean you must define manually each
            layer in :meth:`self.parse_model()` method.
        hparams: Layer's hyperparameters. They are used to change the values
                of :param:`args`. Usually used in grid search or random search
                during training. Defaults to None.
        channels: The input channel. Defaults to 3.
        num_classes: A number of classes for classification or detection tasks.
            Defaults to None.
        classlabels: A :class:`ClassLabels` object that contains all labels in
            the dataset. Defaults to None.
        weights: Initialize weights from weights.
            - If True, use the original weights described by the author
              (usually, ImageNet or COCO). By default, it is the first element
              in the `pretrained_weights` dictionary.
            - If str and is a file/path, then load weights from saved file.
            - In each inherited model, `weights` can be a dictionary's
              key to get the corresponding local file or URL of the weight.
        name: A model's name. If None is given, it will be
            :attr:`self.__class__.__name__`. Defaults to None.
        variant: A model's variant. Defaults to None.
        fullname: A full name should have the following format:
            {name}-{dataset}-{postfix}. Defaults to None.
        root: The root directory of the model. Defaults to :attr:`RUNS_DIR`.
        project: A project name. Defaults to None.
        phase: Model's running phase. Defaults to training.
        loss: Loss function for training the model. Defaults to None.
        metrics: A list metrics for validating and testing model. Defaults to
            None.
        optimizers: Optimizer(s) for training model. Defaults to None.
        debug: Debug configs. Defaults to None.
        verbose: Verbosity.
    """
    
    cfgs = {}
    pretrained_weights = {}
    
    def __init__(
        self,
        cfg        : ConfigType      | None = None,
        hparams    : DictType        | None = None,
        channels   : int                    = 3,
        num_classes: int             | None = None,
        classlabels: ClassLabelsType | None = None,
        weights    : PretrainedType   	    = False,
        # For management
        name       : str             | None = None,
        variant    : str             | None = None,
        fullname   : str             | None = None,
        root       : PathType               = constant.RUNS_DIR,
        project    : str             | None = None,
        # For training
        phase      : ModelPhaseType         = "training",
        loss   	   : LossesType      | None = None,
        metrics	   : MetricsType     | None = None,
        optimizers : OptimizersType  | None = None,
        debug      : DictType        | None = None,
        verbose    : bool                   = True,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.name          = name
        self.variant       = variant
        self.fullname      = fullname
        self.project       = project
        self.root          = root
        self.cfg           = cfg
        self.hyperparams   = hparams
        self.num_classes   = num_classes
        self.weights       = weights
        self.loss          = loss
        self.train_metrics = metrics
        self.val_metrics   = metrics
        self.test_metrics  = metrics
        self.optims        = optimizers
        self.debug         = debug
        self.verbose       = verbose
        self.epoch_step    = 0
        
        # Define model
        self.model = None
        self.save  = None
        self.info  = None
        
        if self.cfg is not None:
            assert isinstance(self.cfg, dict | munch.Munch)
            core.console.log(f"Parsing model from :attr:`cfg`.")
            
            self.channels        = self.cfg.get("channels", channels)
            self.cfg["channels"] = self.channels
            
            self.classlabels = d.ClassLabels.from_value(classlabels)
            if self.classlabels:
                num_classes = num_classes or self.classlabels.num_classes()
            if isinstance(self.weights, dict | munch.Munch) and "num_classes" in self.weights:
                num_classes = num_classes or self.weights["num_classes"]
            self.num_classes = num_classes
            
            if self.num_classes:
                nc = self.cfg.get("num_classes", None)
                if self.num_classes != nc:
                    core.console.log(
                        f"Overriding model.yaml num_classes={nc} "
                        f"with num_classes={self.num_classes}."
                    )
                    self.cfg["num_classes"] = self.num_classes
            
            assert isinstance(self.cfg, dict | munch.Munch) \
                   and hasattr(self.cfg, "backbone") \
                   and hasattr(self.cfg, "head")
            
            # Actual model, save list during forward, layer's info
            self.model, self.save, self.info = self.parse_model(
                d       = self.cfg,
                ch      = [self.channels],
                hparams = self.hyperparams,
            )
            
            # Load weights
            if self.weights:
                self.load_weights()
            else:
                self.apply(self.init_weights)
            self.print_info()

        # Set phase to freeze or unfreeze layers
        self.phase = phase
    
    @property
    def cfg(self) -> DictType | None:
        return self._cfg
    
    @cfg.setter
    def cfg(self, cfg: DictType | None = None):
        variant = None
        if isinstance(cfg, str) and cfg in self.cfgs:
            variant = str(cfg)
            cfg     = self.cfgs[cfg]
        elif isinstance(cfg, str | core.Path):
            if not cfg.is_yaml_file():
                cfg = self.cfg_dir / "cfg" / cfg
            variant = str(cfg.stem)
        elif isinstance(cfg, dict):
            variant = cfg.get("name", None)
        self._cfg    = core.load_config(cfg=cfg)
        self.variant = self.variant or variant
    
    @property
    def params(self) -> int:
        if self.info is not None:
            params = [i["params"] for i in self.info]
            return sum(params)
        else:
            return 0
    
    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, weights: PretrainedType = False):
        if isinstance(weights, dict | munch.Munch):
            pass
        elif weights is True and len(self.pretrained_weights):
            weights = list(self.pretrained_weights.values())[0]
        elif weights in self.pretrained_weights:
            weights = self.pretrained_weights[weights]
        else:
            weights = False
        """
        if isinstance(weights, str) \
            and not foundation.Path(weights).is_torch_file():
            if (self.variant is not None) and (self.variant not in weights):
                weights = f"{self.variant}-{weights}"
        """
        self._weights = weights
        
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, name: str | None = None):
        self._name = name \
            if (name is not None and name != "") \
            else humps.kebabize(self.__class__.__name__).lower()
    
    @property
    def fullname(self) -> str:
        return self._fullname
    
    @fullname.setter
    def fullname(self, fullname: str | None = None):
        self._fullname = fullname \
            if (fullname is not None and fullname != "") \
            else self.name
    
    @property
    def root(self) -> PathType:
        return self._root
    
    @root.setter
    def root(self, root: PathType):
        if root is None:
            root = constant.RUNS_DIR / "train"
        else:
            root = core.Path(root)
        self._root = root
        
        if self.project is not None and self.project != "":
            self._root = self._root / self.project
        if self._root.name != self.fullname:
            self._root = self._root / self.fullname

        self._debug_dir   = self._root / "debugs"
        self._weights_dir = self._root / "weights"
    
    @property
    @abstractmethod
    def cfg_dir(self) -> PathType:
        pass
    
    @property
    def debug_dir(self) -> PathType:
        if self._debug_dir is None:
            self._debug_dir = self.root / "debug"
        return self._debug_dir
    
    @property
    def debug_subdir(self) -> PathType:
        """The debug subdir path located at: <debug_dir>/<phase>_<epoch>."""
        debug_dir = self.debug_dir / \
            f"{self.phase.value}_{(self.current_epoch + 1):03d}"
        core.create_dirs(paths=[debug_dir])
        return debug_dir
    
    @property
    def debug_image_filepath(self) -> PathType:
        """The debug image filepath located at: <debug_dir>/"""
        save_dir = self.debug_subdir \
            if self.debug.save_to_subdir \
            else self.debug_dir
            
        return save_dir / f"{self.phase.value}_" \
                          f"{(self.current_epoch + 1):03d}_" \
                          f"{(self.epoch_step + 1):06}.jpg"
    
    @property
    def pretrained_dir(self) -> PathType:
        return constant.PRETRAINED_DIR / self.name
        
    @property
    def weights_dir(self) -> PathType:
        if self._weights_dir is None:
            self._weights_dir = self.root / "weights"
        return self._weights_dir
        
    @property
    def debug(self) -> munch.Munch | None:
        return self._debug
    
    @debug.setter
    def debug(self, debug: DictType | None):
        if debug is None:
            self._debug = None
        else:
            self._debug = munch.Munch.fromDict(debug) if isinstance(debug, dict) else debug
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
    
    @property
    def phase(self) -> constant.ModelPhase:
        return self._phase
    
    @phase.setter
    def phase(self, phase: ModelPhaseType = "training"):
        self._phase = constant.ModelPhase.from_value(phase)
        if self._phase is constant.ModelPhase.TRAINING:
            self.unfreeze()
            if self.cfg is not None:
                freeze = self.cfg.get("freeze", None)
                if isinstance(freeze, list):
                    for k, v in self.model.named_parameters():
                        if any(x in k for x in freeze):
                            v.requires_grad = False
        else:
            self.freeze()
    
    @property
    def loss(self) -> l.Loss | None:
        return self._loss
    
    @loss.setter
    def loss(self, loss: LossesType | None):
        if isinstance(loss, l.Loss |  nn.Module):
            self._loss = loss
        elif isinstance(loss, dict):
            self._loss = constant.LOSS.build_from_dict(cfg=loss)
        else:
            self._loss = None
        
        if self._loss:
            self._loss.requires_grad = True
            # self._loss.cuda()
    
    @property
    def train_metrics(self) -> list[m.Metric] | None:
        return self._train_metrics
    
    @train_metrics.setter
    def train_metrics(self, metrics: MetricsType | None):
        """Assign train metrics.
        
        Args:
            metrics: One of the 2 options:
                - Common metrics for train_/val_/test_metrics:
                    "metrics": dict(name="accuracy")
                  or,
                    "metrics": [dict(name="accuracy"), torchmetrics.Accuracy(),]
                
                - Define train_/val_/test_metrics separately:
                    "metrics": {
                        "train": [dict(name="accuracy"), dict(name="f1")]
                        "val":   torchmetrics.Accuracy(),
                        "test":  None,
                    }
        """
        if isinstance(metrics, dict) and "train" in metrics:
            metrics = metrics.get("train", metrics)
            
        self._train_metrics = self.create_metrics(metrics)
        # This is a simple hack since LightningModule needs the
        # metric to be defined with self.<metric>. Here we dynamically
        # add the metric attribute to the class.
        if self._train_metrics:
            for metric in self._train_metrics:
                name = f"train_{metric.name}"
                setattr(self, name, metric)
    
    @property
    def val_metrics(self) -> list[m.Metric] | None:
        return self._val_metrics
    
    @val_metrics.setter
    def val_metrics(self, metrics: MetricsType | None):
        """Assign val metrics.
        
        Args:
            metrics: One of the 2 options:
                - Common metrics for train_/val_/test_metrics:
                    "metrics": dict(name="accuracy")
                  or,
                    "metrics": [dict(name="accuracy"), torchmetrics.Accuracy(),]
                
                - Define train_/val_/test_metrics separately:
                    "metrics": {
                        "train": [dict(name="accuracy"), dict(name="f1")]
                        "val":   torchmetrics.Accuracy(),
                        "test":  None,
                    }
        """
        if isinstance(metrics, dict) and "val" in metrics:
            metrics = metrics.get("val", metrics)
            
        self._val_metrics = self.create_metrics(metrics)
        # This is a simple hack since LightningModule needs the
        # metric to be defined with self.<metric>. Here we dynamically
        # add the metric attribute to the class.
        if self._val_metrics:
            for metric in self._val_metrics:
                name = f"val_{metric.name}"
                setattr(self, name, metric)
    
    @property
    def test_metrics(self) -> list[m.Metric] | None:
        return self._test_metrics
    
    @test_metrics.setter
    def test_metrics(self, metrics: MetricsType | None):
        """Assign test metrics.
        
        Args:
            metrics: One of the 2 options:
                - Common metrics for train_/val_/test_metrics:
                    "metrics": dict(name="accuracy")
                  or,
                    "metrics": [dict(name="accuracy"), torchmetrics.Accuracy(),]
                
                - Define train_/val_/test_metrics separately:
                    "metrics": {
                        "train": [dict(name="accuracy"), dict(name="f1")]
                        "val":   torchmetrics.Accuracy(),
                        "test":  None,
                    }
        """
        if isinstance(metrics, dict) and "test" in metrics:
            metrics = metrics.get("test", metrics)
            
        self._test_metrics = self.create_metrics(metrics)
        # This is a simple hack since LightningModule needs the
        # metric to be defined with self.<metric>. Here we dynamically
        # add the metric attribute to the class.
        if self._test_metrics:
            for metric in self._test_metrics:
                name = f"test_{metric.name}"
                setattr(self, name, metric)
    
    @staticmethod
    def create_metrics(metrics: MetricsType | None) -> list[m.Metric] | None:
        if isinstance(metrics, m.Metric):
            return [metrics]
        elif isinstance(metrics, dict):
            return [constant.METRIC.build(cfg=metrics)]
        elif isinstance(metrics, list):
            return [
                constant.METRIC.build(cfg=metric)
                if isinstance(metric, dict) else metric for metric in metrics
            ]
        else:
            return None
    
    @abstractmethod
    def parse_model(
        self,
        d      : dict      | None = None,
        ch     : list[int] | None = None,
        hparams: DictType  | None = None,
    ) -> tuple[nn.Sequential, list[int], list[dict]]:
        """Build the model. You have 2 options for building a model: (1) define
        each layer manually, or (2) build model automatically from a config
        dictionary.
        
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
        
        Args:
            d: Model definition dictionary. Default to None means building the
                model manually.
            ch: The first layer's input channels. If given, it will be used to
                further calculate the next layer's input channels. Defaults to
                None means defines each layer in_ and out_channels manually.
            hparams: Layer's hyperparameters. They are used to change the values
                of :param:`args`. Usually used in grid search or random search
                during training. Defaults to None.
            
        Return:
            A Sequential model.
            A list of layer index to save the features during forward pass.
            A list of layer's info for debugging.
        """
        pass
    
    @abstractmethod
    def init_weights(self, model: nn.Module):
        """Initialize model's weights."""
        pass
    
    def load_weights(self):
        """Load weights. It only loads the intersection layers of matching keys
        and shapes between the current model and weights.
        """
        if isinstance(self.weights, dict | munch.Munch | str | core.Path):
            core.create_dirs(paths=[self.pretrained_dir])
            self.model = attempt_load(
                model	  = self.model,
                weights   = self.weights,
                strict	  = False,
                model_dir = self.pretrained_dir,
            )
            if self.verbose:
                core.console.log(f"Load weights from: {self.weights}!")
        else:
            core.error_console.log(
                f"[yellow]Cannot load from weights: {self.weights}!"
            )
    
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally, you’d need one, but for GANs you might have
        multiple.

        Return:
            Any of these 6 options:
                - Single optimizer.
                - List or Tuple of optimizers.
                - Two lists - First list has multiple optimizers, and the
                  second has multiple LR schedulers (or multiple
                  lr_scheduler_config).
                - Dictionary, with an "optimizer" key, and (optionally) a
                  "lr_scheduler" key whose value is a single LR scheduler or
                  lr_scheduler_config.
                - Tuple of dictionaries as described above, with an optional
                  "frequency" key.
                - None - Fit will run without any optimizer.
        """
        optims = self.optims

        if optims is None:
            core.console.log(
                f"[yellow]No optimizers have been defined! Consider subclassing "
                f"this function to manually define the optimizers."
            )
            return None
        if isinstance(optims, dict):
            optims = [optims]
        assert isinstance(optims, list) and all(isinstance(o, dict | munch.Munch) for o in optims)
      
        for optim in optims:
            # Define optimizer measurement
            optimizer = optim.get("optimizer", None)
            if optimizer is None:
                raise ValueError(f"`optimizer` must be defined.")
            if isinstance(optimizer, dict):
                optimizer = constant.OPTIMIZER.build(net=self, cfg=optimizer)
            optim["optimizer"] = optimizer

            # Define learning rate scheduler
            lr_scheduler = optim.get("lr_scheduler", None)
            if "lr_scheduler" in optim and lr_scheduler is None:
                optim.pop("lr_scheduler")
            elif lr_scheduler is not None:
                scheduler = lr_scheduler.get("scheduler", None)
                if scheduler is None:
                    raise ValueError(f"`scheduler` must be defined.")
                if isinstance(scheduler, dict):
                    scheduler = constant.LR_SCHEDULER.build(
                        optimizer = optim["optimizer"],
                        cfg       = scheduler
                    )
                lr_scheduler["scheduler"] = scheduler
            
            # Define optimizer frequency
            frequency = optim.get("frequency", None)
            if "frequency" in optim and frequency is None:
                optim.pop("frequency")
        
        # Re-assign optims
        self.optims = optims
        return self.optims
    
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
            input: An input of shape [B, C, H, W].
            target: A ground-truth of shape [B, C, H, W].
            
        Return:
            Predictions and loss value.
        """
        pred     = self.forward(input=input, *args, **kwargs)
        features = None
        if isinstance(pred, (list, tuple)):
            features = pred[0:-1]
            pred     = pred[-1]
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
            input: An input of shape [B, C, H, W].
            augment: If True, perform test-time augmentation. Defaults to False.
            profile: If True, Measure processing time. Defaults to False.
            out_index: Return specific layer's output from :param:`out_index`.
                Defaults to -1 means the last layer.
            
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
            input: An input of shape [B, C, H, W].
            profile: Measure processing time. Defaults to False.
            out_index: Return specific layer's output from :param:`out_index`.
                Defaults to -1 means the last layer.
                
        Return:
            Predictions.
        """
        x     = input
        y, dt = [], []
        for m in self.model:
            
            # console.log(f"{m.i}")
            
            if m.f != -1:  # Get features from previous layer
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
    
    def on_fit_start(self):
        """Called at the beginning of fit."""
        core.create_dirs(
            paths = [
                self.root,
                self.weights_dir,
                self.debug_dir
            ]
        )
    
    def on_train_epoch_start(self):
        """Called in the training loop at the beginning of the epoch."""
        self.epoch_step = 0
    
    def training_step(
        self,
        batch    : Any,
        batch_idx: int,
        *args, **kwargs
    ) -> StepOutput | None:
        """Training step.

        Args:
            batch: A batch of inputs. It can be a tuple of (input, target, extra).
            batch_idx: A batch index.

        Return:
            Outputs:
                - A single loss tensor.
                - A dictionary with the first key must be the loss.
                - None, training will skip to the next batch.
        """
        input, target, extra = batch[0], batch[1], batch[2:]
        pred, loss = self.forward_loss(
            input  = input,
            target = target,
            *args, **kwargs
        )
        return {
            "loss"  : loss,
            "input" : input,
            "target": target,
            "pred"  : pred
        }
    
    def training_step_end(
        self,
        outputs: StepOutput | None,
        *args, **kwargs
    ) -> StepOutput | None:
        """Use this when training with dp or ddp2 because :meth:`training_step`
        will operate on only part of the batch. However, this is still optional
        and only needed for things like softmax or NCE loss.
        
        Note:
            If you later switch to ddp or some other mode, this will still be
            called so that you don't have to change your code.
        """
        if not isinstance(outputs, dict):
            return None
        
        # Gather results
        # For DDP strategy, gather outputs from multiple devices
        if self.trainer.num_devices > 1:
            outputs = self.all_gather(outputs)
        
        loss   = outputs["loss"]    # losses from each device
        input  = outputs["input"]   # images from each device
        target = outputs["target"]  # ground-truths from each device
        pred   = outputs["pred"]    # predictions from each device
        
        # Tensors
        if self.trainer.num_devices > 1:
            input  = input.flatten(start_dim=0,  end_dim=1)
            target = target.flatten(start_dim=0, end_dim=1)
            pred   = pred.flatten(start_dim=0,   end_dim=1)
        
        # Loss
        loss = loss.mean() if loss is not None else None
        self.ckpt_log_scalar(f"checkpoint/loss/train_step", loss)
        # self.tb_log(f"{loss_tag}", loss, "step")
       
        # Metrics
        if self.train_metrics:
            for i, metric in enumerate(self.train_metrics):
                value = metric(pred, target)
                self.ckpt_log_scalar(
                    f"checkpoint/{metric.name}/train_step", value, True
                )
                # self.tb_log(f"{metric.name}/train_step", value, "step")
        
        self.epoch_step += 1
        return {
            "loss"  : loss,
            # "input" : input,
            # "target": target,
            # "pred"  : pred,
        }
    
    def training_epoch_end(self, outputs: EpochOutput):
        # Loss
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.ckpt_log_scalar(f"checkpoint/loss/train_epoch", loss)
        self.tb_log_scalar(f"loss/train_epoch", loss, "epoch")
        
        # Metrics
        if self.train_metrics:
            for i, metric in enumerate(self.train_metrics):
                value = metric.compute()
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/train_epoch", value)
                self.tb_log_scalar(f"{metric.name}/train_epoch", value, "epoch")
                metric.reset()
    
    def on_validation_epoch_start(self):
        """Called in the validation loop at the beginning of the epoch."""
        self.epoch_step = 0
    
    def validation_step(
        self,
        batch    : Any,
        batch_idx: int,
        *args, **kwargs
    ) -> StepOutput | None:
        """Validation step.

        Args:
            batch: A batch of inputs. It can be a tuple of (input, target, extra).
            batch_idx: A batch index.

        Return:
            Outputs:
                - A single loss image.
                - A dictionary with the first key must be the loss.
                - None, training will skip to the next batch.
        """
        input, target, extra = batch[0], batch[1], batch[2:]
        pred, loss = self.forward_loss(
            input  = input,
            target = target,
            *args, **kwargs
        )
        return {
            "loss"  : loss,
            "input" : input,
            "target": target,
            "pred"  : pred
        }
    
    def validation_step_end(
        self,
        outputs: StepOutput | None,
        *args, **kwargs
    ) -> StepOutput | None:
        """Use this when validating with dp or ddp2 because
        :meth:`validation_step` will operate on only part of the batch. However,
        this is still optional and only needed for things like softmax or NCE
        loss.

        Note:
            If you later switch to ddp or some other mode, this will still be
            called so that you don't have to change your code.
        """
        if not isinstance(outputs, dict):
            return None
        
        # Gather results
        # For DDP strategy, gather outputs from multiple devices.
        if self.trainer.num_devices > 1:
            outputs = self.all_gather(outputs)
        
        loss   = outputs["loss"]    # losses from each device
        input  = outputs["input"]   # images from each device
        target = outputs["target"]  # ground-truths from each device
        pred   = outputs["pred"]    # predictions from each device
        
        # Tensors
        if self.trainer.num_devices > 1:
            input  = input.flatten(start_dim=0,  end_dim=1)
            target = target.flatten(start_dim=0, end_dim=1)
            pred   = pred.flatten(start_dim=0,   end_dim=1)
            
        # Debugging
        epoch = self.current_epoch + 1
        if self.debug \
            and epoch % self.debug.every_n_epochs == 0 \
            and self.epoch_step < self.debug.max_n:
            if self.trainer.is_global_zero:
                self.show_results(
                    input    = input,
                    target   = target,
                    pred     = pred,
                    filepath = self.debug_image_filepath,
                    **self.debug | {
                        "max_n": input[0],
                        "nrow" : input[0],
                    }
                )
            
        # Loss
        loss = loss.mean() if loss is not None else None
        self.ckpt_log_scalar(f"checkpoint/loss/val_step", loss)
        # self.tb_log(f"{loss_tag}", loss, "step")
        
        # Metrics
        if self.val_metrics:
            for i, metric in enumerate(self.val_metrics):
                value = metric(pred, target)
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/val_step", value)
                # self.tb_log(f"{metric.name}/val_step", value, "step")
            
        self.epoch_step += 1
        return {
            "loss"  : loss,
            # "input" : input,
            # "target": target,
            # "pred"  : pred,
        }
    
    def validation_epoch_end(self, outputs: EpochOutput):
        # Loss
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.ckpt_log_scalar(f"checkpoint/loss/val_epoch", loss)
        self.tb_log_scalar(f"loss/val_epoch", loss, "epoch")
        
        # Metrics
        if self.val_metrics:
            for i, metric in enumerate(self.val_metrics):
                value = metric.compute()
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/val_epoch", value)
                self.tb_log_scalar(f"{metric.name}/val_epoch", value, "epoch")
                metric.reset()
    
    def on_test_start(self) -> None:
        """Called at the very beginning of testing."""
        core.create_dirs(
            paths=[
                self.root,
                self.weights_dir,
                self.debug_dir
            ]
        )
    
    def on_test_epoch_start(self):
        """Called in the test loop at the very beginning of the epoch."""
        self.epoch_step = 0
    
    def test_step(
        self,
        batch    : Any,
        batch_idx: int,
        *args, **kwargs
    ) -> StepOutput | None:
        """Test step.

        Args:
            batch: A batch of inputs. It can be a tuple of (input, target, extra).
            batch_idx: A batch index.

        Return:
            Outputs:
                - A single loss image.
                - A dictionary with the first key must be the loss.
                - None, training will skip to the next batch.
        """
        input, target, extra = batch[0], batch[1], batch[2:]
        pred, loss = self.forward_loss(
            input  = input,
            target = target,
            *args, **kwargs
        )
        return {
            "loss"  : loss,
            "input" : input,
            "target": target,
            "pred"  : pred
        }
    
    def test_step_end(
        self,
        outputs: StepOutput | None,
        *args, **kwargs
    ) -> StepOutput | None:
        """Use this when testing with dp or ddp2 because :meth:`test_step` will
        operate on only part of the batch. However, this is still optional and
        only needed for things like softmax or NCE loss.

        Note:
            If you later switch to ddp or some other mode, this will still be
            called so that you don't have to change your code.
        """
        if not isinstance(outputs, dict):
            return None
        
        # Gather results
        # For DDP strategy, gather outputs from multiple devices.
        if self.trainer.num_devices > 1:
            outputs = self.all_gather(outputs)
        
        loss   = outputs["loss"]    # losses from each GPU
        input  = outputs["input"]   # images from each GPU
        target = outputs["target"]  # ground-truths from each GPU
        pred   = outputs["pred"]    # predictions from each GPU
        
        # Tensors
        if self.trainer.num_devices > 1:
            input  = input.flatten(start_dim=0,  end_dim=1)
            target = target.flatten(start_dim=0, end_dim=1)
            pred   = pred.flatten(start_dim=0,   end_dim=1)
        
        # Debugging
        epoch = self.current_epoch + 1
        if self.debug and \
            epoch % self.debug.every_n_epochs == 0 and \
            self.epoch_step < self.debug.max_n:
            if self.trainer.is_global_zero:
                self.show_results(
                    input    = input,
                    target   = target,
                    pred     = pred,
                    filepath = self.debug_image_filepath,
                    **self.debug | {
                        "max_n": input[0],
                        "nrow" : input[0],
                    }
                )
                
        # Loss
        loss = loss.mean() if loss is not None else None
        self.ckpt_log_scalar(f"checkpoint/loss/test_step", loss)
        # self.tb_log(f"loss/test_step", loss, "step")
        
        # Metrics
        if self.test_metrics:
            for i, metric in enumerate(self.test_metrics):
                value = metric(pred, target)
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/test_step", value)
                # self.tb_log(f"{metric.name}/test_step", value, "step")
        
        self.epoch_step += 1
        return {
            "loss"  : loss,
            # "input" : input,
            # "target": target,
            # "pred"  : pred,
        }
    
    def test_epoch_end(self, outputs: EpochOutput):
        # Loss
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.ckpt_log_scalar(f"checkpoint/loss/test_epoch", loss)
        self.tb_log_scalar(f"loss/test_epoch", loss, "epoch")

        # Metrics
        if self.test_metrics:
            for i, metric in enumerate(self.test_metrics):
                value = metric.compute()
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/test_epoch", value)
                self.tb_log_scalar(f"{metric.name}/test_epoch", value, "epoch")
                metric.reset()
    
    def export_to_onnx(
        self,
        input_dims   : Ints     | None = None,
        filepath     : PathType | None = None,
        export_params: bool            = True
    ):
        """Export the model to `onnx` format.

        Args:
            input_dims: Input dimensions. Defaults to None.
            filepath: Path to save the model. If None or empty, then save to
                :attr:`root`. Defaults to None.
            export_params: Should export parameters? Defaults to True.
        """
        # Check filepath
        if filepath in [None, ""]:
            filepath = self.root / f"{self.fullname}.onnx"
        if ".onnx" not in str(filepath):
            filepath = PathType(str(filepath) + ".onnx")
        
        if input_dims is not None:
            input_sample = torch.randn(input_dims)
        else:
            raise ValueError(f"`input_dims` must be defined.")
        
        self.to_onnx(
            file_path     = filepath,
            input_sample  = input_sample,
            export_params = export_params
        )
        
    def export_to_torchscript(
        self,
        input_dims: Ints     | None = None,
        filepath  : PathType | None = None,
        method    : str             = "script"
    ):
        """Export the model to TorchScript format.

        Args:
            input_dims: Input dimensions. Defaults to None.
            filepath: Path to save the model. If None or empty, then save to
                :attr:`root`. Defaults to None.
            method: Whether to use TorchScript's “script” or “trace” method.
                Defaults to “script”.
        """
        # Check filepath
        if filepath in [None, ""]:
            filepath = self.root / f"{self.fullname}.pt"
        if ".pt" not in str(filepath):
            filepath = PathType(str(filepath) + ".pt")
            
        if input_dims is not None:
            input_sample = torch.randn(input_dims)
        else:
            raise ValueError(f"`input_dims` must be defined.")
        
        script = self.to_torchscript(method=method, example_inputs=input_sample)
        torch.jit.save(script, filepath)
    
    @abstractmethod
    def show_results(
        self,
        input        : torch.Tensor | None = None,
        target	     : torch.Tensor | None = None,
        pred		 : torch.Tensor | None = None,
        filepath     : PathType     | None = None,
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
            filepath: A path to save the debug result.
            image_quality: The image quality to be saved. Defaults to 95.
            max_n: Show max n items if :param:`input` has a batch size of more
                than :param:`max_n` items. Defaults to None means show all.
            nrow: The maximum number of items to display in a row. The final
                grid size is (n / nrow, nrow). If None, then the number of items
                in a row will be the same as the number of items in the list.
                Defaults to 8.
            wait_time: Wait for some time (in seconds) to display the figure
                then reset. Defaults to 0.01.
            save: Save debug image. Defaults to False.
            verbose: If True shows the results on the screen. Defaults to False.
        """
        pass
    
    def print_info(self):
        if self.verbose and self.model is not None:
            core.console.log(f"[red]{self.fullname}")
            core.rich.print_table(self.info)
            core.console.log(f"Save indexes: {self.save}")
    
    def tb_log_scalar(
        self,
        tag : str,
        data: Any | None,
        step: str | int = "step"
    ):
        """Log scalar values using tensorboard."""
        if data is None:
            return
        if isinstance(step, str):
            step = self.current_epoch if step == "epoch" else self.global_step
        if self.trainer.is_global_zero and self.logger is not None:
            self.logger.experiment.add_scalar(tag, data, step)
    
    def tb_log_class_metrics(
        self,
        tag : str,
        data: Any | None,
        step: str | int = "step"
    ):
        """Log class metrics using tensorboard."""
        if data is None:
            return
        if self.classlabels is None:
            return
        if isinstance(step, str):
            step = self.current_epoch if step == "epoch" else self.global_step
        if self.trainer.is_global_zero and self.logger is not None:
            for n, a in zip(self.classlabels.names(), data):
                n = f"{tag}/{n}"
                self.logger.experiment.add_scalar(n, a, step)
    
    def ckpt_log_scalar(
        self,
        tag     : str,
        data    : Any | None,
        prog_bar: bool = False
    ):
        """Log for model checkpointing."""
        if data is None:
            return
        if self.trainer.is_global_zero:
            self.log(
                name           = tag,
                value          = data,
                prog_bar       = prog_bar,
                sync_dist      = True,
                rank_zero_only = True
            )

# endregion
