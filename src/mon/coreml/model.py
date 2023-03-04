#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base model class for building machine learning
models.
"""

from __future__ import annotations

__all__ = [
    "Model", "attempt_load", "extract_weight_from_checkpoint", "get_epoch",
    "get_global_step", "get_latest_checkpoint", "intersect_state_dicts",
    "is_parallel", "load_state_dict_from_path", "match_state_dicts",
    "sparsity", "strip_optimizer",
]

import os
from abc import ABC, abstractmethod
from typing import Any

import humps
import lightning.pytorch.utilities.types
import torch
from torch import nn
from torch.nn import parallel

from mon.coreml import data as mdata, layer, loss as mloss, metric as mmetric
from mon.foundation import config, console, error_console, pathlib, rich
from mon.globals import (
    LOSSES, LR_SCHEDULERS, METRICS, ModelPhase, MODELS, OPTIMIZERS,
    ZOO_DIR,
)

StepOutput  = lightning.pytorch.utilities.types.STEP_OUTPUT
EpochOutput = lightning.pytorch.utilities.types.EPOCH_OUTPUT


# region Checkpoint

def extract_weight_from_checkpoint(
    ckpt       : pathlib.Path,
    weight_file: pathlib.Path | None = None,
):
    """Extract and save weights from the checkpoint :attr:`ckpt`.
    
    Args:
        ckpt: A checkpoint file.
        weight_file: A path to save the extracting weights. Defaults to None,
            which saves the weights at the same location as the :param:`ckpt`
            file.
    """
    ckpt = pathlib.Path(ckpt)
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
        weight_file = pathlib.Path(weight_file)
    weight_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, str(weight_file))


def get_epoch(ckpt: pathlib.Path | None) -> int:
    """Get an epoch value stored in a checkpoint file.

    Args:
        ckpt: A checkpoint filepath.
    """
    if ckpt is None:
        return 0
    
    epoch = 0
    ckpt  = pathlib.Path(ckpt)
    if ckpt.is_torch_file():
        ckpt  = torch.load(ckpt)
        epoch = ckpt.get("epoch", 0)
    return epoch


def get_global_step(ckpt: pathlib.Path | None) -> int:
    """Get a global step stored in a checkpoint file.

    Args:
        ckpt: A checkpoint filepath.
    """
    if ckpt is None:
        return 0
    
    global_step = 0
    ckpt = pathlib.Path(ckpt)
    if ckpt.is_torch_file():
        ckpt = torch.load(ckpt)
        global_step = ckpt.get("global_step", 0)
    return global_step


def get_latest_checkpoint(dirpath: pathlib.Path) -> str | None:
    """Get the latest checkpoint (last saved) filepath in a directory.

    Args:
        dirpath: The directory that contains the checkpoints.
    """
    dirpath = pathlib.Path(dirpath)
    ckpt    = dirpath.latest_file()
    if ckpt is None:
        error_console.log(f"[red]Cannot find checkpoint file {dirpath}.")
    return ckpt


# endregion


# region Weight/State Dict

def intersect_state_dicts(x: dict, y: dict, exclude: list = []) -> dict:
    """Find the intersection between two state dictionaries.
    
    Args:
        x: The first state dictionaries.
        y: The second state dictionaries.
        exclude: A list of excluding keys.
    
    Return:
        A dictionary that contains only the keys that are in both dictionaries,
        and whose values have the same shape.
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
    """Filter out unmatched keys btw the model's :attr:`state_dict`, and the
    weights' :attr:`state_dict`. Omitting :param:`exclude` keys.

    Args:
        model_dict: Model :attr:`state_dict`.
        pretrained_dict: Pretrained :attr:`state_dict`.
        exclude: List of excluded keys. Defaults to ().
        
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
    """Strip the optimizer from a saved weight file to finalize the training
    process.
    
    Args:
        weight_file: A PyTorch saved weight filepath.
        new_file: A filepath to save the stripped weights. If :param:`new_file`
            is given, save the weights as a new file. Otherwise, overwrite the
            :param:`weight_file`.
    """
    if not pathlib.Path(weight_file).is_weights_file():
        raise ValueError(
            f"weight_file must be a valid path to a weight file, but got "
            f"{pathlib}."
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
    weights    : dict | pathlib.Path,
    name       : str  | None                = None,
    config     : dict | pathlib.Path | None = None,
    fullname   : str  | None                = None,
    num_classes: int  | None                = None,
    phase      : str                        = "inference",
    strict     : bool                       = True,
    file_name  : str  | None                = None,
    model_dir  : pathlib.Path | None        = ZOO_DIR,
    debugging  : bool                       = True
):
    """Try to create a model from the given configuration and load pretrained
    weights.
    
    Args:
        model: A PyTorch :class:`nn.Module` or a :class:`Model`.
        weights: A weight dictionary or a checkpoint filepath.
        name: A name of the model.
        config: A configuration dictionary or a YAML filepath containing the
            building configuration of the model.
        fullname: An optional fullname of the model, in other words,
            a model's base name + its variant + a training dataset name.
        num_classes: The number of classes, in other words, the final layer's
            output channels.
        phase: The model running phase.
        strict: Defaults to True.
        file_name: Name for the downloaded file. Filename from :param:`path`
        model_dir: Directory in which to save the object. Default to
            :attr:`ZOO_DIR`.
        debugging: If True, stop and raise errors. Otherwise, just return the
            current model.
        
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
    elif isinstance(model, Model) and pathlib.Path(path=weights).is_ckpt_file():
        model = model.load_from_checkpoint(
            checkpoint_path = weights,
            name            = name,
            cfg             = config,
            num_classes     = num_classes,
            phase           = phase,
        )
    # All other cases
    else:
        if isinstance(weights, str | pathlib.Path):
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
    path        : pathlib.Path,
    model_dir   : pathlib.Path | None = None,
    map_location: str | None          = None,
    progress    : bool                = True,
    check_hash  : bool                = False,
    file_name   : str | None          = None,
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
        model_dir = pathlib.Path(model_dir)
    
    path = pathlib.Path(path)
    if not path.is_torch_file() \
        and (model_dir is None or not model_dir.is_dir()):
        raise ValueError(f"'model_dir' must be defined. But got: {model_dir}.")
    
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
    
    Attributes:
        config: A dictionary containing all configurations of the model.
        zoo: A dictionary containing all pretrained weights of the model.
        
    Args:
        config: The model's configuration that is used to build the model.
            Any of:
                - A dictionary.
                - A key in the :attr:`cfgs`.
                - A file name. Ex: 'alexnet.yaml'.
                - A path to a .yaml file. Ex: '../cfgs/alexnet.yaml'.
                - None, define each layer manually.
        hparams: Model's hyperparameters. They are used to change the values of
            :param:`args`. Usually used in grid search or random search during
            training. Defaults to None.
        channels: The first layer's input channel. Defaults to 3.
        num_classes: A number of classes, which is also the last layer's output
            channels. Defaults to None mean it will be determined during model
            parsing.
        classlabels: A :class:`mon.coreml.data.label.ClassLabels` object that
            contains all labels in the dataset. Defaults to None.
        weights: The model's weight. Any of:
            - A state dictionary.
            - A key in the :attr:`zoo`. Ex: 'yolov8x-det-coco'.
            - A path to a weight or ckpt file.
        name: The model's name. Defaults to None mean it will be
            :attr:`self.__class__.__name__`. .
        variant: The model's variant. For example, :param:`name` is 'yolov8' and
            :param:`variant` is 'yolov8x'. Defaults to None mean it will be same
            as :param:`name`.
        fullname: The model's fullname to save the checkpoint or weight. It
            should have the following format:
            {name}/{variant}-{dataset}-{postfix}. Defaults to None mean it will
            be same as :param:`name`.
        root: The root directory of the model. It is used to save the model
            checkpoint during training: {root}/{project}/{fullname}.
        project: A project name. Defaults to None.
        phase: The model's running phase. Defaults to training.
        loss: Loss function for training the model. Defaults to None.
        metrics: A list metrics for validating and testing model. Defaults to
            None.
        optimizers: Optimizer(s) for training model. Defaults to None.
        debug: Debug configs. Defaults to None.
        verbose: Verbosity. Defaults to True.
    """
    
    configs = {}
    zoo     = {}
    
    def __init__(
        self,
        config     : Any                      = None,
        hparams    : dict | None              = None,
        channels   : int                      = 3,
        num_classes: int  | None              = None,
        classlabels: mdata.ClassLabels | None = None,
        weights    : Any                      = None,
        # For saving/loading
        name       : str  | None              = None,
        variant    : str  | None              = None,
        fullname   : str  | None              = None,
        root       : pathlib.Path             = pathlib.Path(),
        project    : str  | None              = None,
        # For training                        
        phase      : ModelPhase | str         = ModelPhase.TRAINING,
        loss       : Any                      = None,
        metrics    : Any                      = None,
        optimizers : Any                      = None,
        debug      : dict | None              = None,
        verbose    : bool                     = True,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.config        = config
        #
        self.hyperparams   = hparams
        self.channels      = channels or self.channels
        self.num_classes   = num_classes
        self.weights       = weights
        self.name          = name     or self.name
        self.variant       = variant  or self.variant
        self.fullname      = fullname
        self.project       = project
        self.root          = root
        self.classlabels   = mdata.ClassLabels.from_value(classlabels) \
                             if classlabels is not None else None
        self.loss          = loss
        self.train_metrics = metrics
        self.val_metrics   = metrics
        self.test_metrics  = metrics
        self.optims        = optimizers
        self.debug         = debug
        self.verbose       = verbose
        self.epoch_step    = 0
        # Define model
        self.model, self.save, self.info = self.parse_model()
        # Load weights
        if self.weights:
            self.load_weight()
        else:
            self.apply(self.init_weight)
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
            fullname = self.name
        self._fullname = fullname
    
    @property
    def root(self) -> pathlib.Path:
        return self._root
    
    @root.setter
    def root(self, root: Any):
        if root is None:
            root = pathlib.Path() / "run"
        else:
            root = pathlib.Path(root)
        self._root = root
        
        if self.project is not None and self.project != "":
            self._root = self._root / self.project
        if self._root.name != self.fullname:
            self._root = self._root / self.fullname
        
        self._debug_dir = self._root / "debug"
        self._ckpt_dir  = self._root / "weight"
    
    @property
    @abstractmethod
    def config_dir(self) -> pathlib.Path:
        pass
    
    @property
    def ckpt_dir(self) -> pathlib.Path:
        if self._ckpt_dir is None:
            self._ckpt_dir = self.root / "weights"
        return self._ckpt_dir
    
    @property
    def debug_dir(self) -> pathlib.Path:
        if self._debug_dir is None:
            self._debug_dir = self.root / "debug"
        return self._debug_dir
    
    @property
    def debug_subdir(self) -> pathlib.Path:
        """The debug subdir path located at: <debug_dir>/<phase>_<epoch>."""
        debug_dir = self.debug_dir / f"{self.phase.value}_{(self.current_epoch + 1):03d}"
        debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir
    
    @property
    def debug_image_filepath(self) -> pathlib.Path:
        """The debug image filepath located at: <debug_dir>/"""
        save_dir = self.debug_subdir \
            if self.debug["save_to_subdir"] \
            else self.debug_dir
        
        return save_dir / f"{self.phase.value}_" \
                          f"{(self.current_epoch + 1):03d}_" \
                          f"{(self.epoch_step + 1):06}.jpg"
    
    @property
    def zoo_dir(self) -> pathlib.Path:
        return ZOO_DIR / self.name
    
    @property
    def config(self) -> dict | None:
        return self._config
    
    @config.setter
    def config(self, config: Any = None):
        variant = None
        if isinstance(config, str) and config in self.configs:
            variant = str(config)
            config     = self.configs[config]
        elif isinstance(config, str) and ".yaml" in config:
            config     = self.config_dir / config
            variant = str(config.stem)
        elif isinstance(config, pathlib.Path):
            variant = str(config.stem)
        elif isinstance(config, dict):
            variant = config.get("variant", None)
        elif config is None:
            pass
        else:
            raise TypeError
            
        self._config  = config.load_config(config=config) if config is not None else None
        self.channels = self._config.get("channels", None)
        self.name     = self._config.get("name", None)
        self.variant  = variant
        
    @property
    def params(self) -> int:
        if self.info is not None:
            params = [i["params"] for i in self.info]
            return sum(params)
        else:
            return 0
    
    @property
    def weights(self) -> pathlib.Path | dict:
        return self._weights
    
    @weights.setter
    def weights(self, weights: Any = None):
        if isinstance(weights, str) and weights in self.zoo:
            weights = pathlib.Path(self.zoo[weights])
        elif isinstance(weights, pathlib.Path):
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
        # metric to be defined with self.<metric>. Here we dynamically
        # add the metric attribute to the class.
        if self._train_metrics:
            for metric in self._train_metrics:
                name = f"train_{metric.name}"
                setattr(self, name, metric)
    
    @property
    def val_metrics(self) -> list[mmetric.Metric] | None:
        return self._val_metrics
    
    @val_metrics.setter
    def val_metrics(self, metrics: Any):
        """Assign val metrics. See Also: :meth:`self.train_metrics()`."""
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
    def test_metrics(self) -> list[mmetric.Metric] | None:
        return self._test_metrics
    
    @test_metrics.setter
    def test_metrics(self, metrics: Any):
        """Assign test metrics. See Also: :meth:`self.train_metrics()`."""
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
    
    def parse_model(self) -> tuple[nn.Sequential, list[int], list[dict]]:
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
        
        Return:
            A Sequential model.
            A list of layer index to save the features during forward pass.
            A list of layer's info for debugging.
        """
        if not isinstance(self.config, dict):
            raise TypeError(
                f"config must be a dictionary, but got {self.config}."
            )
        
        console.log(f"Parsing model from config.")
        
        if "channels" in self.config:
            channels = self.config["channels"]
            if channels != self.channels:
                self.config["channels"] = self.channels
                console.log(
                    f"Overriding model.yaml channels={channels} with "
                    f"num_classes={self.channels}."
                )
        
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
        
        if "backbone" not in self.config and "head" not in self.config:
            raise ValueError("config must contain 'backbone' and 'head' keys.")

        model, save, info = layer.parse_model(
            d       = self.config,
            ch      = [self.channels],
            hparams = self.hyperparams,
        )
        return model, save, info
    
    @abstractmethod
    def init_weight(self, model: nn.Module):
        """Initialize model's weight."""
        pass
    
    def load_weight(self):
        """Load weights. It only loads the intersection layers of matching keys
        and shapes between the current model and weights.
        """
        if isinstance(self.weights, dict | pathlib.Path):
            self.zoo_dir.mkdir(parents=True, exist_ok=True)
            self.model = attempt_load(
                model     = self.model,
                weights   = self.weights,
                strict    = False,
                model_dir = self.zoo_dir,
            )
            if self.verbose:
                console.log(f"Load weight from: {self.weights}!")
        else:
            error_console.log(
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
                raise ValueError(f"optimizer must be defined.")
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
                    raise ValueError(f"scheduler must be defined.")
                if isinstance(scheduler,  dict):
                    scheduler = LR_SCHEDULERS.build(
                        optimizer = optim["optimizer"],
                        config= scheduler
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
            input: An input of shape NCHW.
            target: A ground-truth of shape NCHW. Defaults to None.
            
        Return:
            Predictions and loss value.
        """
        pred = self.forward(input=input, *args, **kwargs)
        # features = None
        if isinstance(pred, (list, tuple)):
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
            input: An input of shape NCHW.
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
            input: An input of shape NCHW.
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
    
    # Training
    
    def on_fit_start(self):
        """Called at the beginning of fit."""
        for path in [self.root, self.ckpt_dir, self.debug_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    def on_train_epoch_start(self):
        """Called in the training loop at the beginning of the epoch."""
        self.epoch_step = 0
    
    def training_step(
        self,
        batch        : Any,
        batch_idx    : int,
        optimizer_idx: int,
        *args, **kwargs
    ) -> StepOutput | None:
        """Here you compute and return the training loss, and some additional
        metrics for e.g. the progress bar or logger.

        Args:
            batch: The output of :class:`~torch.utils.data.DataLoader`. It can
                be a tensor, tuple or list.
            batch_idx: An integer displaying index of this batch.
            optimizer_idx: When using multiple optimizers, this argument will
                also be present.
            
        Return:
            Any of:
                - The loss tensor.
                - A dictionary. Can include any keys, but must include the key
                  'loss'.
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
    
    def training_step_end(self, step_output: StepOutput) -> StepOutput:
        """Use this when training with dp because :meth:`training_step` will
        operate on only part of the batch. However, this is still optional and
        only needed for things like softmax or NCE loss.
        
        Note:
            If you later switch to ddp or some other mode, this will still be
            called so that you don't have to change your code.
        
        Args:
            step_output: What you return in `training_step` for each batch part.
        """
        if not isinstance(step_output,  dict):
            return step_output
        
        # Gather results
        # For DDP strategy, gather outputs from multiple devices
        if self.trainer.num_devices > 1:
            step_output = self.all_gather(step_output)
        
        loss   = step_output["loss"]    # losses from each device
        input  = step_output["input"]   # images from each device
        target = step_output["target"]  # ground-truths from each device
        pred   = step_output["pred"]    # predictions from each device
        
        # Tensors
        if self.trainer.num_devices > 1:
            input  = input.flatten(start_dim=0, end_dim=1)
            target = target.flatten(start_dim=0, end_dim=1)
            pred   = pred.flatten(start_dim=0, end_dim=1)
        
        # Loss
        loss = loss.mean() if loss is not None else None
        self.ckpt_log_scalar(f"checkpoint/loss/train_step", loss)
        # self.tb_log(f"{loss_tag}", loss, "step")
        
        # Metrics
        if self.train_metrics:
            for i, metric in enumerate(self.train_metrics):
                value = metric(pred, target)
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/train_step", value, True)
                # self.tb_log(f"{metric.name}/train_step", value, "step")
        
        self.epoch_step += 1
        return {
            "loss": loss,
            # "input" : input,
            # "target": target,
            # "pred"  : pred,
        }
    
    def training_epoch_end(self, outputs: EpochOutput):
        """Called at the end of the training epoch with the outputs of all
        training steps. Use this in case you need to do something with all the
        outputs returned by :meth:`training_step`.
        
        Args:
            outputs: A list of outputs you defined in :meth:`training_step`.
        """
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
        batch         : Any,
        batch_idx     : int,
        dataloader_idx: int,
        *args, **kwargs
    ) -> StepOutput | None:
        """Operates on a single batch of data from the validation set. In this
        step, you might generate examples or calculate anything of interest like
        accuracy.
        
        Args:
            batch: The output of :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple val dataloaders used).
        
        Return:
            - Any object or value.
            - None, validation will skip to the next batch.
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
        step_output: StepOutput,
        *args, **kwargs
    ) -> StepOutput | None:
        """Use this when validating with dp because :meth:`validation_step` will
        operate on only part of the batch. However, this is still optional and
        only needed for things like softmax or NCE loss.

        Note:
            If you later switch to ddp or some other mode, this will still be
            called so that you don't have to change your code.
        
        Return:
            None or anything
        """
        if not isinstance(step_output,  dict):
            return None
        
        # Gather results
        # For DDP strategy, gather outputs from multiple devices.
        if self.trainer.num_devices > 1:
            step_output = self.all_gather(step_output)
        
        loss   = step_output["loss"]  # losses from each device
        input  = step_output["input"]  # images from each device
        target = step_output["target"]  # ground-truths from each device
        pred   = step_output["pred"]  # predictions from each device
        
        # Tensors
        if self.trainer.num_devices > 1:
            input  = input.flatten(start_dim=0, end_dim=1)
            target = target.flatten(start_dim=0, end_dim=1)
            pred   = pred.flatten(start_dim=0, end_dim=1)
        
        # Debugging
        epoch = self.current_epoch + 1
        if self.debug \
            and epoch % self.debug["every_n_epochs"] == 0 \
            and self.epoch_step < self.debug["max_n"]:
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
            "loss": loss,
            # "input" : input,
            # "target": target,
            # "pred"  : pred,
        }
    
    def validation_epoch_end(self, outputs: EpochOutput | list[EpochOutput]):
        """Called at the end of the validation epoch with the outputs of all
        validation steps.
        
        Args:
            outputs: A list of outputs you defined in :meth:`validation_step`,
                or if there are multiple dataloaders, a list containing a list
                of outputs for each dataloader.
        
        Return:
             None.
        """
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
        for path in [self.root, self.ckpt_dir, self.debug_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    def on_test_epoch_start(self):
        """Called in the test loop at the very beginning of the epoch."""
        self.epoch_step = 0
    
    def test_step(
        self,
        batch        : Any,
        batch_idx    : int,
        dataloader_id: int,
        *args, **kwargs
    ) -> StepOutput | None:
        """Operates on a single batch of data from the test set. In this step
        you'd normally generate examples or calculate anything of interest such
        as accuracy.

        Args:
            batch: The output of your :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_id: The index of the dataloader that produced this batch.
                (only if multiple test dataloaders used).

        Return:
            Any of:
                - Any object or value.
                - None, testing will skip to the next batch.
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
        step_output: StepOutput,
        *args, **kwargs
    ) -> StepOutput | None:
        """Use this when testing with DP because :meth:`test_step` will operate
        on only part of the batch. However, this is still optional and only
        needed for things like softmax or NCE loss.

        Note:
            If you later switch to ddp or some other mode, this will still be
            called so that you don't have to change your code.
        
        Args:
            step_output: What you return in :meth:`test_step` for each batch
            part.
            
        Return:
            None or anything
        """
        if not isinstance(step_output,  dict):
            return step_output
        
        # Gather results
        # For DDP strategy, gather outputs from multiple devices.
        if self.trainer.num_devices > 1:
            step_output = self.all_gather(step_output)
        
        loss   = step_output["loss"]    # losses from each GPU
        input  = step_output["input"]   # images from each GPU
        target = step_output["target"]  # ground-truths from each GPU
        pred   = step_output["pred"]    # predictions from each GPU
        
        # Tensors
        if self.trainer.num_devices > 1:
            input  = input.flatten(start_dim=0, end_dim=1)
            target = target.flatten(start_dim=0, end_dim=1)
            pred   = pred.flatten(start_dim=0, end_dim=1)
        
        # Debugging
        epoch = self.current_epoch + 1
        if self.debug and \
            epoch % self.debug["every_n_epochs"] == 0 and \
            self.epoch_step < self.debug["max_n"]:
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
            "loss": loss,
            # "input" : input,
            # "target": target,
            # "pred"  : pred,
        }
    
    def test_epoch_end(self, outputs: EpochOutput | list[EpochOutput]):
        """Called at the end of a test epoch with the output of all test steps.
        
        Args:
            outputs: A list of outputs you defined in :meth:`test_step_end`, or
                if there are multiple dataloaders, a list containing a list of
                outputs for each dataloader.
        """
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
        input_dims   : list[int]    | None = None,
        filepath     : pathlib.Path | None = None,
        export_params: bool = True
    ):
        """Export the model to `onnx` format.

        Args:
            input_dims: Input dimensions in CHW format. Defaults to None.
            filepath: Path to save the model. If None or empty, then save to
                :attr:`root`. Defaults to None.
            export_params: Should export parameters? Defaults to True.
        """
        # Check filepath
        if filepath in [None, ""]:
            filepath = self.root / f"{self.fullname}.onnx"
        if ".onnx" not in str(filepath):
            filepath = pathlib.Path(str(filepath) + ".onnx")
        
        if input_dims is not None:
            input_sample = torch.randn(input_dims)
        else:
            raise ValueError(f"input_dims must be defined.")
        
        self.to_onnx(
            file_path     = filepath,
            input_sample  = input_sample,
            export_params = export_params
        )
    
    def export_to_torchscript(
        self,
        input_dims: list[int]    | None = None,
        filepath  : pathlib.Path | None = None,
        method    : str = "script"
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
            filepath = pathlib.Path(str(filepath) + ".pt")
        
        if input_dims is not None:
            input_sample = torch.randn(input_dims)
        else:
            raise ValueError(f"'input_dims' must be defined.")
        
        script = self.to_torchscript(method=method, example_inputs=input_sample)
        torch.jit.save(script, filepath)
    
    @abstractmethod
    def show_results(
        self,
        input        : torch.Tensor | None = None,
        target       : torch.Tensor | None = None,
        pred         : torch.Tensor | None = None,
        filepath     : pathlib.Path | None = None,
        image_quality: int                 = 95,
        max_n        : int | None          = 8,
        nrow         : int | None          = 8,
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
            nrow: The maximum number of items to display in a row. Defaults to
                8.
            wait_time: Wait for some time (in seconds) to display the figure
                then reset. Defaults to 0.01.
            save: Save debug image. Defaults to False.
            verbose: If True shows the results on the screen. Defaults to False.
        """
        pass
    
    def print_info(self):
        if self.verbose and self.model is not None:
            console.log(f"[red]{self.fullname}")
            rich.print_table(self.info)
            console.log(f"Save indexes: {self.save}")
    
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
