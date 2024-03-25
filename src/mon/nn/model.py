#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for deep learning model.
"""

from __future__ import annotations

__all__ = [
    "Model",
    "check_weights_loaded",
    "get_epoch_from_checkpoint",
    "get_global_step_from_checkpoint",
    "get_latest_checkpoint",
    "load_state_dict",
    "load_weights",
    "parse_model_fullname",
]

import copy
from abc import ABC, abstractmethod
from typing import Any, Sequence
from urllib.parse import urlparse  # noqa: F401

import humps
import lightning.pytorch.utilities.types
import torch.hub
from thop.profile import *
from torch import nn

from mon import core
from mon.core import _callable
from mon.globals import LOSSES, LR_SCHEDULERS, METRICS, OPTIMIZERS, Scheme, Task, ZOO_DIR
from mon.nn import loss as L, metric as M

console       = core.console
error_console = core.error_console
StepOutput    = lightning.pytorch.utilities.types.STEP_OUTPUT
EpochOutput   = Any  # lightning.pytorch.utilities.types.EPOCH_OUTPUT


# region Checkpoint

def get_epoch_from_checkpoint(ckpt: core.Path | None) -> int:
    """Get an epoch value stored in a checkpoint file.

	Args:
		ckpt: A checkpoint file path.
	"""
    if ckpt is None:
        return 0
    else:
        epoch = 0
        ckpt  = core.Path(ckpt)
        if ckpt.is_torch_file():
            ckpt  = torch.load(ckpt)
            epoch = ckpt.get("epoch", 0)
        return epoch


def get_global_step_from_checkpoint(ckpt: core.Path | None) -> int:
    """Get a global step stored in a checkpoint file.
	
	Args:
		ckpt: A checkpoint file path.
	"""
    if ckpt is None:
        return 0
    else:
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


# region Weights

def load_state_dict(
    model    : nn.Module,
    weights  : dict | str | core.Path,
    overwrite: bool = False,
) -> dict:
    """Load state dict from the given :param:`weights`. If :param:`weights`
    contains a URL, download it.
    """
    path = None
    map  = {}
    
    # Obtain weight's path
    if isinstance(weights, dict):
        assert "path" in weights and "url" in weights
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
    
    # Load state dict
    weights_state_dict = torch.load(str(path), map_location=model.device)
    weights_state_dict = weights_state_dict.get("state_dict", weights_state_dict)
    model_state_dict   = copy.deepcopy(model.state_dict())
    new_state_dict     = {}

    for k, v in weights_state_dict.items():
        replace = False
        for k1, k2 in map.items():
            kr = k.replace(k1, k2)
            if kr in model_state_dict and not replace:
                new_state_dict[kr] = v
                replace = True
        if not replace:
            new_state_dict[k] = v
    
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


def check_weights_loaded(
    old_state_dict: dict,
    new_state_dict: dict,
    verbose       : bool = False,
) -> bool:
    # Manually check if weights have been loaded
    if verbose:
        console.log(f"Checking layer's weights")
    for k, v in new_state_dict.items():
        new_state_dict[k] = float(torch.sum(new_state_dict[k] - old_state_dict[k]))
        if verbose:
            if new_state_dict[k] == 0.0:
                console.log(f"{k}: ❌")
            else:
                console.log(f"{k}: ✅")
    
# endregion


# region Model

def parse_model_fullname(
    name  : str,
    data  : str | None,
    suffix: str | None = None,
) -> str:
    """Parse model's fullname from given components as ``name-data-suffix``.
    
    Args:
        name: The model's name.
        data: The dataset's name.
        suffix: The suffix of the model's name.
    """
    assert name not in [None, ""], f"Model's name must be given."
    fullname = name
    if data not in [None, ""]:
        fullname = f"{fullname}-{data}"
    if suffix not in [None, ""]:
        _fullname = core.snakecase(fullname)
        _suffix   = core.snakecase(suffix)
        if _suffix not in _fullname:
            fullname = f"{fullname}-{core.kebabize(suffix)}"
    return fullname


class Model(lightning.LightningModule, ABC):
    """The base class for all machine learning models.
    
    Attributes:
        _tasks: A list of tasks that the model can perform.
        _zoo: A :class:`dict` containing all pretrained weights of the model.
        
    Args:
        name: The model's name. Default: ``None`` mean it will be
            :attr:`self.__class__.__name__`. .
        root: The root directory of the model. It is used to save the model
            checkpoint during training: {root}/{fullname}.
        fullname: The model's fullname to save the checkpoint or weights. It
            should have the following format: {name}-{dataset}-{suffix}.
            Default: ``None`` mean it will be the same as :param:`name`.
        channels: The first layer's input channel. Default: ``3`` for RGB image.
        num_classes: A number of classes, which is also the last layer's output
            channels. Default: ``None`` mean it will be determined during model
            parsing.
        classlabels: A :class:`mon.nn.data.label.ClassLabels` object that
            contains all labels in the dataset. Default: ``None``.
        weights: The model's weights. Any of:
            - A state :class:`dict`.
            - A key in the :attr:`zoo`. Ex: 'yolov8x-det-coco'.
            - A path to an ``.pt``, ``.pth``, or ``.ckpt`` file.
        loss: Loss function for training the model. Default: ``None``.
        metrics: A list metrics for training, validating and testing model.
            Default: ``None``.
        optimizers: Optimizer(s) for a training model. Default: ``None``.
        verbose: Verbosity. Default: ``True``.
    
    Example:
        LOADING WEIGHTS

        Case 01: Pre-define the weights file in ``zoo`` directory. Pre-define
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
            >>>     weights="imagenet",
            >>> )

        Case 02: Define the full path to an ``.pt``, ``.pth``, or ``.ckpt`` file.

            >>> model = Model(
            >>>     weights="home/workspace/.../vgg19-imagenet.pth",
            >>> )
    """
    
    _tasks : list[Task]   = []  # A list of tasks that the model can perform.
    _scheme: list[Scheme] = []  # A list of learning schemes that the model can perform.
    _zoo   : dict         = {}  # A dictionary containing all pretrained weights of the model.
    
    def __init__(
        self,
        # For saving/loading
        name       : str | None = None,
        root       : core.Path  = core.Path(),
        fullname   : str | None = None,
        # For model architecture
        channels   : int        = 3,
        num_classes: int | None = None,
        classlabels: Any        = None,
        weights    : Any        = None,
        # For training          
        loss       : Any        = None,
        metrics    : Any        = None,
        optimizers : Any        = None,
        # Misc
        verbose    : bool       = True,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.verbose    = verbose
        # For saving/loading
        self._name      = None
        self._fullname  = None  # root/fullname
        self._root      = None  # root/fullname
        self._debug_dir = None
        self._ckpt_dir  = None
        self._set_name(name=name)
        self._set_fullname(fullname=fullname)
        self._set_root(root=root)
        # For model architecture
        self._channels    = channels
        self._num_classes = num_classes
        self.classlabels  = classlabels
        self._weights     = None
        self._set_weights(weights=weights)
        # For training
        self._loss          = None
        self._train_metrics = None
        self._val_metrics   = None
        self._test_metrics  = None
        self.optims         = optimizers
        self._set_loss(loss=loss)
        self._set_train_metrics(metrics=metrics)
        self._set_val_metrics(metrics=metrics)
        self._set_test_metrics(metrics=metrics)
    
    # region Properties
    
    # region Model Metadata
    
    @classmethod
    @property
    def tasks(cls) -> list[Task]:
        return cls._tasks
    
    @classmethod
    @property
    def schemes(cls) -> list[Scheme]:
        return cls._schemes
    
    @classmethod
    @property
    def zoo(cls) -> dict:
        return cls._zoo
    
    # endregion
    
    # region Saving/Loading Properties
    
    @property
    def name(self) -> str:
        """Return the model's name."""
        return self._name
    
    def _set_name(self, name: str | None):
        """Specify the model's name. This value should only be defined once."""
        if name is None or name == "":
            name = humps.kebabize(self.__class__.__name__).lower()
        self._name = name
    
    @property
    def fullname(self) -> str:
        """Return the model's fullname = name-suffix"""
        return self._fullname
    
    def _set_fullname(self, fullname: str | None):
        """Specify the model's fullname. This value should only be defined once."""
        self._fullname = fullname if fullname not in [None, ""] else self.name
    
    @property
    def root(self) -> core.Path:
        return self._root
    
    def _set_root(self, root: Any):
        root = core.Path(root)
        if root.name != self.fullname:
            root /= self.fullname
        self._root      = root
        self._debug_dir = root / "debug"
        self._ckpt_dir  = root / "weights"
    
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
    def zoo_dir(self) -> core.Path:
        """Specify the path of the model's pretrained weights directory in
        :attr:`ZOO_DIR`.
        """
        return ZOO_DIR / self.name
    
    # endregion
    
    # region Model Architecture Properties
    
    @property
    def channels(self) -> int:
        return self._channels
    
    @property
    def num_classes(self) -> int | None:
        return self._num_classes
    
    @property
    def weights(self) -> core.Path | dict:
        return self._weights
    
    def _set_weights(self, weights: Any):
        if isinstance(weights, str):
            if core.Path(weights).is_weights_file():
                weights = core.Path(weights)
            elif weights in self._zoo:
                weights         = self._zoo[weights]
                weights["path"] = self.zoo_dir / weights.get("path", "")
                num_classes     = getattr(weights, "num_classes", None)
                if num_classes is not None and num_classes != self.num_classes:
                    self._num_classes = num_classes
                    console.log(f"Overriding :attr:`num_classes` with {num_classes}.")
            else:
                error_console.log(f"The key ``'{weights}'`` has not been defined in :attr:`zoo`.")
                weights = None
        elif isinstance(weights, core.Path):
            assert weights.is_weights_file(), \
                (f":param:`weights` must be a valid path to a weight file, "
                 f"but got {weights}.")
        elif isinstance(weights, dict):
            pass
        self._weights = weights or self._weights
        
        # endregion
    
    # endregion
    
    # region Training Properties
    
    @property
    def predicting(self) -> bool:
        """Return ``True`` if the model is in predicting mode (not eval).
        
        This property is needed because, while in ``'validation'`` mode,
        :attr:`training` is also set to ``False``, so using
        ``self.training == False`` does not work.
        
        True ``'predicting'`` mode happens when :attr:`_trainer` is ``None``,
        i.e., not being handled by :class:`lightning.Trainer`.
        """
        return True if not self.training and getattr(self, "_trainer", None) is None else None
    
    @property
    def loss(self) -> L.Loss | None:
        """Return the model's loss functions."""
        return self._loss
    
    def _set_loss(self, loss: Any):
        """Specify the model's loss functions. This value should only be
        defined once.
        """
        if isinstance(loss, L.Loss):
            self._loss = loss
        elif isinstance(loss, str):
            self._loss = LOSSES.build(name=loss)
        elif isinstance(loss, dict):
            self._loss = LOSSES.build(config=loss)
        else:
            self._loss = None
        
        if self._loss:
            self._loss.requires_grad = True
    
    @property
    def train_metrics(self) -> list[M.Metric] | None:
        """Return the training metrics."""
        return self._train_metrics
    
    def _set_train_metrics(self, metrics: Any):
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
        # This is a simple hack since LightningModule needs the metric to be
        # defined with self.<metric>. So, here we dynamically add the metric
        # attribute to the class.
        if self._train_metrics:
            for metric in self._train_metrics:
                name = f"train/{metric.name}"
                setattr(self, name, metric)
    
    @property
    def val_metrics(self) -> list[M.Metric] | None:
        """Return the validation metrics."""
        return self._val_metrics
    
    def _set_val_metrics(self, metrics: Any):
        """Assign val metrics. Similar to: :meth:`self._set_train_metrics()`."""
        if isinstance(metrics, dict) and "val" in metrics:
            metrics = metrics.get("val", metrics)
        
        self._val_metrics = self.create_metrics(metrics)
        if self._val_metrics:
            for metric in self._val_metrics:
                name = f"val/{metric.name}"
                setattr(self, name, metric)
    
    @property
    def test_metrics(self) -> list[M.Metric] | None:
        """Return the testing metrics."""
        return self._test_metrics
    
    def _set_test_metrics(self, metrics: Any):
        """Assign test metrics. Similar to: :meth:`self._set_train_metrics()`."""
        if isinstance(metrics, dict) and "test" in metrics:
            metrics = metrics.get("test", metrics)
        
        self._test_metrics = self.create_metrics(metrics)
        if self._test_metrics:
            for metric in self._test_metrics:
                name = f"test/{metric.name}"
                setattr(self, name, metric)
    
    @staticmethod
    def create_metrics(metrics: Any):
        """Create metrics."""
        if isinstance(metrics, M.Metric):
            if getattr(metrics, "name", None) is None:
                metrics.name = humps.depascalize(humps.pascalize(metrics.__class__.__name__))
            return [metrics]
        elif isinstance(metrics, dict):
            return [METRICS.build(config=metrics)]
        elif isinstance(metrics, list | tuple):
            return [METRICS.build(config=m) if isinstance(m, dict) else m for m in metrics]
        else:
            return None
    
    # endregion
    
    # endregion
    
    # region Initialize Model
    
    def _create_dir(self):
        """Create directories before training begins."""
        for path in [self.root, self.ckpt_dir, self.debug_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def _init_weights(self, model: nn.Module):
        """Initialize the model's weights."""
        pass
    
    def load_weights(self, weights: Any = None, overwrite: bool = False):
        """Load weights. It only loads the intersection layers of matching keys
        and shapes between the current model and weights.
        """
        self._set_weights(weights=weights)
        
        # Get the state_dict
        state_dict = None
        if isinstance(self.weights, core.Path | str) and core.Path(self.weights).is_weights_file():
            self.zoo_dir.mkdir(parents=True, exist_ok=True)
            state_dict = load_state_dict(model=self, weights=self.weights)
            state_dict = getattr(state_dict, "state_dict", state_dict)
        elif isinstance(self.weights, dict):
            if "path" in self.weights:
                path = self.weights["path"]
                if core.Path(path).is_weights_file():
                    state_dict = load_state_dict(model=self, weights=self.weights)
                    state_dict = getattr(state_dict, "state_dict", state_dict)
            else:
                state_dict = getattr(self.weights, "state_dict", self.weights)
        else:
            error_console.log(f"[yellow]Cannot load from weights from: {self.weights}!")
        
        if state_dict is not None:
            self.load_state_dict(state_dict=state_dict)
            if self.verbose:
                console.log(f"Load model's weights from: {self.weights}!")
        
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
                raise ValueError(f":param:`optimizer` must be defined.")
            if isinstance(optimizer,  dict):
                optimizer = OPTIMIZERS.build(net=self, config=optimizer)
            optim["optimizer"] = optimizer
            
            # Define learning rate scheduler
            lr_scheduler = optim.get("lr_scheduler", None)
            if "lr_scheduler" in optim and lr_scheduler is None:
                optim.pop("lr_scheduler")
            elif lr_scheduler is not None and isinstance(lr_scheduler, dict):
                scheduler = lr_scheduler.get("scheduler", None)
                if scheduler is None:
                    raise ValueError(f":param:`scheduler` must be defined.")
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
        if isinstance(optims, list | tuple) and len(optims) == 1:
            optims = optims[0]
        self.optims = optims
        return self.optims
    
    # endregion
    
    # region Forward Pass
    
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
        pred = pred[-1] if isinstance(pred, list | tuple) else pred
        loss = self.loss(pred, target)
        return pred, loss
        
    @abstractmethod
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        """Forward pass. This is the primary :meth:`forward` function of the
        model. It supports augmented inference. In this function, we perform
        test-time augmentation and pass the transformed input to
        :meth:`forward_once()`.

        Args:
            input: An input of shape :math:`[N, C, H, W]`.
            augment: If ``True``, perform test-time augmentation. Usually used
                in predicting phase. Default: ``False``.
            profile: If ``True``, measure processing time. Usually used in
                predicting phase. Default: ``False``.
            out_index: If the model produces multiple outputs, return the one
                with the index :param:`out_index`. Usually used in predicting
                phase. Default: ``-1`` means the last one.
            
        Return:
            Predictions.
        """
        pass
    
    # endregion
    
    # region Training
    def fit_one(self, *args, **kwargs) -> Any:
        """Train the model with a single sample. This method is used for any
        learning scheme performed on one single instance such as online learning,
        zero-shot learning, one-shot learning, etc.
        
        Note:
            In order to use this method, the model must implement the optimizer
            and/or scheduler.
        
        Returns:
            Return ``None`` by default if the model does not support this feature.
        """
        error_console.log(f"[yellow]The {self.__class__.__name__} does not support this feature.")
        return None
    
    def on_fit_start(self):
        """Called at the beginning of fit."""
        self._create_dir()

    def training_step(self, batch: Any, batch_idx: int, *args, **kwargs) -> StepOutput | None:
        """Here you compute and return the training loss, and some additional
        metrics for e.g., the progress bar or logger.

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
        
        # Log
        log_dict = {
            f"step"      : self.current_epoch,
            f"train/loss": loss,
        }
        if self.train_metrics:
            for i, metric in enumerate(self.train_metrics):
                log_dict[f"train/{metric.name}"] = metric(pred, target)
        self.log_dict(
            dictionary     = log_dict,
            prog_bar       = False,
            logger         = True,
            on_step        = False,
            on_epoch       = True,
            sync_dist      = True,
            rank_zero_only = False,
        )

        return loss

    def on_train_epoch_end(self):
        """Called in the training loop at the very end of the epoch."""
        if self.train_metrics:
            for i, metric in enumerate(self.train_metrics):
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

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs) -> StepOutput | None:
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
        
        # Log
        log_dict = {
            f"step"    : self.current_epoch,
            f"val/loss": loss,
        }
        if self.val_metrics:
            for i, metric in enumerate(self.val_metrics):
                log_dict[f"val/{metric.name}"] = metric(pred, target)
        self.log_dict(
            dictionary     = log_dict,
            prog_bar       = False,
            logger         = True,
            on_step        = False,
            on_epoch       = True,
            sync_dist      = True,
            rank_zero_only = False,
        )
        
        if self._should_save_image():
            self._log_image(input, target, pred, self.current_epoch, self.global_step)
        
        return loss

    def on_validation_epoch_end(self):
        """Called in the validation loop at the very end of the epoch."""
        if self.val_metrics:
            for i, metric in enumerate(self.val_metrics):
                metric.reset()

    def on_test_start(self) -> None:
        """Called at the very beginning of testing."""
        self._create_dir()

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs) -> StepOutput | None:
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
        
        # Log
        log_dict = {
            f"step"     : self.current_epoch,
            f"test/loss": loss,
        }
        if self.test_metrics:
            for i, metric in enumerate(self.test_metrics):
                log_dict[f"test/{metric.name}"] = metric(pred, target)
        self.log_dict(
            dictionary     = log_dict,
            prog_bar       = False,
            logger         = True,
            on_step        = False,
            on_epoch       = True,
            sync_dist      = True,
            rank_zero_only = False,
        )
        
        return loss
    
    def on_test_epoch_end(self):
        """Called in the test loop at the very end of the epoch."""
        if self.test_metrics:
            for i, metric in enumerate(self.test_metrics):
                metric.reset()
    
    # endregion
    
    # region Exporting
    
    def export_to_onnx(
        self,
        input_dims   : list[int] | None = None,
        file_path    : core.Path | None = None,
        export_params: bool             = True
    ):
        """Export the model to ``onnx`` format.

        Args:
            input_dims: Input dimensions in :math:`[C, H, W]` format.
                Default: ``None``.
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
            raise ValueError(f":param:`input_dims` must be defined.")
        
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
            file_path: Path to save the model. If ``None`` or empty, then save
                to :attr:`root`. Default: ``None``.
            method: Whether to use TorchScript's `''script''` or ``'trace'``
                method. Default: ``'script'``.
        """
        # Check filepath
        if file_path in [None, ""]:
            file_path = self.root / f"{self.fullname}.pt"
        if ".pt" not in str(file_path):
            file_path = core.Path(str(file_path) + ".pt")
        
        if input_dims is not None:
            input_sample = torch.randn(input_dims)
        else:
            raise ValueError(f":param:`input_dims` must be defined.")
        
        script = self.to_torchscript(method=method, example_inputs=input_sample)
        torch.jit.save(script, file_path)
    
    # endregion
    
    # region Logging
    
    def _should_save_image(self) -> bool:
        return (
            self.trainer.is_global_zero
            and self.trainer.log_image_every_n_epochs > 0
            and self.current_epoch % self.trainer.log_image_every_n_epochs == 0
        )
    
    def _log_image(
        self,
        input : torch.Tensor,
        target: torch.Tensor,
        pred  : torch.Tensor | Sequence[torch.Tensor],
        epoch : int,
        step  : int,
    ):
        pass

    # endregion
    
# endregion
