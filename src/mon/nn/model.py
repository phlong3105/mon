#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for deep learning model.
"""

from __future__ import annotations

__all__ = [
    "ExtraModel",
    "Model",
    "get_epoch_from_checkpoint",
    "get_global_step_from_checkpoint",
    "get_latest_checkpoint",
    "load_state_dict",
    "load_weights",
    "parse_model_fullname",
]

from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse  # noqa: F401

import humps
import lightning.pytorch.utilities.types
import torch.hub
from thop.profile import *
from torch import nn

from mon import core
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
    ckpts   = dirpath.files(recursive=True)
    ckpts   = [ckpt for ckpt in ckpts if ckpt.is_torch_file()]
    ckpts   = sorted(ckpts, key=lambda x: x.stat().st_mtime, reverse=True)
    ckpt    = ckpts[0] if ckpts else None
    if ckpt is None:
        error_console.log(f"[red]Cannot find checkpoint file {dirpath}.")
    return ckpt

# endregion


# region Weights

def load_state_dict(
    model       : nn.Module,
    weights     : dict | str | core.Path,
    weights_only: bool = False,
    overwrite   : bool = False,
) -> dict:
    """Load state dict from the given :param:`weights`. If :param:`weights`
    contains a URL, download it.
    """
    path = None
    url  = None

    # Obtain weight's path
    if isinstance(weights, dict):
        path = weights.get("path", path)
        url  = weights.get("url",  url)
        if path and url:
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
    weights_state_dict = torch.load(str(path), weights_only=weights_only, map_location=model.device)
    '''
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
    '''
    return weights_state_dict


def load_weights(
    model       : nn.Module,
    weights     : dict | str | core.Path,
    weights_only: bool = True,
    overwrite   : bool = False,
) -> nn.Module:
    """Load weights to model."""
    model_state_dict = load_state_dict(model=model, weights=weights, weights_only=True)
    model.load_state_dict(model_state_dict)
    return model

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
        fullname = f"{fullname}_{data}"
    if suffix not in [None, ""]:
        _fullname = core.snakecase(fullname)
        _suffix   = core.snakecase(suffix)
        if _suffix not in _fullname:
            fullname = f"{fullname}_{core.kebabize(suffix)}"
    return fullname


class Model(lightning.LightningModule, ABC):
    """The base class for all machine learning models.
    
    Attributes:
        arch: The model's architecture. Default: ``None`` mean it will be
            :attr:`self.__class__.__name__`.
        tasks: A list of tasks that the model can perform.
        schemes: A list of learning schemes that the model can perform.
        zoo: A :class:`dict` containing all pretrained weights of the model.
        zoo_dir: The directory containing all pretrained weights of the model.
        
    Args:
        name: The model's name. Default: ``None`` mean it will be
            :attr:`self.__class__.__name__`.
        fullname: The model's fullname to save the checkpoint or weights. It
            should have the following format: {name}-{dataset}-{suffix}.
            Default: ``None`` mean it will be the same as :param:`name`.
        root: The root directory of the model. It is used to save the model
            checkpoint during training: {root}/{fullname}.
        in_channels: The first layer's input channel. Default: ``3`` for RGB image.
        out_channels: The last layer's output channels (number of classes).
            Default: ``None`` mean it will be determined during model parsing.
        num_classes: Alias to :param:`out_channels`, but for classification tasks.
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
    
    arch   : str          = ""       # The model's architecture.
    tasks  : list[Task]   = []       # A list of tasks that the model can perform.
    schemes: list[Scheme] = []       # A list of learning schemes that the model can perform.
    zoo    : dict         = {}       # A dictionary containing all pretrained weights of the model.
    zoo_dir: core.Path    = ZOO_DIR  # The directory containing all pretrained weights of the model.
    
    def __init__(
        self,
        # For saving/loading
        name        : str | None = None,
        fullname    : str | None = None,
        root        : core.Path  = core.Path(),
        # For model architecture
        in_channels : int        = 3,
        out_channels: int | None = None,
        num_classes : int | None = None,
        weights     : Any        = None,
        # For training          
        loss        : Any        = None,
        metrics     : Any        = None,
        optimizers  : Any        = None,
        # Misc
        verbose     : bool       = True,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.verbose       = verbose
        # For saving/loading
        self.name          = name
        self.fullname      = fullname
        self.root          = root
        # For model architecture
        self.in_channels   = in_channels
        self.out_channels  = out_channels or num_classes or self.in_channels
        self.weights       = None
        self.assign_weights(weights)
        # For training
        self.loss          = None
        self.train_metrics = None
        self.val_metrics   = None
        self.test_metrics  = None
        self.optims        = None
        self.init_loss(loss)
        self.init_metrics(metrics)
        self.init_optimizers(optimizers)
        
    # region Properties
    
    @property
    def name(self) -> str:
        """Return the model's name."""
        return self._name
    
    @name.setter
    def name(self, name: str | None):
        """Specify the model's name. This value should only be defined once."""
        if name is None or name == "":
            name = humps.kebabize(self.__class__.__name__).lower()
        self._name = name
    
    @property
    def fullname(self) -> str:
        """Return the model's fullname = name-suffix"""
        return self._fullname
    
    @fullname.setter
    def fullname(self, fullname: str | None):
        """Specify the model's fullname. This value should only be defined once."""
        self._fullname = fullname if fullname not in [None, ""] else self.name
    
    @property
    def root(self) -> core.Path:
        return self._root
    
    @root.setter
    def root(self, root: Any):
        root = core.Path(root)
        # if root.name != self.fullname:
        #     root /= self.fullname
        self._root      = root
        self._debug_dir = root / "debug"
        self._ckpt_dir  = root
    
    @property
    def ckpt_dir(self) -> core.Path:
        if self._ckpt_dir is None:
            self._ckpt_dir = self.root
        return self._ckpt_dir
    
    @property
    def debug_dir(self) -> core.Path:
        if self._debug_dir is None:
            self._debug_dir = self.root / "debug"
        return self._debug_dir
    
    @property
    def num_classes(self) -> int:
        """Just an alias to :attr:`out_channels`."""
        return self.out_channels
    
    @num_classes.setter
    def num_classes(self, num_classes: int):
        """Just an alias to :attr:`out_channels`."""
        self.out_channels = num_classes
    
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
    
    # endregion
    
    # region Initialization
    
    def create_dir(self):
        """Create directories before training begins."""
        for path in [self.root, self.ckpt_dir, self.debug_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
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
    
    @abstractmethod
    def init_weights(self, model: nn.Module):
        """Initialize the model's weights."""
        pass
    
    def assign_weights(self, weights: Any):
        if isinstance(weights, str):
            if core.Path(weights).is_weights_file():
                weights = core.Path(weights)
            elif weights in self.zoo:
                weights         = self.zoo[weights]
                weights["path"] = self.zoo_dir / weights.get("path", "")
                num_classes     = getattr(weights, "num_classes", None)
                if num_classes and num_classes != self.num_classes:
                    self.num_classes = num_classes
                    console.log(f"Overriding :attr:`num_classes` with {num_classes}.")
            else:
                error_console.log(f"The key ``'{weights}'`` has not been defined in :attr:`zoo`.")
                weights = None
        elif isinstance(weights, core.Path):
            assert weights.is_weights_file(), f":param:`weights` must be a valid path to a weight file, but got {weights}."
        elif isinstance(weights, dict):
            pass  # No need to do anything here
        else:
            weights = None
        self.weights = weights
        
    def load_weights(self, weights: Any = None, overwrite: bool = False):
        """Load weights. It only loads the intersection layers of matching keys
        and shapes between the current model and weights.
        """
        self.weights = weights or self.weights
        
        # Get the state_dict
        state_dict = None
        if isinstance(self.weights, core.Path | str) and core.Path(self.weights).is_weights_file():
            self.zoo_dir.mkdir(parents=True, exist_ok=True)
            state_dict = load_state_dict(model=self, weights=self.weights, weights_only=False)
            state_dict = state_dict.get("state_dict", state_dict)
        elif isinstance(self.weights, dict):
            if "path" in self.weights:
                path = core.Path(self.weights["path"])
                if path.is_weights_file():
                    state_dict = load_state_dict(model=self, weights=path, weights_only=False)
                    state_dict = state_dict.get("state_dict", state_dict)
            else:
                state_dict = self.weights.get("state_dict", self.weights)
        else:
            error_console.log(f"[yellow]Cannot load from weights from: {self.weights}!")
        
        if state_dict:
            self.load_state_dict(state_dict=state_dict)
            if self.verbose:
                console.log(f"Load model's weights from: {self.weights}!")
    
    def init_loss(self, loss: Any):
        """Specify the model's loss functions. This value should only be defined once."""
        if isinstance(loss, str):
            self.loss = LOSSES.build(name=loss)
        elif isinstance(loss, dict):
            self.loss = LOSSES.build(config=loss)
        else:
            self.loss = loss
        if isinstance(self.loss, nn.Module):
            self.loss.requires_grad = True
            self.loss.eval()
    
    def init_metrics(self, metrics: Any):
        """Assign metrics.
        
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
        # Train
        train_metrics = metrics.get("train") if isinstance(metrics, dict) else metrics
        self.train_metrics = self.create_metrics(metrics=train_metrics)
        # This is a simple hack since LightningModule needs the metric to be
        # defined with self.<metric>. So, here we dynamically add the metric
        # attribute to the class.
        if self.train_metrics:
            for metric in self.train_metrics:
                name = f"train/{metric.name}"
                setattr(self, name, metric)
        
        # Val
        val_metrics = metrics.get("val") if isinstance(metrics, dict) else metrics
        self.val_metrics = self.create_metrics(val_metrics)
        if self.val_metrics:
            for metric in self.val_metrics:
                name = f"val/{metric.name}"
                setattr(self, name, metric)
        
        # Test
        test_metrics = metrics.get("test") if isinstance(metrics, dict) else metrics
        self.test_metrics = self.create_metrics(test_metrics)
        if self.test_metrics:
            for metric in self.test_metrics:
                name = f"test/{metric.name}"
                setattr(self, name, metric)
    
    def init_optimizers(self, optimizers: Any):
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally, you need one, but for GANs you might have
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
            
        Examples:
            def configure_optimizers(self):
                optimizer = Adam(...)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": ReduceLROnPlateau(optimizer, ...),
                        "monitor": "metric_to_track",
                        "frequency": "indicates how often the metric is updated",
                        #  If "monitor" references validation metrics, then "frequency" should be set to a
                        #  multiple of "trainer.check_val_every_n_epoch".
                    },
                }
            
            def configure_optimizers(self):
                optimizer1 = Adam(...)
                optimizer2 = SGD(...)
                scheduler1 = ReduceLROnPlateau(optimizer1, ...)
                scheduler2 = LambdaLR(optimizer2, ...)
                return (
                    {
                        "optimizer": optimizer1,
                        "lr_scheduler": {
                            "scheduler": scheduler1,
                            "monitor": "metric_to_track",
                        },
                    },
                    {"optimizer": optimizer2, "lr_scheduler": scheduler2},
                )
        """
        if optimizers is None:
            console.log(
                f"[yellow]No optimizers have been defined! Consider subclassing "
                f"this function to manually define the optimizers."
            )
            return None
        if isinstance(optimizers, dict):
            optimizers = [optimizers]
        assert isinstance(optimizers, list) and all(isinstance(o, dict) for o in optimizers)
        
        for optim in optimizers:
            optimizer           = optim.get("optimizer"          , None)
            network_params_only = optim.get("network_params_only", True)
            lr_scheduler        = optim.get("lr_scheduler"       , None)
           
            # Define optimizer
            if optimizer is None:
                raise ValueError(f":param:`optimizer` must be defined.")
            if isinstance(optimizer, dict):
                optimizer = OPTIMIZERS.build(network=self, config=optimizer, network_params_only=network_params_only)
            optim["optimizer"] = optimizer
            
            # Define learning rate scheduler
            if "lr_scheduler" in optim and lr_scheduler is None:
                optim.pop("lr_scheduler")
            elif lr_scheduler and isinstance(lr_scheduler, dict):
                scheduler = lr_scheduler.get("scheduler", None)
                if scheduler is None:
                    raise ValueError(f":param:`scheduler` must be defined.")
                if isinstance(scheduler, dict):
                    # after scheduler
                    if "after_scheduler" in scheduler:
                        after_scheduler = scheduler["after_scheduler"]
                        scheduler.pop("after_scheduler")
                    else:
                        after_scheduler = None
                    if isinstance(after_scheduler, dict):
                        after_scheduler = LR_SCHEDULERS.build(optimizer=optim["optimizer"], config=after_scheduler)
                        scheduler["after_scheduler"] = after_scheduler
                    #
                    scheduler = LR_SCHEDULERS.build(optimizer=optim["optimizer"], config=scheduler)
                lr_scheduler["scheduler"] = scheduler
                optim["lr_scheduler"]     = lr_scheduler
            
            # Update optim
            if "network_params_only" in optim:
                _ = optim.pop("network_params_only")
        
        # Re-assign optims
        if isinstance(optimizers, list | tuple) and len(optimizers) == 1:
            optimizers = optimizers[0]
        self.optims = optimizers
        return self.optims
    
    def configure_optimizers(self):
        return self.optims
    
    def compute_efficiency_score(self, *args, **kwargs) -> tuple[float, float, float]:
        """Compute the efficiency score of the model, including FLOPs, number
        of parameters, and runtime.
        """
        error_console.log(f"[yellow]This method has not been implemented yet!")
        return 0, 0, 0
    
    # endregion
    
    # region Forward Pass
    
    @abstractmethod
    def assert_datapoint(self, datapoint: dict) -> bool:
        """Check the datapoint before passing it to the :meth:`forward()`.
        Because each type of model requires different attributes in the datapoint,
        this method is used to ensure that the datapoint is valid.
        
        Args:
            datapoint: A :class:`dict` containing all attributes of a datapoint.
        """
        pass
    
    @abstractmethod
    def assert_outputs(self, outputs: dict) -> bool:
        """Check the outputs after passing it to the :meth:`forward()`. Because
        each type of model returns different attributes in the outputs, this
        method is used to ensure that the outputs are valid.

		Args:
			outputs: A :class:`dict` containing all predictions.
		"""
        pass
    
    @abstractmethod
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        """Forward pass, then compute the loss value.
        
        Args:
            datapoint: A :class:`dict` containing the attributes of a datapoint.
            
        Return:
            A :class:`dict` of all predictions with corresponding names. Note
            that the dictionary must contain the key ``'loss'`` and ``'pred'``.
        """
        '''
        # Forward
        outputs = forward(datapoint=datapoint, *args, **kwargs)
        assert_datapoint(datapoint)
        assert_outputs(outputs)
        # Loss
        pred   = outputs.get("pred")
        target = datapoint.get("target")
        outputs["loss"] = self.loss(pred, target) if self.loss else None
        # Return
        return outputs
        '''
        pass
    
    @abstractmethod
    def compute_metrics(
        self,
        datapoint: dict,
        outputs  : dict,
        metrics  : list[M.Metric] | None = None
    ) -> dict:
        """Compute metrics.

        Args:
            datapoint: A :class:`dict` containing the attributes of a datapoint.
            outputs: A :class:`dict` containing all predictions.
            metrics: A list of metric functions to compute. Default: ``None``.
        """
        '''
        # Check
        self.assert_datapoint(datapoint)
        self.assert_outputs(outputs)
        # Metrics
        pred    = outputs.get(<prediction>)
        target  = datapoint.get(<class_id>)
        results = {}
        if metrics:
            for i, metric in enumerate(metrics):
                metric_name = getattr(metric, "name", f"metric_{i}")
                results[metric_name] = metric(pred, target)
        # Return
        return results
        '''
        pass
    
    def forward_debug(self, datapoint: dict, *args, **kwargs) -> dict:
        """Forward pass for debugging. This method is used to debug the model's
        forward pass by returning the intermediate outputs.

        Args:
            datapoint: A :class:`dict` containing the attributes of a datapoint.
            
        Return:
            A :class:`dict` of all intermediate outputs with corresponding names.
            Default: ``{}``.
        """
        return {}
    
    @abstractmethod
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        """Forward pass. This is the primary :meth:`forward` function of the model.
        
        Args:
            datapoint: A :class:`dict` containing the attributes of a datapoint.
            
        Return:
            A :class:`dict` of all predictions with corresponding names.
            Default: ``{}``.
        """
        '''
        assert_datapoint(datapoint)
        ...
        return {}
        '''
        pass
    
    # endregion
    
    # region Training
    
    def on_fit_start(self):
        """Called at the beginning of fit."""
        self.create_dir()

    def training_step(self, batch: dict, batch_idx: int, *args, **kwargs) -> StepOutput | None:
        """Here you compute and return the training loss, and some additional
        metrics for e.g., the progress bar or logger.
        
        Args:
            batch: The output of :class:`~torch.utils.data.DataLoader`. It is a
                :class:`dict` containing the attributes of a datapoint.
            batch_idx: An integer displaying index of this batch.
            
        Return:
            Any of:
                - The loss tensor.
                - A :class:`dict`. Can include any keys, but must include the key ``'loss'``.
                - ``None``, training will skip to the next batch.
        """
        # Forward
        outputs  = self.forward_loss(datapoint=batch, *args, **kwargs)
        outputs |= self.compute_metrics(
            datapoint = batch,
            outputs   = outputs,
            metrics   = self.train_metrics
        )
        # Log values
        log_values  = {"step": self.current_epoch}
        log_values |= {f"train/{k}": v for k, v in outputs.items() if v is not None and not core.is_image(v)}
        self.log_dict(
            dictionary     = log_values,
            prog_bar       = False,
            logger         = True,
            on_step        = False,
            on_epoch       = True,
            sync_dist      = True,
            rank_zero_only = False,
        )
        # Return
        loss = outputs.get("loss", None)
        return loss

    def on_train_epoch_end(self):
        """Called in the training loop at the very end of the epoch."""
        if self.train_metrics:
            for i, metric in enumerate(self.train_metrics):
                metric.reset()

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs) -> StepOutput | None:
        """Operates on a single batch of data from the validation set. In this
        step, you might generate examples or calculate anything of interest like
        accuracy.
        
        Args:
            batch: The output of :class:`~torch.utils.data.DataLoader`. It is a
                :class:`dict` containing the attributes of a datapoint.
            batch_idx: An integer displaying index of this batch.
            
        Return:
            Any of:
                - The loss tensor.
                - A :class:`dict`. Can include any keys, but must include the key ``'loss'``.
                - ``None``, training will skip to the next batch.
        """
        # Forward
        outputs  = self.forward_loss(datapoint=batch, *args, **kwargs)
        outputs |= self.compute_metrics(
            datapoint = batch,
            outputs   = outputs,
            metrics   = self.val_metrics
        )
        # Log values
        log_values  = {"step": self.current_epoch}
        log_values |= {f"val/{k}": v for k, v in outputs.items() if v is not None and not core.is_image(v)}
        self.log_dict(
            dictionary     = log_values,
            prog_bar       = False,
            logger         = True,
            on_step        = False,
            on_epoch       = True,
            sync_dist      = True,
            rank_zero_only = False,
        )
        # Log images
        if self.should_log_images():
            data = batch | {"outputs": outputs}
            self.log_images(
                epoch = self.current_epoch,
                step  = self.global_step,
                data  = data,
            )
        # Return
        loss = outputs.get("loss", None)
        return loss
    
    def on_validation_epoch_end(self):
        """Called in the validation loop at the very end of the epoch."""
        if self.val_metrics:
            for i, metric in enumerate(self.val_metrics):
                metric.reset()

    def on_test_start(self) -> None:
        """Called at the very beginning of testing."""
        self.create_dir()

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs) -> StepOutput | None:
        """Operates on a single batch of data from the test set. In this step
        you'd normally generate examples or calculate anything of interest such
        as accuracy.

        Args:
            batch: The output of :class:`~torch.utils.data.DataLoader`. It is a
                :class:`dict` containing the attributes of a datapoint.
            batch_idx: An integer displaying index of this batch.
            
        Return:
            Any of:
                - The loss tensor.
                - A :class:`dict`. Can include any keys, but must include the key ``'loss'``.
                - ``None``, training will skip to the next batch.
        """
        # Forward
        outputs  = self.forward_loss(datapoint=batch, *args, **kwargs)
        outputs |= self.compute_metrics(
            datapoint = batch,
            outputs   = outputs,
            metrics   = self.test_metrics
        )
        # Log values
        log_values  = {"step": self.current_epoch}
        log_values |= {f"test/{k}": v for k, v in outputs.items() if v is not None and not core.is_image(v)}
        self.log_dict(
            dictionary     = log_values,
            prog_bar       = False,
            logger         = True,
            on_step        = False,
            on_epoch       = True,
            sync_dist      = True,
            rank_zero_only = False,
        )
        # Log images
        if self.should_log_images():
            data = batch | {"outputs": outputs}
            self.log_images(
                epoch = self.current_epoch,
                step  = self.global_step,
                data  = data,
            )
        # Return
        loss = outputs.get("loss", None)
        return loss
    
    def on_test_epoch_end(self):
        """Called in the test loop at the very end of the epoch."""
        if self.test_metrics:
            for i, metric in enumerate(self.test_metrics):
                metric.reset()
    
    # endregion
    
    # region Predicting
    
    @abstractmethod
    @torch.no_grad()
    def infer(self, datapoint: dict, *args, **kwargs) -> dict:
        """Infer the model on a single datapoint. This method is different from
        :meth:`forward()` in term that you may want to perform additional
        pre-processing or post-processing steps.
        
        Notes:
            If you want to perform specific pre-processing or post-processing
            steps, you should override this method.
        
        Args:
            datapoint: A :class:`dict` containing the attributes of a datapoint.
        """
        pass
    
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
        
        if input_dims:
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
        
        if input_dims:
            input_sample = torch.randn(input_dims)
        else:
            raise ValueError(f":param:`input_dims` must be defined.")
        
        script = self.to_torchscript(method=method, example_inputs=input_sample)
        torch.jit.save(script, file_path)
    
    # endregion
    
    # region Logging
    
    def should_log_images(self) -> bool:
        """Check if we should save debug images."""
        log_image_every_n_epochs = getattr(self.trainer, "log_image_every_n_epochs", 0)
        return (
            self.trainer.is_global_zero
            and log_image_every_n_epochs > 0
            and self.current_epoch % log_image_every_n_epochs == 0
        )
    
    def log_images(
        self,
        epoch    : int,
        step     : int,
        data     : dict,
        extension: str = ".jpg"
    ):
        """Log debug images to :attr:`debug_dir`."""
        pass
    
    # endregion
    
# endregion


# region Extra Model

class ExtraModel(Model, ABC):
    """A wrapper model that wraps around another model defined in :mod:`mon_extra`.
    This is useful when we want to add the third-party models to :mod:`mon`'s models.
    without reimplementing the entire model.
    
    Args:
        model: The model to wrap around. To make thing simple, we agree on
            the following naming convention: ``'model'``.
    
    Todo:
        Usually, we only need to define the model architecture and load the
        pretrained weights. The training should be performed using the original
        package's script.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: nn.Module = None
    
    def load_weights(self, weights: Any = None, overwrite: bool = False):
        """Load weights. It only loads the intersection layers of matching keys
        and shapes between the current model and weights.
        """
        self.weights = weights or self.weights
        
        # Get the state_dict
        state_dict = None
        if isinstance(self.weights, core.Path | str) and core.Path(self.weights).is_weights_file():
            self.zoo_dir.mkdir(parents=True, exist_ok=True)
            state_dict = load_state_dict(model=self, weights=self.weights, weights_only=False)
            state_dict = state_dict.get("state_dict", state_dict)
        elif isinstance(self.weights, dict):
            if "path" in self.weights:
                path = core.Path(self.weights["path"])
                if path.is_weights_file():
                    state_dict = load_state_dict(model=self, weights=path, weights_only=False)
                    state_dict = state_dict.get("state_dict", state_dict)
            else:
                state_dict = self.weights.get("state_dict", self.weights)
        else:
            error_console.log(f"[yellow]Cannot load from weights from: {self.weights}!")
        
        if state_dict:
            self.model.load_state_dict(state_dict=state_dict)
            if self.verbose:
                console.log(f"Load model's weights from: {self.weights}!")
                
# endregion
