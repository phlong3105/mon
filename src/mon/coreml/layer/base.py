#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all models, and the corresponding
helper functions.
"""

from __future__ import annotations

__all__ = [
    "ConcatLayerParsingMixin", "ConvLayerParsingMixin", "HeadLayerParsingMixin",
    "Layer", "LayerParsingMixin", "MergingLayerParsingMixin",
    "PassThroughLayerParsingMixin", "SameChannelsLayerParsingMixin",
]

from abc import ABC, abstractmethod

from torch import nn


# region Parsing Mixin

class LayerParsingMixin:
    """The base class for layers' mixins. It adds additional methods used in the
    layer parsing process.
    """
    
    @classmethod
    def parse_layer(
        cls,
        f      : int | list[int],
        args   : list,
        ch     : list,
        hparams: dict | None = None,
    ) -> tuple[list, list]:
        """Get predefined layer's arguments and calculate the appropriate output
        channels. Also, adjust arguments' values with the given hyperparameter.
        
        Notes:
            This method is used in the process of parsing the model's layer.
            It's an improvement of YOLOv5's :meth:`parse_model` by delegating
            the task of calculate :param:`in_channels` and :param:`out_channels`
            to each layer instead of relying on :meth:`parse_model`.
            
        Args:
            f: From, i.e., the current layer receives output from the f-th layer.
                For example: -1 means from a previous layer; -2 means from 2
                previous layers; [99, 101] means from the 99th and 101st layers.
                This attribute is used in forward pass.
            args: Layer's parameters.
            ch: A list containing output channels of previous layers (of the
                model)
            hparams: Layer's hyperparameters. They are used to change the values
                of :param:`args`. Usually used in grid search or random search
                during training. Defaults to None.
        
        Returns:
            Layer's :param:`in_channels`.
            Layer's :param:`out_channels`.
            The adjusted :param:`args`.
        """
        if hparams is not None:
            args = cls.parse_hparams(args=args, hparams=hparams) or args
        args, ch = cls.parse_layer_args(f=f, args=args, ch=ch)
        return args, ch
    
    @classmethod
    @abstractmethod
    def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
        """Parse layer's arguments :param:`args`, calculate the
        :param:`out_channels`, and update :param:`args`. Also, append the
        :param:`out_channels` to :param:`ch` if needed.

        Args:
            f: From, i.e., the current layer receives output from the f-th layer.
                For example: -1 means from a previous layer; -2 means from 2
                previous layers; [99, 101] means from the 99th and 101st layers.
                This attribute is used in forward pass.
            args: Layer's parameters.
            ch: A list containing output channels of previous layers (of the
                model)
        
        Returns:
            The adjusted :param:`args` and :param:`ch`.
        """
        pass
        
    @classmethod
    def parse_hparams(cls, args: list, hparams: dict | None = None) -> list | None:
        """Parse hyperparameters and updated the corresponding values in
        :param:`args`.
        
        Args:
            args: Layer's parameters.
            hparams: Layer's hyperparameters. They're used to change the values
                of :param:`args`. Usually used in grid search or random search
                during training. Defaults to None.

        Returns:
            The adjusted :param:`args`.
        """
        return args
      

class ConvLayerParsingMixin(LayerParsingMixin):
    """The parsing mixin for convolutional layers. It implements the layer
    parsing method where :param:`in_channels` (equal to the previous layer's
    :param:`out_channels`) and the :param:`out_channels` can be different. In
    addition, the current :param:`out_channels` will be the next layer's
    :param:`in_channels`.
    """
    
    @classmethod
    def parse_layer_args(cls, f : int, args: list, ch: list) -> tuple[list, list]:
        """Parse layer's arguments :param:`args`, calculate the
        :param:`out_channels`, and update :param:`args`. Also, append the
        :param:`out_channels` to :param:`ch` if needed.

        Args:
            f: From, i.e., the current layer receives output from the f-th layer.
                For example: -1 means from the previous layer; -2 means from 2
                previous layers; [99, 101] means from the 99th and 101st layers.
                This attribute is used in forward pass.
            args: Layer's parameters.
            ch: A list containing output channels of previous layers (of the
                model)
        
        Returns:
            The adjusted :param:`args` and :param:`ch`.
        """
        if isinstance(f, list | tuple):
            c1, c2 = ch[f[0]], args[0]
        else:
            c1, c2 = ch[f], args[0]
        args = [c1, c2, *args[1:]]
        ch.append(c2)
        return args, ch
    

class ConcatLayerParsingMixin(LayerParsingMixin):
    """The parsing mixin for concatenating layers. It implements the layer
    parsing method where multiple features are concatenated from previous layers
    to create a single output feature of a similar shape.
    """
    
    @classmethod
    def parse_layer_args(cls, f : int, args: list, ch: list) -> tuple[list, list]:
        """Parse layer's arguments :param:`args`, calculate the
        :param:`out_channels`, and update :param:`args`. Also, append the
        :param:`out_channels` to :param:`ch` if needed.

        Args:
            f: From, i.e., the current layer receives output from the f-th layer.
                For example: -1 means from the previous layer; -2 means from 2
                previous layers; [99, 101] means from the 99th and 101st layers.
                This attribute is used in forward pass.
            args: Layer's parameters.
            ch: A list containing output channels of previous layers (of the
                model)
        
        Returns:
            The adjusted :param:`args` and :param:`ch`.
        """
        assert isinstance(f, list | tuple)
        c2 = sum([ch[i] for i in f])
        ch.append(c2)
        return args, ch


class MergingLayerParsingMixin(LayerParsingMixin):
    """The parsing mixin for merging layers. It implements the layer parsing
    method where multiple features of the same shape are merged from previous
    layers to create a single output feature of a similar shape.
    """
    
    @classmethod
    def parse_layer_args(cls, f : int, args: list, ch: list) -> tuple[list, list]:
        """Parse layer's arguments :param:`args`, calculate the
        :param:`out_channels`, and update :param:`args`. Also, append the
        :param:`out_channels` to :param:`ch` if needed.

        Args:
            f: From, i.e., the current layer receives output from the f-th layer.
                For example: -1 means from a previous layer; -2 means from 2
                previous layers; [99, 101] means from the 99th and 101st layers.
                This attribute is used in forward pass.
            args: Layer's parameters.
            ch: A list containing output channels of previous layers (of the
                model)
        
        Returns:
            The adjusted :param:`args` and :param:`ch`.
        """
        assert isinstance(f, list | tuple)
        c2 = ch[f[-1]]
        ch.append(c2)
        return args, ch
    

class HeadLayerParsingMixin(LayerParsingMixin):
    """The parsing mixin for head layers. It implements the layer parsing method
    where :param:`out_channels` is the predicting number of classes.
    """
    
    @classmethod
    def parse_layer(
        cls,
        f      : int | list[int],
        args   : list,
        nc     : int,
        ch     : list,
        hparams: dict | None = None,
    ) -> tuple[list, list]:
        """Get predefined layer's arguments and calculate the appropriate output
        channels. Additionally, adjust arguments' values with the given
        hyperparameter.
        
        Notes:
            This method is used in the process of parsing the model's layer.
            It is an improvement of YOLOv5's :meth:`parse_model` by delegating
            the task of calculate :param:`in_channels` and :param:`out_channels`
            to each layer instead of relying on :meth:`parse_model`.
            
        Args:
            f: From, i.e., the current layer receives output from the f-th layer.
                For example: -1 means from a previous layer; -2 means from 2
                previous layers; [99, 101] means from the 99th and 101st layers.
                This attribute is used in forward pass.
            args: Layer's parameters.
            nc: Number of predicting classes.
            ch: A list containing output channels of previous layers (of the
                model)
            hparams: Layer's hyperparameters. They are used to change the values
                of :param:`args`. Usually used in grid search or random search
                during training. Defaults to None.
        
        Returns:
            Layer's :param:`in_channels`.
            Layer's :param:`out_channels`.
            The adjusted :param:`args`.
        """
        if hparams is not None:
            args = cls.parse_hparams(args=args, hparams=hparams) or args
        args, ch = cls.parse_layer_args(f=f, args=args, nc=nc, ch=ch)
        return args, ch
    
    @classmethod
    def parse_layer_args(
        cls,
        f   : int | list[int],
        args: list,
        nc  : int,
        ch  : list,
    ) -> tuple[list, list]:
        """Parse layer's arguments :param:`args`, calculate the
        :param:`out_channels`, and update :param:`args`. Also, append the
        :param:`out_channels` to :param:`ch` if needed.

        Args:
            f: From, i.e., the current layer receives output from the f-th layer.
                For example: -1 means from the previous layer; -2 means from 2
                previous layers; [99, 101] means from the 99th and 101st layers.
                This attribute is used in forward pass.
            args: Layer's parameters.
            nc: Number of predicting classes.
            ch: A list containing output channels of previous layers (of the
                model)
        
        Returns:
            The adjusted :param:`args` and :param:`ch`.
        """
        c1   = args[0]
        c2   = nc
        args = [c1, c2, *args[1:]]
        ch.append(c2)
        return args, ch


class PassThroughLayerParsingMixin(LayerParsingMixin):
    """The parsing mixin for pass-through layers. It implements the layer
    parsing method where the features simply passing through without any changes
     in the number of layers.
    """
    
    @classmethod
    def parse_layer_args(cls, f : int, args: list, ch: list) -> tuple[list, list]:
        """Parse layer's arguments :param:`args`, calculate the
        :param:`out_channels`, and update :param:`args`. Also, append the
        :param:`out_channels` to :param:`ch` if needed.

        Args:
            f: From, i.e., the current layer receives output from the f-th layer.
                For example: -1 means from a previous layer; -2 means from 2
                previous layers; [99, 101] means from the 99th and 101st layers.
                This attribute is used in forward pass.
            args: Layer's parameters.
            ch: A list containing output channels of previous layers (of the
                model)
        
        Returns:
            The adjusted :param:`args` and :param:`ch`.
        """
        c2 = ch[f]
        ch.append(c2)
        return args, ch


class SameChannelsLayerParsingMixin(LayerParsingMixin):
    """The parsing mixin for layers that has :param:`in_channels` equals to
    :param:`out_channels`.
    """
    
    @classmethod
    def parse_layer_args(cls, f : int, args: list, ch: list) -> tuple[list, list]:
        """Parse layer's arguments :param:`args`, calculate the
        :param:`out_channels`, and update :param:`args`. Also, append the
        :param:`out_channels` to :param:`ch` if needed.

        Args:
            f: From, i.e., the current layer receives output from the f-th layer.
                For example: -1 means from the previous layer; -2 means from 2
                previous layers; [99, 101] means from the 99th and 101st layers.
                This attribute is used in forward pass.
            args: Layer's parameters.
            ch: A list containing output channels of previous layers (of the
                model)
        
        Returns:
            The adjusted :param:`args` and :param:`ch`.
        """
        if isinstance(f, list | tuple):
            c1 = c2 = ch[f[0]]
        else:
            c1 = c2 = ch[f]
        args = [c1, *args[0:]]
        ch.append(c2)
        return args, ch

# endregion


# region Layer

class Layer(LayerParsingMixin, nn.Module, ABC):
    """The base class for all layers."""
    pass

# endregion
