#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements multiple factory classes."""

from __future__ import annotations

__all__ = [
    "LRSchedulerFactory", "OptimizerFactory",
]

import copy

import humps
from torch import nn, optim

from mon import foundation


class OptimizerFactory(foundation.Factory):
    """The factory for registering and building optimizers."""
    
    def build(
        self,
        net : nn.Module,
        name: str  | None = None,
        cfg : dict | None = None,
        **kwargs
    ):
        """Build an instance of the registered optimizer corresponding to the
        given name.
        
        Args:
            net: A neural network.
            name: An optimizer's name.
            cfg: The optimizer's arguments.
            **kwargs: Additional arguments that may be needed for the optimizer.
        
        Returns:
            An optimizer.
        """
        if name is None and cfg is None:
            return None
        if name is None and cfg is not None:
            assert "name" in cfg
            cfg_    = copy.deepcopy(cfg)
            name_   = cfg_.pop("name")
            name    = name or name_
            kwargs |= cfg_
        
        assert name is not None
        assert name in self.registry
        if name not in self.registry:
            return None

        name = humps.camelize(name) if humps.is_snakecase(name) else name
        if hasattr(net, "parameters"):
            instance = self.registry[name](params=net.parameters(), **kwargs)
            if not hasattr(instance, "name"):
                instance.name = name
            return instance
        return None

    def build_instances(self, net: nn.Module, cfgs: list | None, *kwargs):
        """Build multiple instances of different optimizers with the given
        arguments.
        
        Args:
            net: A neural network.
            cfgs: A list of optimizers' arguments. Each item can be:
                - A name (string)
                - A dictionary of arguments containing the “name” key.
        
        Returns:
            A list of optimizers
        """
        if cfgs is None:
            return None
        assert isinstance(cfgs, list)

        cfgs_      = copy.deepcopy(cfgs)
        optimizers = []
        for cfg in cfgs_:
            if isinstance(cfg, str):
                name    = cfg
                optimizers.append(self.build(net=net, name=name, **kwargs))
            else:
                name    = cfg.pop("name")
                kwargs |= cfg
                optimizers.append(self.build(net=net, name=name, **kwargs))

        return optimizers if len(optimizers) > 0 else None


class LRSchedulerFactory(foundation.Factory):
    """The factory for registering and building learning rate schedulers."""
    
    def build(
        self,
        optimizer: optim.Optimizer,
        name     : str  | None = None,
        cfg      : dict | None = None,
        **kwargs
    ):
        """Build an instance of the registered scheduler corresponding to the
        given name.
        
        Args:
            optimizer: An optimizer.
            name: A scheduler's name.
            cfg: The scheduler's arguments.
        
        Returns:
            A learning rate scheduler.
        """
        if name is None and cfg is None:
            return None
        if name is None and cfg is not None:
            assert "name" in cfg
            cfg_    = copy.deepcopy(cfg)
            name_   = cfg_.pop("name")
            name    = name or name_
            kwargs |= cfg_

        assert name is not None
        assert name in self.registry
        if name not in self.registry:
            return None

        name = humps.camelize(name) if humps.is_snakecase(name) else name
        if name in ["GradualWarmupScheduler"]:
            after_scheduler = kwargs.pop("after_scheduler")
            if isinstance(after_scheduler, dict):
                name_ = after_scheduler.pop("name")
                if name_ in self.registry:
                    after_scheduler = self.registry[name_](
                        optimizer=optimizer,
                        **after_scheduler
                    )
                else:
                    after_scheduler = None
            return self.registry[name](
                optimizer       = optimizer,
                after_scheduler = after_scheduler,
                **kwargs
            )
        
        return self.registry[name](optimizer=optimizer, **kwargs)
    
    def build_instances(
        self,
        optimizer: optim.Optimizer,
        cfgs     : list | None,
        **kwargs
    ):
        """Build multiple instances of different schedulers with the given
        arguments.
        
        Args:
            optimizer: An optimizer.
            cfgs: A list of schedulers' arguments. Each item can be:
                - A name (string)
                - A dictionary of arguments containing the “name” key.
        
        Returns:
            A list of learning rate schedulers
        """
        if cfgs is None:
            return None
        assert isinstance(cfgs, list)
        
        cfgs_      = copy.deepcopy(cfgs)
        schedulers = []
        for cfg in cfgs_:
            if isinstance(cfg, str):
                name    = cfg
                schedulers.append(self.build(optimizer=optimizer, name=name, **kwargs))
            else:
                name    = cfg.pop("name")
                kwargs |= cfg
                schedulers.append(self.build(optimizer=optimizer, name=name, **kwargs))
        
        return schedulers if len(schedulers) > 0 else None
