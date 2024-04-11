#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements multiple factory classes."""

from __future__ import annotations

__all__ = [
    "LRSchedulerFactory",
    "OptimizerFactory",
]

import copy

import humps
from torch import nn, optim

import mon
from mon.core import factory


# region Factory

class OptimizerFactory(factory.Factory):
    """The factory for registering and building optimizers."""
    
    def build(
        self,
        network            : nn.Module,
        name               : str  | None = None,
        config             : dict | None = None,
        network_params_only: bool        = False,
        to_dict            : bool        = False,
        **kwargs
    ):
        """Build an instance of the registered optimizer corresponding to the given name.
        
        Args:
            network: A neural network.
            name: An optimizer's name.
            config: The optimizer's arguments.
            network_params_only: If ``True``, only the network's parameters are used.
            to_dict: If ``True``, return a :class:`dict` of ``{:param:`name`: attr:`instance`}``. Default: ``False``.
            **kwargs: Additional arguments that may be needed for the optimizer.
        
        Returns:
            An optimizer.
        """
        if name is None and config is None:
            return None
        if name is None and config is not None:
            assert "name" in config
            config_  = copy.deepcopy(config)
            name_    = config_.pop("name")
            name     = name or name_
            kwargs  |= config_
        assert name is not None and name in self
        
        if hasattr(network, "named_parameters"):
            params = []
            if network_params_only:
                # Note: we must remove losses' and metrics' parameters
                for n, p in network.named_parameters():
                    if (
                              "loss" not in n
                        and "train/" not in n
                        and   "val/" not in n
                        and  "test/" not in n
                    ):
                        params.append(p)
            else:
                params = network.parameters()
            instance = self[name](params=params, **kwargs)
            
            if getattr(instance, "name", None) is None:
                instance.name = humps.depascalize(humps.pascalize(name))
            if to_dict:
                return {f"{name}": instance}
            else:
                return instance
        
        return None
    
    def build_instances(
        self,
        network            : nn.Module,
        configs            : list | None,
        network_params_only: bool        = True,
        to_dict            : bool        = False,
        *kwargs
    ):
        """Build multiple instances of different optimizers with the given
        arguments.
        
        Args:
            network: A neural network.
            configs: A :class:`list` of optimizers' arguments. Each item can be:
                - A name (:class:`str`).
                - A :class:`dict` of arguments containing the ``'name'`` key.
            network_params_only: If ``True``, only the network's parameters are used.
            to_dict: If ``True``, return a :class:`dict` of ``{:param:`name`: attr:`instance`}``. Default: ``False``.
                
        Returns:
            A :class:`list`, or :class:`dict` of optimizers.
        """
        if configs is None:
            return None
        assert isinstance(configs, list)
        
        configs_ = copy.deepcopy(configs)
        optimizers = {} if to_dict else []
        for config in configs_:
            if isinstance(config, str):
                name    = config
            else:
                name    = config.pop("name")
                kwargs |= config
            opt = self.build(
                network             = network,
                name                = name,
                network_params_only = network_params_only,
                to_dict             = to_dict,
                **kwargs
            )
            if opt is not None:
                if to_dict:
                    optimizers |= opt
                else:
                    optimizers.append(opt)
        
        return optimizers if len(optimizers) > 0 else None
    

class LRSchedulerFactory(factory.Factory):
    """The factory for registering and building learning rate schedulers."""
    
    def build(
        self,
        optimizer: optim.Optimizer,
        name     : str  | None = None,
        config   : dict | None = None,
        **kwargs
    ):
        """Build an instance of the registered scheduler corresponding to the
        given name.
        
        Args:
            optimizer: An optimizer.
            name: A scheduler's name.
            config: The scheduler's arguments.
        
        Returns:
            A learning rate scheduler.
        """
        if name is None and config is None:
            return None
        if name is None and config is not None:
            assert "name" in config
            config_ = copy.deepcopy(config)
            name_   = config_.pop("name")
            name    = name or name_
            kwargs |= config_
        assert name is not None and name in self
        
        if name in [
            "GradualWarmupScheduler",
            "gradual_warmup_scheduler",
            "gradual-warmup-scheduler"
        ]:
            after_scheduler = kwargs.pop("after_scheduler")
            if isinstance(after_scheduler, dict):
                name_ = after_scheduler.pop("name")
                if name_ in self:
                    after_scheduler = self[name_](optimizer=optimizer, **after_scheduler)
                else:
                    after_scheduler = None
            return self[name](
                optimizer       = optimizer,
                after_scheduler = after_scheduler,
                **kwargs
            )
        
        return self[name](optimizer=optimizer, **kwargs)
    
    def build_instances(
        self,
        optimizer: optim.Optimizer,
        configs  : list | None,
        **kwargs
    ):
        """Build multiple instances of different schedulers with the given
        arguments.
        
        Args:
            optimizer: An optimizer.
            configs: A :class:`list` of schedulers' arguments. Each item can be:
                - A name (:class:`str`)
                - A :class:`dict` of arguments containing the ''`name`'' key.
        
        Returns:
            A :class:`list` of learning rate schedulers
        """
        if configs is None:
            return None
        assert isinstance(configs, list)
        
        configs_   = copy.deepcopy(configs)
        schedulers = []
        for config in configs_:
            if isinstance(config, str):
                name = config
            else:
                name    = config.pop("name")
                kwargs |= config
            schedulers.append(
                self.build(optimizer=optimizer, name=name, **kwargs)
            )
        
        return schedulers if len(schedulers) > 0 else None

# endregion


mon.globals.OPTIMIZERS    = OptimizerFactory("Optimizer")
mon.globals.LR_SCHEDULERS = LRSchedulerFactory("LRScheduler")
