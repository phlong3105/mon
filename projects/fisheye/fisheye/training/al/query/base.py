#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines the base Query class."""

__all__ = [
    "BaseQuery",
]

import abc

#----- Base Query -----
class BaseQuery(abc.ABC):
    """Base class for Query strategies."""
    
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def query(self, unlabeled_data: list, budget: int) -> list:
        """Query the most informative samples from the unlabeled data.

        Args:
            unlabeled_data: A list of unlabeled data samples.
            budget: The number of samples to query.

        Returns:
            A list of queried data samples.
        """
        pass
