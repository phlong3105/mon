#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for mixins that add querying and filtering functionality to data
containers.

This module defines mixin classes that can be used to extend data containers
with additional capabilities for querying and filtering data based on specific
criteria.
"""

__all__ = [
    "BatchCollateMixin",
]

import abc
from typing import Any


class BatchCollateMixin(abc.ABC):
    """A mixin class that adds batch collation functionality to data containers.

    This class defines the collate function for batching datapoints when using
    with a DataLoader.
    """

    @abc.abstractmethod
    def collate_fn(self, batch: list[dict]) -> dict[str, Any]:
        """Collates a batch of input items for torch.utils.data.dataset.DataLoader.

        By default, batch is a list of dicts, where each dict is a datapoint.
        We need to collate these into a single dict where each key corresponds to
        a modality and the values are stacked tensors or arrays.

        Args:
            batch (list[dict]): List of dicts, each dict is a datapoint.

        Returns:
            dict[str, Any]: Collated dictionary for torch.utils.data.dataset.DataLoader.
        """
        pass
