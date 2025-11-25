#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines the base classes for active learning."""

__all__ = []

import abc
from typing import Any, Iterator


class BaseQueue(abc.ABC):
    """Defines a generic queue protocol for data that is passed between active
    learning components.

    This can be a simple local `queue.Queue`, or a more sophisticated distributed
    queue system.
    
    The primary use case for this is to allow a query strategy to enqueue some
    data point for the labeling strategy to consume, and once the labeling
    is done, enqueue to a data serialization workflow. While there is no explicit
    restriction on the **type** of queue that is implemented, a reasonable assumption
    to make would be a FIFO queue, unless otherwise specified by the concrete
    implementation.

    Optional Serialization Methods
    -------------------------------
    Implementations may optionally provide ``to_list()`` and ``from_list()``
    methods for checkpoint serialization. If not provided, the queue will be
    serialized using ``torch.save()`` as a fallback.
    
    See Also:
        - QueryStrategy : Enqueues data to be labeled.
        - LabelStrategy : Dequeues data for labeling and enqueues labeled data.
        - DriverProtocol: Uses queues to pass data between strategies.
    """
    
    @abc.abstractmethod
    def put(self, item: Any):
        """Put a data point into the queue.

        Args:
            item: The data point to put into the queue.
        """
        pass
    
    @abc.abstractmethod
    def get(self) -> Any:
        """Get a data point from the queue.

        This method should remove the data point from the queue, and return it
        to a consumer.

        Returns:
            The data point that was removed from the queue.
        """
        pass
    
    @abc.abstractmethod
    def empty(self) -> bool:
        """Check if the queue is empty/has been depleted.

        Returns:
            ``True`` if the queue is empty, ``False`` otherwise.
        """
        pass


class BaseDataPool(abc.ABC):
    """An abstract protocol for some reservoir of data that is used for some part
    of active learning, parametrized such that it will return data points of
    an arbitrary type.

    **All** methods are left abstract, and need to be defined by concrete implementations.
    For the most part, a ``torch.utils.data.Dataset`` would match this protocol,
    provided that it implements the :meth:`append` method which will allow data
    to be persisted to a filesystem.
    
    See Also:
        - DriverProtocol: Uses data pools for training, validation, and unlabeled data.
        - AbstractQueue : Queue protocol for passing data between components.
    """

    def __getitem__(self, index: int) -> Any:
        """Get a data structure from the data pool.

        This method should retrieve an item from the pool by a flat index.

        Args:
            index: The index of the data structure to get.

        Returns:
            The data structure at the given index.
        """
        ...

    def __len__(self) -> int:
        """Get the length of the data pool.
        
        Returns
        -------
        int
            The length of the data pool.
        """
        ...

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the data pool.

        This method should return an iterator over the data pool.

        Returns
        -------
        Iterator[T]
            An iterator over the data pool.
        """
        ...

    def append(self, item: Any):
        """Append a data structure to the data pool.

        For persistent storage pools, this will actually mean that the
        ``item`` is serialized to a filesystem.

        Parameters
        ----------
        item: T
            The data structure to append to the data pool.
        """
        ...
