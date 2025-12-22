#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for mixins that handles metadata and orchestration for data
containers.

This module defines mixin classes that facilitate the management and organization
of data containers, enabling features such as factory registration based on
supported tasks.
"""

__all__ = [
    "RegistrableMixin",
]

import abc

from mon.core import Task


class RegistrableMixin(abc.ABC):
    """A mixin class that adds metadata attribute to data containers for
    factory registration purposes.

    This class defines common dataset attributes for categorization, such as
    supported tasks. This is useful for factory-related operations.

    Attributes:
        _tasks (list[Task]): A list of supported tasks. Defaults to an empty
            list and should be overridden in subclasses.
    """

    _tasks: list[Task] = []

    # --- Properties ---
    @property
    def tasks(self) -> list[Task]:
        """Getter for the list of supported tasks.

        Returns:
            list[Task]: The list of supported tasks.
        """
        return self._tasks
