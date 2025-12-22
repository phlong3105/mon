#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for base data mixins.

This module defines abstract mixin classes that can be used to extend the
functionality of data handling classes in the MON framework. Mixins provide
a way to add specific features or behaviors to classes without using
traditional inheritance.
"""

__all__ = [
    "DatasetMixin",
]

import abc


# --- Abstract Mixins ---
class DatasetMixin(abc.ABC):
    """An abstract mixin class for data containers (e.g., data pools and dataset).
    Since DataPool and Dataset share similar functionalities, this mixin
    can be applied to both.

    This class serves as a base for mixin classes that extend the functionality
    of data pools. It does not implement any specific functionality itself,
    but provides a common interface for mixins to build upon.
    """
    
    # --- Hook ---
    @abc.abstractmethod
    def on_load_start(self):
        """A hook method called at the start of the data pool loading process.
        
        This method can be overridden by subclasses to perform additional
        operations before the data pool is loaded.
        """
        pass
    
    @abc.abstractmethod
    def on_load_end(self):
        """A hook method called at the end of the data pool loading process.
        
        This method can be overridden by subclasses to perform additional
        operations after the data pool has been loaded.
        """
        pass
        
