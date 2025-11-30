# my_package/__init__.py
# This is the package-level docstring (for the entire package).

"""A sample package demonstrating Google-style docstrings.

This package contains modules for basic calculations and data handling.
It serves as an example for structured documentation.
"""

__all__ = ['Calculator']  # Optional: Exported names

from .my_module import Calculator  # Example import for package init

# my_package/my_module.py
# This is the module-level docstring (at the top of the file).
"""A module for simple calculations with Google-style docstrings.

This module provides a Calculator class and utility functions.
It emphasizes clear, structured documentation for readability.
"""

import math


class Calculator:
    """A class for performing basic arithmetic operations.

    This class demonstrates Google-style docstrings for classes, including
    attributes and methods. It handles addition and square root calculations.

    Attributes:
        base_value (int): The starting value for calculations (default 0).

    Examples:
        calc = Calculator(5)
        calc.add(3)  # Returns 8
    """

    def __init__(self, base_value: int = 0):
        """Initializes the calculator with a base value.

        Args:
            base_value (int, optional): Initial value. Defaults to 0.
        """
        self.base_value = base_value

    def add(self, x: int) -> int:
        """Adds a value to the base_value.

        This method performs simple addition and returns the result.

        Args:
            x (int): The value to add.

        Returns:
            int: The sum of base_value and x.

        Raises:
            ValueError: If x is negative.
        """
        if x < 0:
            raise ValueError("x must be non-negative.")
        return self.base_value + x

    def square_root(self) -> float:
        """Computes the square root of the base_value.

        Returns:
            float: The square root.

        Raises:
            ValueError: If base_value is negative.
        """
        if self.base_value < 0:
            raise ValueError("base_value must be non-negative for square root.")
        return math.sqrt(self.base_value)


def multiply(a: int, b: int) -> int:
    """Multiplies two integers.

    This is a standalone function example with Google-style docstring.

    Args:
        a (int): First multiplier.
        b (int): Second multiplier.

    Returns:
        int: The product of a and b.

    Examples:
        multiply(2, 3)  # Returns 6
    """
    return a * b
