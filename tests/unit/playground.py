
import sys

import inspect


def square(x):
    return x*x


def cube(x):
    return x**3


cube2 = cube


class Square:
    pass


__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
