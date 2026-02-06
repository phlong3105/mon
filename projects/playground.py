#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Playground."""

class Foo:

    name: str

    def __init__(self, name: str):
        if name is not None:
            self.name = name


foo1 = Foo("foo1")
foo2 = Foo(name=None)
print(foo1.name)
print(foo2.name)
