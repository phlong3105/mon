#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generic Enum Structures.

This module provides enhanced Enum classes with additional functionality.
"""

from __future__ import annotations

__all__ = [
    "AddValue",
    "AddValueEnum",
    "AutoNumberEnum",
    "Constant",
    "EJECT",
    "Enum",
    "EnumMeta",
    "EnumType",
    "Flag",
    "FlagBoundary",
    "IntEnum",
    "IntFlag",
    "KEEP",
    "LowerStrEnum",
    "MagicValue",
    "Member",
    "MultiStrEnum",
    "MultiValue",
    "MultiValueEnum",
    "NamedConstant",
    "NamedTuple",
    "NoAlias",
    "NoAliasEnum",
    "NonMember",
    "OrderedEnum",
    "ReprEnum",
    "StrEnum",
    "Unique",
    "UniqueEnum",
    "UpperStrEnum",
    "add_stdlib_integration",
    "bin",
    "constant",
    "enum",
    "enum_property",
    "extend_enum",
    "member",
    "no_arg",
    "nonmember",
    "property",
    "remove_stdlib_integration",
    "skip",
    "unique",
]

from aenum import (
    add_stdlib_integration,
    AddValue,
    AddValueEnum,
    AutoNumberEnum,
    bin,
    Constant,
    constant,
    EJECT,
    enum,
    Enum as Enum_,
    enum_property,
    EnumMeta,
    EnumType,
    extend_enum,
    Flag,
    FlagBoundary,
    IntEnum,
    IntFlag,
    KEEP,
    LowerStrEnum,
    MagicValue,
    Member,
    member,
    MultiValue,
    MultiValueEnum,
    NamedConstant,
    NamedTuple,
    no_arg,
    NoAlias,
    NoAliasEnum,
    NonMember,
    nonmember,
    OrderedEnum,
    property,
    remove_stdlib_integration,
    ReprEnum,
    skip,
    StrEnum as StrEnum_,
    Unique,
    unique,
    UniqueEnum,
    UpperStrEnum,
)


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class Enum(Enum_):

    # --- Lifecycle & Initialization ---
    @classmethod
    def _missing_(cls, value):
        # 1. If no value is passed (None), return the first member
        if value is None:
            value = "default"

        # 2. If the user explicitly passes the string "default",
        # return the first member if "DEFAULT" is not defined, otherwise return
        # the "DEFAULT" member
        if value == "default":
            if "DEFAULT" in cls.__members__:
                return cls.DEFAULT
            else:
                return list(cls)[0]

        # 3. Otherwise, it's an invalid extension
        raise ValueError(f"'{value}' is not a valid '{cls.__name__}' extension.")

    # --- Retrieval ---
    @classmethod
    def names(cls):
        """Returns a list of all enum member names."""
        return [member.name for member in cls]

    @classmethod
    def values(cls):
        """Returns a list of all enum member values."""
        return [member.value for member in cls]


class StrEnum(StrEnum_):

    # --- Lifecycle & Initialization ---
    @classmethod
    def _missing_(cls, value):
        # 1. If no value is passed (None), return the first member
        if value is None:
            return list(cls)[0]

        # 2. If the user explicitly passes the string "default",
        # return the first member if "DEFAULT" is not defined, otherwise return
        # the "DEFAULT" member
        if value == "default":
            if "DEFAULT" in cls.__members__:
                return cls.DEFAULT
            else:
                return list(cls)[0]

        # 3. Otherwise, it's an invalid extension
        raise ValueError(f"'{value}' is not a valid '{cls.__name__}' extension.")

    # --- Retrieval ---
    @classmethod
    def names(cls):
        """Returns a list of all enum member names."""
        return [member.name for member in cls]

    @classmethod
    def values(cls):
        """Returns a list of all enum member values."""
        return [member.value for member in cls]


class MultiStrEnum(str, MultiValueEnum):

    # --- Lifecycle & Initialization ---
    @classmethod
    def _missing_(cls, value):
        if value is None:
            # 1. Route 'None' to act exactly as if the user typed "default"
            try:
                return cls("default")
            except ValueError:
                # 2. Safety net: If this specific Enum doesn't have a "default" alias,
                # safely fall back to the very first item in the Enum.
                return list(cls)[0]

        # 3. If it's not None, and not a valid string, crash cleanly.
        raise ValueError(f"'{value}' is not a valid '{cls.__name__}' extension.")

    # --- Representation ---
    def __str__(self) -> str:
        """Informal string representation for end-users (print)."""
        return self.value

    def __format__(self, format_spec: str) -> str:
        """Custom behavior for f-string formatting."""
        return str.__format__(self.value, format_spec)

    # --- Retrieval ---
    @classmethod
    def names(cls) -> list[str]:
        """Returns a list of all enum member names."""
        return [member.name for member in cls]

    @classmethod
    def values(cls) -> list[str]:
        """Returns a list of all primary enum member values."""
        return [member.value for member in cls]

    @classmethod
    def all_values(cls) -> list[str]:
        """Returns a list of all enum member values, including aliases."""
        # _value2member_map_ contains every primary value and alias as its keys
        return list(cls._value2member_map_.keys())

    @classmethod
    def values_dict(cls) -> dict[str, tuple[str, ...]]:
        """Returns a dictionary mapping the enum name to all its values."""
        return {member.name: member._values_ for member in cls}

    @classmethod
    def values_repr(cls, aligned: bool = True) -> list[str]:
        """Returns a list of strings formatted as 'primary (extra, ...)'.

        Args:
            aligned (bool, optional): Whether to align the primary values for
                better readability. Defaults to True.
        """
        # 1. Find the longest primary value to know how much to pad
        max_len = max(len(member.value) for member in cls)

        formatted_items = []
        for member in cls:
            primary = member.value
            extras = member._values_[1:]  # Get everything after the first value

            padded_primary = ""
            if aligned:
                # 2. Pad the primary string with spaces so they all match max_len
                # e.g., f"{'error':<{max_len}}" becomes "error  "
                padded_primary = f"{primary:<{max_len}}"

            if extras:
                # Join the extra values with a comma and space
                extras_str = ", ".join(extras)
                formatted_items.append(f"{padded_primary}  ({extras_str})")
            else:
                # If there are no aliases, just append the primary value
                formatted_items.append(f"{padded_primary}")

        return formatted_items

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
