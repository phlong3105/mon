from __future__ import annotations

from typing import List, Type

from .bbox import Bbox
from .bitmask import Bitmask
from .classification import Classification
from .point import Point  # noqa: F401
from .polygon import Polygon
from .primitive import Primitive
from .segmentation import Segmentation  # noqa: F401
from .utils import class_color_by_name  # noqa: F401

# Rebuild models to resolve forward references after all types are imported
Bbox.model_rebuild()
Bitmask.model_rebuild()
Polygon.model_rebuild()
Classification.model_rebuild()


PRIMITIVE_TYPES: List[Type[Primitive]] = [Bbox, Classification, Polygon, Bitmask]
PRIMITIVE_NAME_TO_TYPE = {cls.__name__: cls for cls in PRIMITIVE_TYPES}
PRIMITIVE_COLUMN_NAMES: List[str] = [PrimitiveType.column_name() for PrimitiveType in PRIMITIVE_TYPES]


def get_primitive_type_from_string(name: str) -> Type[Primitive]:
    if name not in PRIMITIVE_NAME_TO_TYPE:
        raise ValueError(
            f"Primitive '{name}' is not recognized. Available primitives: {list(PRIMITIVE_NAME_TO_TYPE.keys())}"
        )
    return PRIMITIVE_NAME_TO_TYPE[name]
