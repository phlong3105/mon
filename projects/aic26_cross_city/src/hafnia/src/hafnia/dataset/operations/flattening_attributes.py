from typing import List, Tuple

import polars as pl

from hafnia.dataset.dataset_names import PrimitiveField, SampleField
from hafnia.dataset.hafnia_dataset_types import ClassInfo, TaskInfo
from hafnia.dataset.operations.table_transformations import update_class_indices


def _lookup_attr_expr(classifications_expr: pl.Expr, attr_path: List[str]) -> pl.Expr:
    """Build a native Polars expression that walks a ``List(Struct)`` classifications
    column following *attr_path* and returns the ``class_name`` at the deepest matched
    level, or ``null`` if any level is not found.

    Each step uses ``list.eval`` (no Python UDF) so the whole chain runs inside the
    Polars engine.
    """
    attr_name = attr_path[0]
    remaining = attr_path[1:]

    if not remaining:
        # Base case: return class_name where task_name == attr_name
        return (
            classifications_expr.list.eval(
                pl.when(pl.element().struct.field(PrimitiveField.TASK_NAME) == attr_name)
                .then(pl.element().struct.field(PrimitiveField.CLASS_NAME))
                .otherwise(None)
            )
            .list.drop_nulls()
            .list.first()
        )

    # Recursive case: navigate into the sub-classifications of the matched entry
    sub_classifications = (
        classifications_expr.list.eval(
            pl.when(pl.element().struct.field(PrimitiveField.TASK_NAME) == attr_name)
            .then(pl.element().struct.field(SampleField.CLASSIFICATIONS))
            .otherwise(None)
        )
        .list.drop_nulls()
        .list.first()  # List(Struct) – the sub-classifications of the matched entry
    )
    return _lookup_attr_expr(sub_classifications, remaining)


def _expand_class_info(
    class_info: ClassInfo,
    attr_path: List[str],
    separator: str = ".",
) -> List[ClassInfo]:
    """Recursively expand a ClassInfo by walking ``attr_path`` through nested attributes.

    Examples (path=["Vehicle Color", "Gray Tone"])::

        Vehicle(Red)        -> ClassInfo("Vehicle.Red",  other_attrs...)
        Vehicle(Gray,light) -> ClassInfo("Vehicle.Gray.light", other_attrs...)
        Person              -> ClassInfo("Person", attrs unchanged)
    """
    if not attr_path:
        return [class_info]

    attr_name = attr_path[0]
    remaining = attr_path[1:]
    attrs = class_info.attributes or []
    matching_attr = next((a for a in attrs if a.name == attr_name), None)

    if matching_attr is None:
        return [class_info]

    other_attrs = [a for a in attrs if a.name != attr_name]
    base_class = ClassInfo(
        name=class_info.name,
        attributes=other_attrs if other_attrs else None,
    )
    result: List[ClassInfo] = []
    for sub_class in matching_attr.classes or []:
        for expanded in _expand_class_info(sub_class, remaining, separator):
            merged_attrs = other_attrs + (expanded.attributes or [])
            result.append(
                ClassInfo(
                    name=f"{class_info.name}{separator}{expanded.name}",
                    attributes=merged_attrs if merged_attrs else None,
                )
            )
    return [base_class] + result


def flatten_class_names(
    samples: pl.DataFrame,
    task_info: TaskInfo,
    flatten_types: List[str],
    separator: str = ".",
) -> Tuple[pl.DataFrame, TaskInfo]:
    """Flatten nested classification attributes into the annotation ``class_name``

    For each annotation in *col_name* whose ``task_name`` matches, the function
    builds nested ``list.eval`` chains to walk *attribute_types* through the
    ``classifications`` sub-lists and concatenate the matched ``class_name`` values.

    Example::

        flatten_class_names(samples, bbox_task, ["Vehicle Color", "Gray Tone"])

        "Vehicle" (color=Gray, tone=very light) → "Vehicle.Gray.very light"
        "Vehicle" (color=Red)                   → "Vehicle.Red"
        "Person"                                → "Person"  # no Vehicle Color attr

    Args:
        samples:   Polars ``DataFrame`` of samples.
        task_info: :class:`TaskInfo` for the annotation task to flatten.
        flatten_types: Ordered attribute names to fold into ``class_name``,
                       e.g. ``["Vehicle Color", "Gray Tone"]``.
        separator: Separator between levels (default ``"."``).

    Returns:
        ``(updated_samples, new_task_info)``
    """
    task_column_name = task_info.primitive.column_name()

    # Update TaskInfo
    new_classes: List[ClassInfo] = []
    for cls in task_info.classes or []:
        new_classes.extend(_expand_class_info(cls, flatten_types, separator))
    new_task_info = TaskInfo(
        primitive=task_info.primitive,
        name=task_info.name,
        classes=new_classes,
    )

    # Update Samples
    struct_fields = samples[task_column_name].dtype.inner.fields

    if has_null_attribute_column(struct_fields):
        samples_updated = update_class_indices(samples, new_task_info)
        return samples_updated, new_task_info

    classifications_expr = pl.element().struct.field(SampleField.CLASSIFICATIONS)
    is_target_task = pl.element().struct.field(PrimitiveField.TASK_NAME) == task_info.name
    base_name = pl.element().struct.field(PrimitiveField.CLASS_NAME)

    name_expr = base_name
    for i in range(len(flatten_types)):
        suffix = _lookup_attr_expr(classifications_expr, flatten_types[: i + 1])
        name_expr = name_expr + pl.when(suffix.is_not_null()).then(pl.lit(separator) + suffix).otherwise(pl.lit(""))

    new_class_name = pl.when(is_target_task).then(name_expr).otherwise(base_name)

    samples_updated = samples.with_columns(
        pl.col(task_column_name)
        .list.eval(pl.element().struct.with_fields(new_class_name.alias(PrimitiveField.CLASS_NAME)))
        .alias(task_column_name)
    )

    samples_updated = update_class_indices(samples_updated, new_task_info)
    return samples_updated, new_task_info


def has_null_attribute_column(struct_fields: List[pl.Field]) -> bool:
    has_no_classifications = SampleField.CLASSIFICATIONS not in [f.name for f in struct_fields]
    if has_no_classifications:
        return True
    null_type = [f for f in struct_fields if f.name == SampleField.CLASSIFICATIONS and f.dtype == pl.Null]
    if null_type:
        return True
    return False
