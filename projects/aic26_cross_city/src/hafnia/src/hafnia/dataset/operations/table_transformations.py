from pathlib import Path
from typing import List, Optional, Tuple, Type

import polars as pl

from hafnia.dataset.dataset_names import (
    PrimitiveField,
    SampleField,
)
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.operations import table_transformations
from hafnia.dataset.primitives import PRIMITIVE_TYPES, Classification
from hafnia.dataset.primitives.primitive import Primitive
from hafnia.log import user_logger
from hafnia.utils import progress_bar


def create_primitive_table(
    samples_table: pl.DataFrame,
    PrimitiveType: Type[Primitive],
    keep_sample_data: bool = False,
    task_name: Optional[str] = None,
) -> Optional[pl.DataFrame]:
    """
    Returns a DataFrame with objects of the specified primitive type.
    """
    if not has_primitive(samples_table, PrimitiveType):
        return None

    column_name = PrimitiveType.column_name()

    # Remove frames without objects
    remove_no_object_frames = samples_table.filter(pl.col(column_name).list.len() > 0)

    if keep_sample_data:
        # Drop other primitive columns to avoid conflicts

        drop_columns_primitives = set(PRIMITIVE_TYPES) - {PrimitiveType, Classification}
        drop_columns_names = [primitive.column_name() for primitive in drop_columns_primitives]
        drop_columns_names = [c for c in drop_columns_names if c in remove_no_object_frames.columns]

        remove_no_object_frames = remove_no_object_frames.drop(drop_columns_names)
        # Rename columns "height", "width" and "meta" for sample to avoid conflicts with object fields names
        remove_no_object_frames = remove_no_object_frames.rename(
            {"height": "image.height", "width": "image.width", "meta": "image.meta"},
            strict=False,
        )
        # tmp_column_name = "tmp_column_name_will_be_dropped_after_unnest"
        # remove_no_object_frames = remove_no_object_frames.rename({column_name: tmp_column_name}, strict=True)
        row_per_object = remove_no_object_frames.explode(column_name)
        nested_fields = row_per_object[column_name].struct.fields
        existing_fields = set(row_per_object.columns)
        overlap_fields = set(nested_fields).intersection(existing_fields)
        if len(overlap_fields) > 0:
            # To avoid conflicts struct fields are renamed by column_name + "." + field_name (e.g. "bboxes.xmin", "bboxes.ymin" etc.)
            nested_fields_renamed = []
            for field in nested_fields:
                if field in overlap_fields:
                    nested_fields_renamed.append(f"{column_name}.{field}")
                else:
                    nested_fields_renamed.append(field)

            row_per_object = row_per_object.with_columns(
                pl.col(column_name).struct.rename_fields(nested_fields_renamed).alias(column_name)
            )

        objects_df = row_per_object.unnest(column_name)
    else:
        objects_df = remove_no_object_frames.select(pl.col(column_name).explode().struct.unnest())

    if task_name is not None:
        objects_df = objects_df.filter(pl.col(PrimitiveField.TASK_NAME) == task_name)
    return objects_df


def has_primitive(samples: pl.DataFrame, PrimitiveType: Type[Primitive]) -> bool:
    col_name = PrimitiveType.column_name()
    if col_name not in samples.columns:
        return False

    if samples[col_name].dtype != pl.List(pl.Struct):
        return False

    return True


def merge_samples(samples0: pl.DataFrame, samples1: pl.DataFrame) -> pl.DataFrame:
    has_same_schema = samples0.schema == samples1.schema
    if not has_same_schema:
        shared_columns = []
        for column_name, s0_column_type in samples0.schema.items():
            if column_name not in samples1.schema:
                continue
            samples0, samples1 = correction_of_list_struct_primitives(samples0, samples1, column_name)
            column0_type = samples0.schema[column_name]
            column1_type = samples1.schema[column_name]

            both_are_structs = isinstance(column0_type, pl.Struct) and isinstance(column1_type, pl.Struct)
            one_is_struct_other_is_null0 = isinstance(column0_type, pl.Struct) and column1_type == pl.Null
            one_is_struct_other_is_null1 = isinstance(column1_type, pl.Struct) and column0_type == pl.Null
            if both_are_structs or one_is_struct_other_is_null0 or one_is_struct_other_is_null1:
                pass  # Keep Struct columns even if they do not match exactly.
            elif column0_type != column1_type:
                continue
            shared_columns.append(column_name)

        dropped_columns0 = [
            f"{n}[{ctype._string_repr()}]" for n, ctype in samples0.schema.items() if n not in shared_columns
        ]
        dropped_columns1 = [
            f"{n}[{ctype._string_repr()}]" for n, ctype in samples1.schema.items() if n not in shared_columns
        ]
        user_logger.warning(
            "Datasets with different schemas are being merged. "
            "Only the columns with the same name and type will be kept in the merged dataset.\n"
            f"Dropped columns in samples0: {dropped_columns0}\n"
            f"Dropped columns in samples1: {dropped_columns1}\n"
        )

        samples0 = samples0.select(list(shared_columns))
        samples1 = samples1.select(list(shared_columns))
    merged_samples = pl.concat([samples0, samples1], how="vertical_relaxed")
    merged_samples = add_sample_index(merged_samples)
    return merged_samples


def correction_of_list_struct_primitives(
    samples0: pl.DataFrame,
    samples1: pl.DataFrame,
    column_name: str,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Corrects primitive columns (bboxes, polygons etc of type 'list[struct]') by removing non-matching struct fields
    between two datasets. This is useful when merging two datasets with the same primitive (e.g. Bbox), where
    some (less important) field types in the struct differ between the two datasets.
    This issue often occurs with the 'meta' field as different dataset formats may store different metadata information.
    """
    s0_column_type = samples0.schema[column_name]
    s1_column_type = samples1.schema[column_name]
    is_list_structs = s1_column_type == pl.List(pl.Struct) and s0_column_type == pl.List(pl.Struct)
    is_non_matching_types = s1_column_type != s0_column_type
    if is_list_structs and is_non_matching_types:  # Only perform correction for list[struct] types that do not match
        s0_fields = set(s0_column_type.inner.fields)
        s1_fields = set(s1_column_type.inner.fields)
        similar_fields = s0_fields.intersection(s1_fields)
        s0_dropped_fields = s0_fields - similar_fields
        if len(s0_dropped_fields) > 0:
            samples0 = samples0.with_columns(
                pl.col(column_name)
                .list.eval(pl.struct([pl.element().struct.field(k.name) for k in similar_fields]))
                .alias(column_name)
            )
        s1_dropped_fields = s1_fields - similar_fields
        if len(s1_dropped_fields) > 0:
            samples1 = samples1.with_columns(
                pl.col(column_name)
                .list.eval(pl.struct([pl.element().struct.field(k.name) for k in similar_fields]))
                .alias(column_name)
            )
        user_logger.warning(
            f"Primitive column '{column_name}' has none-matching fields in the two datasets. "
            f"Dropping fields in samples0: {[f.name for f in s0_dropped_fields]}. "
            f"Dropping fields in samples1: {[f.name for f in s1_dropped_fields]}."
        )

    return samples0, samples1


def filter_table_for_class_names(
    samples_table: pl.DataFrame, class_names: List[str], PrimitiveType: Type[Primitive]
) -> Optional[pl.DataFrame]:
    table_with_selected_class_names = samples_table.filter(
        pl.col(PrimitiveType.column_name())
        .list.eval(pl.element().struct.field(PrimitiveField.CLASS_NAME).is_in(class_names))
        .list.any()
    )

    return table_with_selected_class_names


def split_primitive_columns_by_task_name(
    samples_table: pl.DataFrame,
    coordinate_types: Optional[List[Type[Primitive]]] = None,
) -> pl.DataFrame:
    """
    Convert Primitive columns such as "bboxes" (Bbox) into a column for each task name.
    For example, if the "bboxes" column (containing Bbox objects) has tasks "task1" and "task2".


    This:
    ─┬────────────┬─
     ┆ bboxes    ┆
     ┆ ---        ┆
     ┆ list[struc ┆
     ┆ t[11]]     ┆
    ═╪════════════╪═
    becomes this:
    ─┬────────────┬────────────┬─
     ┆ bboxes.   ┆ bboxes.   ┆
     ┆ task1      ┆ task2      ┆
     ┆ ---        ┆ ---        ┆
     ┆ list[struc ┆ list[struc ┆
     ┆ t[11]]     ┆ t[13]]     ┆
    ═╪════════════╪════════════╪═

    """
    coordinate_types = coordinate_types or PRIMITIVE_TYPES
    for PrimitiveType in coordinate_types:
        col_name = PrimitiveType.column_name()

        if col_name not in samples_table.columns:
            continue

        if samples_table[col_name].dtype != pl.List(pl.Struct):
            continue

        task_names = samples_table[col_name].explode().struct.field(PrimitiveField.TASK_NAME).unique().to_list()
        samples_table = samples_table.with_columns(
            [
                pl.col(col_name)
                .list.filter(pl.element().struct.field(PrimitiveField.TASK_NAME).eq(task_name))
                .alias(f"{col_name}.{task_name}")
                for task_name in task_names
            ]
        )
        samples_table = samples_table.drop(col_name)
    return samples_table


def check_image_paths(table: pl.DataFrame) -> bool:
    missing_files = []
    org_paths = table[SampleField.FILE_PATH].to_list()
    for org_path in progress_bar(org_paths, description="Check image paths"):
        org_path = Path(org_path)
        if not org_path.exists():
            missing_files.append(org_path)

    if len(missing_files) > 0:
        user_logger.warning(f"Missing files: {len(missing_files)}. Show first 5:")
        for missing_file in missing_files[:5]:
            user_logger.warning(f" - {missing_file}")
        raise FileNotFoundError(f"Some files are missing in the dataset: {len(missing_files)} files not found.")

    return True


def unnest_classification_tasks(table: pl.DataFrame, strict: bool = True) -> pl.DataFrame:
    """
    Unnest classification tasks in table.
    Classificiations tasks are all stored in the same column in the HafniaDataset table.
    This function splits them into separate columns for each task name.

    Type is converted from a list of structs (pl.List[pl.Struct]) to a struct (pl.Struct) column.

    Converts classification column from this:
       ─┬─────────────────┬─
        ┆ classifications ┆
        ┆ ---             ┆
        ┆ list[struct[6]] ┆
       ═╪═════════════════╪═

    For example, if the classification column has tasks "task1" and "task2",
       ─┬──────────────────┬──────────────────┬─
        ┆ classifications. ┆ classifications. ┆
        ┆ task1            ┆ task2            ┆
        ┆ ---              ┆ ---              ┆
        ┆ struct[6]        ┆ struct[6]        ┆
       ═╪══════════════════╪══════════════════╪═

    """
    coordinate_types = [Classification]
    table_out = table_transformations.split_primitive_columns_by_task_name(table, coordinate_types=coordinate_types)

    classification_columns = [c for c in table_out.columns if c.startswith(Classification.column_name() + ".")]
    for classification_column in classification_columns:
        has_multiple_items_per_sample = all(table_out[classification_column].list.len() > 1)
        if has_multiple_items_per_sample:
            if strict:
                raise ValueError(
                    f"Column {classification_column} has multiple items per sample, but expected only one item."
                )
            else:
                user_logger.warning(
                    f"Warning: Unnesting of column '{classification_column}' is skipped because it has multiple items per sample."
                )

    table_out = table_out.with_columns([pl.col(c).list.first() for c in classification_columns])
    return table_out


def update_class_indices(samples: pl.DataFrame, task: TaskInfo) -> pl.DataFrame:
    class_names = task.get_class_names()
    if class_names is None or len(class_names) == 0:
        raise ValueError(f"Task '{task.name}' does not have defined class names to update class indices.")

    objs = (
        samples[task.primitive.column_name()]
        .explode()
        .struct.unnest()
        .filter(pl.col(PrimitiveField.TASK_NAME) == task.name)
    )
    expected_class_names = set(objs[PrimitiveField.CLASS_NAME].unique())
    missing_class_names = expected_class_names - set(class_names)
    if len(missing_class_names) > 0:
        raise ValueError(
            f"Task '{task.name}' is missing class names: {missing_class_names}. Cannot update class indices."
        )

    name_2_idx_mapping = {name: idx for idx, name in enumerate(class_names)}

    samples_updated = samples.with_columns(
        pl.col(task.primitive.column_name())
        .list.eval(
            pl.element().struct.with_fields(
                pl.when(pl.field(PrimitiveField.TASK_NAME) == task.name)
                .then(pl.field(PrimitiveField.CLASS_NAME).replace_strict(name_2_idx_mapping, default=-1))
                .otherwise(pl.field(PrimitiveField.CLASS_IDX))
                .alias(PrimitiveField.CLASS_IDX)
            )
        )
        .alias(task.primitive.column_name())
    )

    return samples_updated


def add_sample_index(samples: pl.DataFrame) -> pl.DataFrame:
    """
    Adds a sample index column to the samples DataFrame.

    Note: Unlike the built-in 'polars.DataFrame.with_row_count', this function
    always guarantees 'pl.UInt64' type for the index column.
    """
    if SampleField.SAMPLE_INDEX in samples.columns:
        samples = samples.drop(SampleField.SAMPLE_INDEX)
    samples = samples.select(
        pl.int_range(0, pl.len(), dtype=pl.UInt64).alias(SampleField.SAMPLE_INDEX),
        pl.all(),
    )
    return samples


def add_dataset_name_if_missing(table: pl.DataFrame, dataset_name: str) -> pl.DataFrame:
    if SampleField.DATASET_NAME not in table.columns:
        table = table.with_columns(pl.lit(dataset_name).alias(SampleField.DATASET_NAME))
    else:
        table = table.with_columns(
            pl.when(pl.col(SampleField.DATASET_NAME).is_null())
            .then(pl.lit(dataset_name))
            .otherwise(pl.col(SampleField.DATASET_NAME))
            .alias(SampleField.DATASET_NAME)
        )

    return table
