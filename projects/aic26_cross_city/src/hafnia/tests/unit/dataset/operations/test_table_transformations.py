from typing import List, Type

import polars as pl
import pytest

from hafnia.dataset.dataset_names import PrimitiveField
from hafnia.dataset.operations import table_transformations
from hafnia.dataset.operations.table_transformations import unnest_classification_tasks
from hafnia.dataset.primitives import Bbox, Bitmask, Classification
from hafnia.dataset.primitives.primitive import Primitive
from tests import helper_testing


@pytest.mark.parametrize("dataset_name", helper_testing.MICRO_DATASETS)
def test_create_primitive_table(dataset_name: str):
    hafnia_dataset = helper_testing.get_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    hafnia_dataset.samples

    PrimitiveTypes = [Classification, Bbox, Bitmask]

    for PrimitiveType in PrimitiveTypes:
        n_primitive_fields = len(PrimitiveType.model_fields)
        only_primitives = table_transformations.create_primitive_table(
            samples_table=hafnia_dataset.samples,
            PrimitiveType=PrimitiveType,  # type: ignore[type-abstract]
            keep_sample_data=False,
        )
        if only_primitives is not None:
            assert len(only_primitives.columns) <= n_primitive_fields

        all_columns = table_transformations.create_primitive_table(
            samples_table=hafnia_dataset.samples,
            PrimitiveType=PrimitiveType,  # type: ignore[type-abstract]
            keep_sample_data=True,
        )

        if all_columns is not None:
            assert len(all_columns.columns) > n_primitive_fields


def test_filter_table_for_class_names():
    hafnia_dataset = helper_testing.get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset", force_update=False)

    n_samples_before_filtering = len(hafnia_dataset.samples)
    table_after = table_transformations.filter_table_for_class_names(
        samples_table=hafnia_dataset.samples,
        class_names=["Vehicle.Car"],
        PrimitiveType=Bbox,
    )

    n_samples_after_filtering = len(table_after)
    assert n_samples_after_filtering < n_samples_before_filtering


def test_split_primitive_columns_by_task_name():
    dataset = helper_testing.get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset", force_update=False)
    table = dataset.samples

    def check_expected_column_names(table_before, table_after, PrimitiveTypes: List[Type[Primitive]]):
        for PrimitiveType in PrimitiveTypes:
            assert PrimitiveType.column_name() not in table_after.columns
            task_names = (
                table_before[PrimitiveType.column_name()]
                .explode()
                .struct.field(PrimitiveField.TASK_NAME)
                .unique()
                .to_list()
            )
            for task_name in task_names:
                assert f"{PrimitiveType.column_name()}.{task_name}" in table_after.columns

    coordinate_types = [Classification]
    table_out = table_transformations.split_primitive_columns_by_task_name(table, coordinate_types=coordinate_types)
    check_expected_column_names(table_before=table, table_after=table_out, PrimitiveTypes=coordinate_types)

    coordinate_types = [Classification, Bbox]
    table_out = table_transformations.split_primitive_columns_by_task_name(table, coordinate_types=coordinate_types)
    check_expected_column_names(table_before=table, table_after=table_out, PrimitiveTypes=coordinate_types)

    coordinate_types = [Classification, Bbox]
    table_out = table_transformations.split_primitive_columns_by_task_name(table)
    check_expected_column_names(table_before=table, table_after=table_out, PrimitiveTypes=coordinate_types)


def test_unnest_classification_tasks():
    dataset = helper_testing.get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset", force_update=False)
    table = dataset.samples

    table_unnested = unnest_classification_tasks(table)

    class_tasks = (
        table[Classification.column_name()].explode().struct.field(PrimitiveField.TASK_NAME).unique().to_list()
    )

    for task_name in class_tasks:
        expected_column_name = f"{Classification.column_name()}.{task_name}"
        assert expected_column_name in table_unnested.columns
        assert table_unnested[expected_column_name].dtype == pl.Struct
