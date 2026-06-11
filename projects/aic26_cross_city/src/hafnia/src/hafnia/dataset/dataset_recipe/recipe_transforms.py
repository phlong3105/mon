from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from pydantic import field_validator

from hafnia.dataset.dataset_recipe.recipe_types import RecipeTransform
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.primitives.primitive import Primitive


class Shuffle(RecipeTransform):
    seed: int = 42

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.shuffle


class SelectSamples(RecipeTransform):
    n_samples: int
    shuffle: bool = True
    seed: int = 42
    with_replacement: bool = False

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.select_samples


class SplitsByRatios(RecipeTransform):
    split_ratios: Dict[str, float]
    seed: int = 42

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.splits_by_ratios


class SplitIntoMultipleSplits(RecipeTransform):
    split_name: str
    split_ratios: Dict[str, float]

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.split_into_multiple_splits


class DefineSampleSetBySize(RecipeTransform):
    n_samples: int
    seed: int = 42

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.define_sample_set_by_size


class ClassMapper(RecipeTransform):
    class_mapping: Union[Dict[str, str], List[Tuple[str, str]]]
    method: str = "strict"
    primitive: Optional[Type[Primitive]] = None
    task_name: Optional[str] = None

    @field_validator("class_mapping", mode="after")
    @classmethod
    def serialize_class_mapping(cls, value: Union[Dict[str, str], List[Tuple[str, str]]]) -> List[Tuple[str, str]]:
        # Converts the dictionary class mapping to a list of tuples
        #  e.g. {"old_class": "new_class", } --> [("old_class", "new_class")]
        # The reason is that storing class mappings as a dictionary does not preserve order of json fields
        # when stored in a database as a jsonb field (postgres).
        # Preserving order of class mapping fields is important as it defines the indices of the classes.
        # So to ensure that class indices are maintained, we preserve order of json fields, by converting the
        # dictionary to a list of tuples.
        if isinstance(value, dict):
            value = list(value.items())
        return value

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.class_mapper


class RenameTask(RecipeTransform):
    old_task_name: str
    new_task_name: str

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.rename_task


class SelectSamplesByClassName(RecipeTransform):
    name: Union[List[str], str]
    task_name: Optional[str] = None
    primitive: Optional[Type[Primitive]] = None

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.select_samples_by_class_name


class DropSamplesByClassName(RecipeTransform):
    name: Union[List[str], str]
    task_name: Optional[str] = None
    primitive: Optional[Type[Primitive]] = None
    drop_classes_from_task_info: bool = True

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.drop_samples_by_class_name
