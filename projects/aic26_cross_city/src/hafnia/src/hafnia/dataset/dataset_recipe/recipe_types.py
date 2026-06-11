from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel, computed_field

from hafnia import utils

if TYPE_CHECKING:  # Using 'TYPE_CHECKING' to avoid circular imports during type checking
    from hafnia.dataset.hafnia_dataset import HafniaDataset


class Serializable(BaseModel, ABC):
    @computed_field  # type: ignore[prop-decorator]
    @property
    def __type__(self) -> str:
        return self.__class__.__name__

    @classmethod
    def get_nested_subclasses(cls) -> List[type["Serializable"]]:
        """Recursively get all subclasses of a class."""
        all_subclasses = []
        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(subclass.get_nested_subclasses())
        return all_subclasses

    @classmethod
    def name_to_type_mapping(cls) -> Dict[str, type["Serializable"]]:
        """Create a mapping from class names to class types."""
        return {subclass.__name__: subclass for subclass in cls.get_nested_subclasses()}

    @staticmethod
    def from_dict(data: Dict) -> "Serializable":
        dataset_spec_args = data.copy()
        dataset_type_name = dataset_spec_args.pop("__type__", None)
        name_to_type_mapping = Serializable.name_to_type_mapping()
        SerializableClass = name_to_type_mapping[dataset_type_name]
        return SerializableClass(**dataset_spec_args)

    def get_kwargs(self, keep_default_fields: bool) -> Dict:
        """Return a dictionary of fields that are not set to their default values."""
        kwargs = dict(self)
        kwargs.pop("__type__", None)

        if keep_default_fields:
            return kwargs

        kwargs_no_defaults = {}
        for key, value in kwargs.items():
            default_value = self.model_fields[key].get_default()
            if value != default_value:
                kwargs_no_defaults[key] = value

        return kwargs_no_defaults

    @abstractmethod
    def as_short_name(self) -> str:
        pass

    def as_python_code(self, keep_default_fields: bool = False, as_kwargs: bool = True) -> str:
        """Generate code representation of the operation."""
        kwargs = self.get_kwargs(keep_default_fields=keep_default_fields)

        args_as_strs = []
        for argument_name, argument_value in kwargs.items():
            # In case an argument is a Serializable, we want to keep its default fields
            str_value = recursive_as_code(argument_value, keep_default_fields=keep_default_fields, as_kwargs=as_kwargs)
            if as_kwargs:
                args_as_strs.append(f"{argument_name}={str_value}")
            else:
                args_as_strs.append(str_value)

        args_as_str = ", ".join(args_as_strs)
        class_name = self.__class__.__name__
        function_name = utils.pascal_to_snake_case(class_name)
        return f"{function_name}({args_as_str})"


def recursive_as_code(value: Any, keep_default_fields: bool = False, as_kwargs: bool = True) -> str:
    if isinstance(value, Serializable):
        return value.as_python_code(keep_default_fields=keep_default_fields, as_kwargs=as_kwargs)

    elif isinstance(value, list):
        as_strs = []
        for item in value:
            str_item = recursive_as_code(item, keep_default_fields=keep_default_fields, as_kwargs=as_kwargs)
            as_strs.append(str_item)
        as_str = ", ".join(as_strs)
        return f"[{as_str}]"

    elif isinstance(value, dict):
        as_strs = []
        for key, item in value.items():
            str_item = recursive_as_code(item, keep_default_fields=keep_default_fields, as_kwargs=as_kwargs)
            as_strs.append(f"{key!r}: {str_item}")
        as_str = ", ".join(as_strs)
        return "{" + as_str + "}"

    return f"{value!r}"


class RecipeCreation(Serializable):
    @staticmethod
    @abstractmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        pass

    @abstractmethod
    def get_dataset_names(self) -> List[str]:
        pass

    def build(self, download_files: bool = True) -> "HafniaDataset":
        from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe

        kwargs = dict(self)
        kwargs_recipes_as_datasets = {}
        for key, value in kwargs.items():
            if isinstance(value, DatasetRecipe):
                value = value.build(download_files=download_files)
                key = key.replace("recipe", "dataset")
            kwargs_recipes_as_datasets[key] = value
        return self.get_function()(**kwargs_recipes_as_datasets)

    def as_python_code(self, keep_default_fields: bool = False, as_kwargs: bool = True) -> str:
        """Generate code representation of the operation."""
        as_python_code = Serializable.as_python_code(self, keep_default_fields=keep_default_fields, as_kwargs=as_kwargs)
        return f"DatasetRecipe.{as_python_code}"


class RecipeTransform(Serializable):
    @staticmethod
    @abstractmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        pass

    def build(self, dataset: "HafniaDataset") -> "HafniaDataset":
        kwargs = dict(self)
        return self.get_function()(dataset=dataset, **kwargs)

    def as_short_name(self) -> str:
        return self.__class__.__name__
