from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from pydantic import (
    field_serializer,
    field_validator,
)

from hafnia import utils
from hafnia.dataset.dataset_helpers import dataset_name_and_version_from_string
from hafnia.dataset.dataset_names import FILENAME_RECIPE_JSON
from hafnia.dataset.dataset_recipe import recipe_transforms
from hafnia.dataset.dataset_recipe.recipe_types import (
    RecipeCreation,
    RecipeTransform,
    Serializable,
)
from hafnia.dataset.hafnia_dataset import (
    HafniaDataset,
    available_dataset_versions_from_name,
)
from hafnia.dataset.hafnia_dataset_types import DatasetMetadataFilePaths
from hafnia.dataset.primitives.primitive import Primitive
from hafnia.log import user_logger


class DatasetRecipe(Serializable):
    creation: RecipeCreation
    operations: Optional[List[RecipeTransform]] = None

    def build(self, download_files: bool = True) -> HafniaDataset:
        """Materialize the recipe into a concrete `HafniaDataset`.

        Runs the creation step (e.g. `from_name`, `from_path`, `from_merge`, ...) and then applies
        each registered operation in order. Until `build` is called, a `DatasetRecipe` is just a
        specification — no I/O or downloads happen.

        Args:
            download_files: If True, download image files to disk during creation. Set to False to
                materialize annotations only (e.g. for cache lookups or recipe inspection).
        """
        if hasattr(self.creation, "download_files"):
            self.creation.download_files = download_files
        dataset = self.creation.build(download_files=download_files)
        if self.operations:
            for operation in self.operations:
                dataset = operation.build(dataset)
        return dataset

    def append_operation(self, operation: RecipeTransform) -> DatasetRecipe:
        """Append a `RecipeTransform` to the recipe's operations list and return the recipe.

        Used internally by the transformation methods (`shuffle`, `select_samples`, ...) to chain
        operations. Most callers should use those instead of constructing `RecipeTransform`
        instances directly.

        Args:
            operation: The operation to append.
        """
        if self.operations is None:
            self.operations = []
        self.operations.append(operation)
        return self

    ### Creation Methods (using the 'from_X' )###
    @staticmethod
    def from_name(
        name: str,
        version: Optional[str] = None,
        force_redownload: bool = False,
        download_files: bool = True,
    ) -> DatasetRecipe:
        """Create a recipe that loads a Hafnia-platform dataset by name when built.

        Recipe equivalent of `HafniaDataset.from_name`. A specific `version` is required so that
        the recipe is reproducible — passing ``"latest"`` is resolved to the current latest version
        at recipe-creation time and a warning is logged.

        Args:
            name: Dataset name as registered on the platform.
            version: Pinned dataset version (e.g. ``"1.0.0"``). Required for reproducibility.
            force_redownload: Forwarded to the underlying loader at build time.
            download_files: Forwarded to the underlying loader at build time.
        """
        if version == "latest":
            user_logger.info(
                f"The dataset '{name}' in a dataset recipe uses 'latest' as version. For dataset recipes the "
                "version is pinned to a specific version. Consider specifying a specific version to ensure "
                "reproducibility of your experiments. "
            )
            available_versions = available_dataset_versions_from_name(name)
            version = str(max(available_versions))
        if version is None:
            available_versions = available_dataset_versions_from_name(name)
            str_versions = ", ".join([str(v) for v in available_versions])
            raise ValueError(
                f"Version must be specified when creating a DatasetRecipe from name. "
                f"Available versions are: {str_versions}"
            )

        creation = FromName(
            name=name, version=version, force_redownload=force_redownload, download_files=download_files
        )
        return DatasetRecipe(creation=creation)

    @staticmethod
    def from_name_public_dataset(
        name: str, force_redownload: bool = False, n_samples: Optional[int] = None
    ) -> DatasetRecipe:
        """Create a recipe that loads a public dataset (e.g. a torchvision dataset) by name.

        Recipe equivalent of `HafniaDataset.from_name_public_dataset`. Does not require platform
        credentials.

        Args:
            name: Public dataset identifier supported by the Hafnia loaders.
            force_redownload: Forwarded to the underlying loader at build time.
            n_samples: Optional cap on the number of samples to materialize.
        """
        creation = FromNamePublicDataset(
            name=name,
            force_redownload=force_redownload,
            n_samples=n_samples,
        )
        return DatasetRecipe(creation=creation)

    @staticmethod
    def from_path(path_folder: Path, check_for_images: bool = True) -> DatasetRecipe:
        """Create a recipe that loads a `HafniaDataset` from a local folder when built.

        Recipe equivalent of `HafniaDataset.from_path`.

        Args:
            path_folder: Local dataset folder.
            check_for_images: Forwarded to the underlying loader at build time.
        """
        creation = FromPath(path_folder=path_folder, check_for_images=check_for_images)
        return DatasetRecipe(creation=creation)

    @staticmethod
    def from_merge(recipe0: DatasetRecipe, recipe1: DatasetRecipe) -> DatasetRecipe:
        """Create a recipe that merges two recipes into a single dataset when built.

        For merging more than two recipes, prefer `from_merger`, which avoids nested binary merges.
        """
        return DatasetRecipe(creation=FromMerge(recipe0=recipe0, recipe1=recipe1))

    @staticmethod
    def from_merger(recipes: List[DatasetRecipe]) -> DatasetRecipe:
        """Create a recipe that merges several recipes into a single dataset when built.

        Returns the original recipe unchanged if `recipes` has length one. Use `from_merge` for
        the two-recipe case if you prefer the binary-merge form.

        Args:
            recipes: Non-empty list of `DatasetRecipe` instances to merge.
        """
        if not recipes:
            raise ValueError("The list of recipes cannot be empty.")
        if len(recipes) == 1:
            return recipes[0]
        creation = FromMerger(recipes=recipes)
        return DatasetRecipe(creation=creation)

    @staticmethod
    def from_json_str(json_str: str) -> "DatasetRecipe":
        """Deserialize a `DatasetRecipe` from a JSON string produced by `as_json_str`.

        Raises:
            TypeError: If the JSON does not deserialize into a `DatasetRecipe` (e.g. it describes
                a different `Serializable` subclass).
        """
        data = json.loads(json_str)
        dataset_recipe = DatasetRecipe.from_dict(data)
        if not isinstance(dataset_recipe, DatasetRecipe):
            raise TypeError(f"Expected DatasetRecipe, got {type(dataset_recipe).__name__}.")
        return dataset_recipe

    @staticmethod
    def from_json_file(path_json: Path) -> "DatasetRecipe":
        """Deserialize a `DatasetRecipe` from a JSON file written by `as_json_file`."""
        json_str = path_json.read_text(encoding="utf-8")
        return DatasetRecipe.from_json_str(json_str)

    @staticmethod
    def from_recipe_field(recipe_field: Union[str, Dict[str, Any]]) -> "DatasetRecipe":
        """Deserialize a recipe from either a `name:version` string or a recipe dictionary.

        Used to accept both the shorthand (e.g. ``"mnist:1.0.0"``) and the full serialized form
        in API payloads and configuration files.

        Args:
            recipe_field: ``"name:version"`` string (delegates to `from_name_and_version_string`)
                or a dict produced by `as_dict` (delegates to `from_dict`).

        Raises:
            TypeError: If `recipe_field` is neither `str` nor `dict`.
        """
        if isinstance(recipe_field, str):
            return DatasetRecipe.from_name_and_version_string(recipe_field)
        elif isinstance(recipe_field, dict):
            return DatasetRecipe.from_dict(recipe_field)

        raise TypeError(f"Expected str or dict for recipe_field, got {type(recipe_field).__name__}.")

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DatasetRecipe":
        """Deserialize a `DatasetRecipe` from a dictionary produced by `as_dict`.

        The dict's `__class__` discriminator field is used to dispatch to the correct
        `Serializable` subclass before validation.
        """
        dataset_recipe = Serializable.from_dict(data)
        return dataset_recipe

    @staticmethod
    def from_recipe_id(recipe_id: str) -> "DatasetRecipe":
        """Fetch a dataset recipe stored on the Hafnia platform by its UUID.

        Uses the active `hafnia_cli.config.Config` for credentials. Equivalent to calling the CLI
        with `hafnia recipe ls` and looking up the recipe by id.

        Args:
            recipe_id: Platform-assigned recipe identifier.
        """
        from hafnia.platform.dataset_recipe import get_dataset_recipe_by_id
        from hafnia_cli.config import Config

        cfg = Config()
        recipe_dict = get_dataset_recipe_by_id(recipe_id, cfg=cfg)
        recipe_dict = recipe_dict["template"]["body"]
        return DatasetRecipe.from_recipe_field(recipe_dict)

    @staticmethod
    def from_recipe_name(name: str) -> "DatasetRecipe":
        """Fetch a dataset recipe stored on the Hafnia platform by its display name.

        Uses the active `hafnia_cli.config.Config` for credentials. Internally resolves the name
        to a recipe id and delegates to `from_recipe_id`.

        Args:
            name: Recipe name as registered on the platform.

        Raises:
            ValueError: If no recipe with the given name exists.
        """
        from hafnia.platform.dataset_recipe import get_dataset_recipe_by_name
        from hafnia_cli.config import Config

        cfg = Config()
        recipe = get_dataset_recipe_by_name(name=name, cfg=cfg)
        if not recipe:
            raise ValueError(f"Dataset recipe '{name}' not found.")
        recipe_id = recipe["id"]
        return DatasetRecipe.from_recipe_id(recipe_id)

    @staticmethod
    def from_name_and_version_string(string: str, resolve_missing_version: bool = False) -> "DatasetRecipe":
        """Build a `from_name` recipe from a ``"name:version"`` shorthand string.

        Args:
            string: Dataset shorthand. Either ``"name:version"`` or just ``"name"`` (only valid
                when `resolve_missing_version=True`).
            resolve_missing_version: If True, accept a missing version segment and default to
                ``"latest"``. If False (default), raise an error when the version is missing —
                this is the safer mode for reproducibility.
        """

        dataset_name, version = dataset_name_and_version_from_string(
            string=string,
            resolve_missing_version=resolve_missing_version,
        )

        return DatasetRecipe.from_name(name=dataset_name, version=version)

    ### Upload, store and recipe conversions ###
    def as_python_code(self, keep_default_fields: bool = False, as_kwargs: bool = True) -> str:
        """Render the recipe as the equivalent chained Python source (e.g. ``DatasetRecipe.from_name(...).shuffle(...)``).

        Useful for logging or copying a recipe back into a script.

        Args:
            keep_default_fields: If True, include arguments even when they equal their default
                values. Defaults to False to keep output compact.
            as_kwargs: If True, render arguments as ``name=value``; otherwise render positionally.
        """
        str_operations = [self.creation.as_python_code(keep_default_fields=keep_default_fields, as_kwargs=as_kwargs)]
        if self.operations:
            for op in self.operations:
                str_operations.append(op.as_python_code(keep_default_fields=keep_default_fields, as_kwargs=as_kwargs))
        operations_str = ".".join(str_operations)
        return operations_str

    def as_short_name(self) -> str:
        """Return a compact, human-readable summary of the recipe (e.g. ``Recipe(mnist,Shuffle,SelectSamples)``).

        Combines the creation step's short name with each operation's short name. Used as the
        prefix for the on-disk cache folder produced by `from_recipe_with_cache`.
        """

        creation_name = self.creation.as_short_name()
        if self.operations is None or len(self.operations) == 0:
            return creation_name
        short_names = [creation_name]
        for operation in self.operations:
            short_names.append(operation.as_short_name())
        transforms_str = ",".join(short_names)
        return f"Recipe({transforms_str})"

    def as_json_str(self, indent: int = 2) -> str:
        """Serialize the recipe to an indented JSON string. Round-trips via `from_json_str`.

        Args:
            indent: Number of spaces per indent level.
        """
        dict_data = self.as_dict()
        return json.dumps(dict_data, indent=indent, ensure_ascii=False)

    def as_json_file(self, path_json: Path, indent: int = 2) -> None:
        """Serialize the recipe to a JSON file (creating parent directories as needed).

        Round-trips via `from_json_file`.

        Args:
            path_json: Output path for the JSON file.
            indent: Number of spaces per indent level.
        """
        path_json.parent.mkdir(parents=True, exist_ok=True)
        json_str = self.as_json_str(indent=indent)
        path_json.write_text(json_str, encoding="utf-8")

    def as_dict(self) -> dict:
        """Serialize the recipe to a JSON-compatible dictionary. Round-trips via `from_dict`."""
        return self.model_dump(mode="json")

    def as_platform_recipe(self, recipe_name: Optional[str], overwrite: bool = False) -> Dict:
        """Upload the recipe to the Hafnia platform and return the platform's response.

        Equivalent to running `hafnia recipe create` from the CLI. Uses the active `Config` for
        credentials.

        Args:
            recipe_name: Display name for the recipe on the platform. If None, the platform may
                generate a name from the recipe contents.
            overwrite: If True, overwrite an existing recipe with the same name; otherwise the
                upload is rejected on conflict.
        """
        from hafnia.platform.dataset_recipe import get_or_create_dataset_recipe
        from hafnia_cli.config import Config

        cfg = Config()

        recipe = self.as_dict()
        recipe_dict = get_or_create_dataset_recipe(recipe=recipe, name=recipe_name, overwrite=overwrite, cfg=cfg)
        return recipe_dict

    ### Dataset Recipe Transformations ###
    def shuffle(recipe: DatasetRecipe, seed: int = 42) -> DatasetRecipe:
        """Append a deterministic shuffle operation to the recipe (recipe equivalent of `HafniaDataset.shuffle`)."""
        operation = recipe_transforms.Shuffle(seed=seed)
        recipe.append_operation(operation)
        return recipe

    def select_samples(
        recipe: DatasetRecipe,
        n_samples: int,
        shuffle: bool = True,
        seed: int = 42,
        with_replacement: bool = False,
    ) -> DatasetRecipe:
        """Append a "select N samples" operation to the recipe.

        Recipe equivalent of `HafniaDataset.select_samples`. See that method for the meaning of
        each argument.
        """
        operation = recipe_transforms.SelectSamples(
            n_samples=n_samples,
            shuffle=shuffle,
            seed=seed,
            with_replacement=with_replacement,
        )
        recipe.append_operation(operation)
        return recipe

    def splits_by_ratios(recipe: DatasetRecipe, split_ratios: Dict[str, float], seed: int = 42) -> DatasetRecipe:
        """Append a "split by ratios" operation that assigns split names to samples by the given ratios.

        Recipe equivalent of `HafniaDataset.splits_by_ratios`.
        """
        operation = recipe_transforms.SplitsByRatios(split_ratios=split_ratios, seed=seed)
        recipe.append_operation(operation)
        return recipe

    def split_into_multiple_splits(
        recipe: DatasetRecipe, split_name: str, split_ratios: Dict[str, float]
    ) -> DatasetRecipe:
        """Append an operation that subdivides one existing split into several smaller splits.

        Recipe equivalent of `HafniaDataset.split_into_multiple_splits` — useful when the source
        dataset only ships two splits and a third (e.g. `val`) needs to be carved out.
        """
        operation = recipe_transforms.SplitIntoMultipleSplits(split_name=split_name, split_ratios=split_ratios)
        recipe.append_operation(operation)
        return recipe

    def define_sample_set_by_size(recipe: DatasetRecipe, n_samples: int, seed: int = 42) -> DatasetRecipe:
        """Append an operation that tags `n_samples` random samples as the dataset's sample subset.

        Recipe equivalent of `HafniaDataset.define_sample_set_by_size`.
        """
        operation = recipe_transforms.DefineSampleSetBySize(n_samples=n_samples, seed=seed)
        recipe.append_operation(operation)
        return recipe

    def class_mapper(
        recipe: DatasetRecipe,
        class_mapping: Union[Dict[str, str], List[Tuple[str, str]]],
        method: str = "strict",
        primitive: Optional[Type[Primitive]] = None,
        task_name: Optional[str] = None,
    ) -> DatasetRecipe:
        """Append a class-renaming/grouping operation to the recipe.

        Recipe equivalent of `HafniaDataset.class_mapper`. See that method's docstring for the
        semantics of `method`, `primitive` and `task_name`.
        """
        operation = recipe_transforms.ClassMapper(
            class_mapping=class_mapping,
            method=method,
            primitive=primitive,
            task_name=task_name,
        )
        recipe.append_operation(operation)
        return recipe

    def rename_task(recipe: DatasetRecipe, old_task_name: str, new_task_name: str) -> DatasetRecipe:
        """Append a task-rename operation to the recipe (recipe equivalent of `HafniaDataset.rename_task`)."""
        operation = recipe_transforms.RenameTask(old_task_name=old_task_name, new_task_name=new_task_name)
        recipe.append_operation(operation)
        return recipe

    def select_samples_by_class_name(
        recipe: DatasetRecipe,
        name: Union[List[str], str],
        task_name: Optional[str] = None,
        primitive: Optional[Type[Primitive]] = None,
    ) -> DatasetRecipe:
        """Append an operation that keeps only samples containing at least one annotation with the named class(es).

        Recipe equivalent of `HafniaDataset.select_samples_by_class_name`.
        """
        operation = recipe_transforms.SelectSamplesByClassName(name=name, task_name=task_name, primitive=primitive)
        recipe.append_operation(operation)
        return recipe

    def drop_samples_by_class_name(
        recipe: DatasetRecipe,
        name: Union[List[str], str],
        task_name: Optional[str] = None,
        primitive: Optional[Type[Primitive]] = None,
        drop_classes_from_task_info: bool = True,
    ) -> DatasetRecipe:
        """Append an operation that drops samples containing the named class(es) and (optionally) removes the classes from `info`.

        Recipe equivalent of `HafniaDataset.drop_samples_by_class_name`.
        """
        operation = recipe_transforms.DropSamplesByClassName(
            name=name,
            task_name=task_name,
            primitive=primitive,
            drop_classes_from_task_info=drop_classes_from_task_info,
        )
        recipe.append_operation(operation)
        return recipe

    ### Helper methods ###
    def get_dataset_names(self) -> List[str]:
        """Return every Hafnia-platform dataset name referenced in the recipe (recursively).

        Walks the creation tree (including nested `from_merge` / `from_merger` recipes) and
        collects the names passed to `from_name`. Names from `from_path` and
        `from_name_public_dataset` are excluded — only platform-managed datasets are returned.
        Duplicates are not de-duplicated.
        """
        if self.creation is None:
            return []
        return self.creation.get_dataset_names()

    ### Validation and Serialization ###
    @field_validator("creation", mode="plain")
    @classmethod
    def validate_creation(cls, creation: Union[Dict, RecipeCreation]) -> RecipeCreation:
        if isinstance(creation, dict):
            creation = Serializable.from_dict(creation)  # type: ignore[assignment]
        if not isinstance(creation, RecipeCreation):
            raise TypeError(f"Operation must be an instance of RecipeCreation, got {type(creation).__name__}.")
        return creation

    @field_serializer("creation")
    def serialize_creation(self, creation: RecipeCreation) -> dict:
        return creation.model_dump()

    @field_validator("operations", mode="plain")
    @classmethod
    def validate_operation(cls, operations: List[Union[Dict, RecipeTransform]]) -> List[RecipeTransform]:
        if operations is None:
            return None
        validated_operations = []
        for operation in operations:
            if isinstance(operation, dict):
                operation = Serializable.from_dict(operation)  # type: ignore[assignment]
            if not isinstance(operation, RecipeTransform):
                raise TypeError(f"Operation must be an instance of RecipeTransform, got {type(operation).__name__}.")
            validated_operations.append(operation)
        return validated_operations

    @field_serializer("operations")
    def serialize_operations(self, operations: Optional[List[RecipeTransform]]) -> Optional[List[dict]]:
        """Serialize the operations to a list of dictionaries."""
        if operations is None:
            return None
        return [operation.model_dump() for operation in operations]


def unique_name_from_recipe(recipe: DatasetRecipe) -> str:
    if isinstance(recipe.creation, FromName) and recipe.operations is None:
        # If the dataset recipe is simply a DatasetFromName, we bypass the hashing logic
        # and return the name directly. The dataset is already uniquely identified by its name.
        # Add  version if need... Optionally, you may also completely delete this exception
        # and always return the unique name including the hash to support versioning.
        return recipe.creation.name  # Dataset name e.g 'mnist'
    recipe_json_str = recipe.model_dump_json()
    hash_recipe = utils.hash_from_string(recipe_json_str)
    short_recipe_str = recipe.as_short_name()
    unique_name = f"{short_recipe_str}_{hash_recipe}"
    return unique_name


class FromPath(RecipeCreation):
    path_folder: Path
    check_for_images: bool = True

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.from_path

    def as_short_name(self) -> str:
        return f"'{self.path_folder}'".replace(os.sep, "-")

    def get_dataset_names(self) -> List[str]:
        return []  # Only counts 'from_name' datasets


class FromName(RecipeCreation):
    name: str
    version: Optional[str] = None
    force_redownload: bool = False
    download_files: bool = True

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.from_name

    def as_short_name(self) -> str:
        return self.name

    def get_dataset_names(self) -> List[str]:
        return [self.name]


class FromNamePublicDataset(RecipeCreation):
    name: str
    force_redownload: bool = False
    n_samples: Optional[int] = None

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.from_name_public_dataset

    def as_short_name(self) -> str:
        return f"Torchvision('{self.name}')"

    def get_dataset_names(self) -> List[str]:
        return []


class FromMerge(RecipeCreation):
    recipe0: DatasetRecipe
    recipe1: DatasetRecipe

    @staticmethod
    def get_function():
        return HafniaDataset.merge

    def as_short_name(self) -> str:
        merger = FromMerger(recipes=[self.recipe0, self.recipe1])
        return merger.as_short_name()

    def get_dataset_names(self) -> List[str]:
        """Get the dataset names from the merged recipes."""
        names = [
            *self.recipe0.creation.get_dataset_names(),
            *self.recipe1.creation.get_dataset_names(),
        ]
        return names


class FromMerger(RecipeCreation):
    recipes: List[DatasetRecipe]

    def build(self, download_files: bool = True) -> HafniaDataset:
        """Build the dataset from the merged recipes."""
        datasets = [recipe.build(download_files=download_files) for recipe in self.recipes]
        return self.get_function()(datasets=datasets)

    @staticmethod
    def get_function():
        return HafniaDataset.from_merger

    def as_short_name(self) -> str:
        return f"Merger({','.join(recipe.as_short_name() for recipe in self.recipes)})"

    def get_dataset_names(self) -> List[str]:
        """Get the dataset names from the merged recipes."""
        names = []
        for recipe in self.recipes:
            names.extend(recipe.creation.get_dataset_names())
        return names


def get_dataset_path_from_recipe(recipe: DatasetRecipe, path_datasets: Optional[Union[Path, str]] = None) -> Path:
    path_datasets = path_datasets or utils.PATH_DATASETS
    path_datasets = Path(path_datasets)
    unique_dataset_name = unique_name_from_recipe(recipe)
    return path_datasets / unique_dataset_name


def get_or_create_dataset_path_from_recipe(
    dataset_recipe: DatasetRecipe,
    force_redownload: bool = False,
    path_datasets: Optional[Union[Path, str]] = None,
) -> Path:
    path_dataset = get_dataset_path_from_recipe(dataset_recipe, path_datasets=path_datasets)

    if force_redownload:
        shutil.rmtree(path_dataset, ignore_errors=True)

    dataset_metadata_files = DatasetMetadataFilePaths.from_path(path_dataset)
    if dataset_metadata_files.exists(raise_error=False):
        return path_dataset

    path_dataset.mkdir(parents=True, exist_ok=True)
    path_recipe_json = path_dataset / FILENAME_RECIPE_JSON
    path_recipe_json.write_text(dataset_recipe.model_dump_json(indent=4))

    dataset: HafniaDataset = dataset_recipe.build()
    dataset.write(path_dataset)

    return path_dataset
