# Dataset Recipes

A `DatasetRecipe` is a serializable **specification** of a dataset: which
sources to start from and which operations to apply on top of them. A recipe
is *not* a materialized dataset — call `.build()` to turn it into a
[`HafniaDataset`](dataset.md).

## Why recipes?

`HafniaDataset` is great for interactive exploration: every operation runs
immediately on the data. But that makes it less convenient when you want to:

- **Reproduce** the exact same dataset locally and inside Training-aaS.
- **Share** a dataset definition with teammates as a JSON file.
- **Combine** multiple datasets (with class remapping, filtering, splits)
  into one training source.
- **Re-execute** the same composition against the full dataset on the
  platform after validating it against the sample dataset locally.

Recipes capture all of that as a small, declarative object that travels
freely between environments.

## The recipe lifecycle

1. **Define** the recipe with the same chainable interface as `HafniaDataset`.
2. **Build** it locally to verify it produces the dataset you expected.
3. **Save** it as JSON, or **upload** it to the platform.
4. **Reuse** the recipe in a Training-aaS experiment — where it builds against
   the full dataset rather than the sample.

```python
from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe

recipe  = DatasetRecipe.from_name("mnist", version="1.0.0").shuffle().select_samples(n_samples=20)
dataset = recipe.build()   # materialize as a HafniaDataset
```

The recipe and the equivalent eager call

```python
HafniaDataset.from_name("mnist", version="1.0.0").shuffle().select_samples(n_samples=20)
```

produce the same dataset; the recipe just defers execution.

## Sources

Recipes can start from several kinds of sources:

```python
DatasetRecipe.from_name("mnist", version="1.0.0")            # named platform dataset
DatasetRecipe.from_name_public_dataset("mnist", n_samples=100) # bundled public dataset (no login)
DatasetRecipe.from_recipe_name("person-vehicle-detection")    # platform-registered recipe
```

## Operations

Recipes mirror the `HafniaDataset` transformation surface, so anything you
verified eagerly translates 1:1 into the recipe:

```python
recipe = (
    DatasetRecipe.from_name("coco-2017", version="1.0.0")
    .class_mapper(class_mapping={"car": "Vehicle", "person": "Person"},
                  method="remove_undefined", task_name="object_detection")
    .select_samples_by_class_name(name=["Person", "Vehicle"], task_name="object_detection")
    .shuffle(seed=42)
    .splits_by_ratios({"train": 0.8, "val": 0.1, "test": 0.1})
)
```

Supported operations include `shuffle`, `select_samples`,
`select_samples_by_class_name`, `drop_samples_by_class_name`, `class_mapper`,
`rename_task`, `splits_by_ratios`, and `split_into_multiple_splits`.

## Merging datasets

Multiple recipes can be combined into one — recipes can be nested arbitrarily.

```python
merged = DatasetRecipe.from_merger(
    recipes=[
        DatasetRecipe.from_name("mnist", version="1.0.0"),
        DatasetRecipe.from_name("mnist", version="1.0.0").select_samples(n_samples=20),
        DatasetRecipe.from_path(Path(".data/datasets/mnist"))
                     .splits_by_ratios({"train": 0.8, "val": 0.1, "test": 0.1}),
    ]
)
dataset = merged.build()
```

`DatasetRecipe.from_merge(recipe0, recipe1)` is the two-recipe shortcut.

## Saving, loading, inspecting

A recipe is a plain JSON payload:

```python
recipe.as_json_file(Path(".data/dataset_recipes/example.json"))
again = DatasetRecipe.from_json_file(Path(".data/dataset_recipes/example.json"))
assert again == recipe

print(recipe.as_json_str())     # JSON string
print(recipe.as_python_code())  # equivalent Python code
print(recipe.as_short_name())   # short identifier for logs/paths
```

## Uploading to Training-aaS

Once you are happy with a recipe locally, upload it to the platform so it
can be used in experiments:

```python
recipe.as_platform_recipe(recipe_name="example-mnist-recipe", overwrite=True)

# Later, anywhere in your org:
recipe = DatasetRecipe.from_recipe_name("example-mnist-recipe")
```

From the CLI, manage platform recipes with:

```bash
hafnia recipe ls
hafnia recipe create <path-to-recipe.json>
hafnia recipe rm    <recipe-name>
```

And launch an experiment using a recipe:

```bash
hafnia experiment create --recipe example-mnist-recipe --trainer-path ../trainer-classification
```

## Recommended workflow

1. Explore the involved datasets eagerly with `HafniaDataset` (use
   `print_basic_stats()`, `get_task_by_primitive(...).get_class_names()`, ...).
2. Validate transformations (class remapping, merging, filtering) on the
   sample datasets, still using `HafniaDataset`.
3. Translate the validated operations into a single `DatasetRecipe` and
   `.build()` it on the sample datasets to confirm parity.
4. Upload the recipe to the platform and reference it from an experiment to
   train on the full dataset.

## See also

- [docs/dataset.md](dataset.md) — the `HafniaDataset` format that recipes build into.
- [examples/example_dataset_recipe.py](../examples/example_dataset_recipe.py) — runnable walkthrough including a Person+Vehicle merge across COCO and Midwest.
