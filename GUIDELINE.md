# Developer Guidelines

Python is the main dynamic language used in this project. This style guide is a
list of dos and don’ts for Python programs.

Most of the guidelines here are adopted from [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

## 1. Import
**Use import statements for packages and modules only, not for individual
classes or functions.**

Rule:
- Use `import x` for importing packages and modules.
- Use `from x import y` where `x` is the package prefix and y is the module name
  with no prefix.
- Use `from x import y as z` if two modules named `y` are to be imported, if `y`
conflicts with a top-level name defined in the current module, or if `y` is an
inconveniently long name.
- Use `import y as z` only when `z` is a standard abbreviation (e.g., np for
  numpy).

For example, the module `sound.effects.echo` may be imported as follows:
```python
from sound.effects import echo
...
echo.EchoFilter(input, output, delay=0.7, atten=4)
```

Do not use relative names in imports. Even if the module is in the same package,
use the full package name. This helps prevent unintentionally importing a
package twice.

Exemption:
- typing module
- collections.abc module
- typing_extensions module

## 2. Packaging Layout
There are two layouts for structuring a Python project: `src` or `non-src`.

For example, the `non-src` layout is:
```
sample
├── docs/
├── sample/
│   ├── __init__.py
│   └── module.py
├── tools/
│   └── generate_awesomeness.py
├── test/
├── README.md
├── pyproject.toml
```

The `src` layout is:
```
sample
├── docs/
├── src/
│   └── sample/
│       ├── __init__.py
│       └── module.py
├── tools/
│   └── generate_awesomeness.py
├── test/
├── README.md
├── pyproject.toml
```

Here’s a breakdown of the important behaviour differences between the src layout
and the flat layout:
- The `src` layout requires installation of the project to be able to run its
  code, and the `non-src` layout doesn't.

Rule:
- `non-src` or flat layout, any single module or single Python script. It means
  you just need to gather a bunch of third-party code together and write 
  running/main scripts. You can also write executable notebooks.
- `src` layout, anything else.

## 3. Naming Things: Singular or Plural
Rule:
  - Use singular nouns, except for words that don't have singular forms (i.e.,
    news, weights, etc.).

Reasons:
- **Reason 1 (Convenience)**: It is easier come out with singular names, than with
plural ones. Objects can have irregular plurals or not plural at all, but will
always have a singular one (with few exceptions like News). 
- **Reason 2 (Aesthetic and Order)**: Especially in master-detail scenarios, this
reads better, aligns better by name, and has more logical order (Master first,
Detail second):
  ```
  Good:
    1.Order
    2.OrderDetail 
   
  Compared to:
    1.OrderDetails
    2.Orders
  ```

See more: https://stackoverflow.com/questions/338156/table-naming-dilemma-singular-vs-plural-names
