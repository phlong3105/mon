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

Here’s a breakdown of the important behaviour differences between the `src`
layout and the flat layout:
- The `src` layout requires installation of the project to be able to run its
  code, and the `non-src` layout doesn't.
- The `src` layout helps prevent accidental usage of the in-development copy of
  the code.
- The `src` layout helps enforce that an editable installation is only able to
  import files that were meant to be importable.

Rule:
- `non-src` or flat layout, any single module or single Python script. It means
  you just need to gather a bunch of third-party code together and write 
  running/main scripts. You can write executable notebooks.
- `src` layout, anything else.

## 3. Naming Things: Singular or Plural
Rule:
  - Use singular nouns, except for words that don't have singular forms (i.e.,
    news, weights, etc.).

Reasons:
- **Reason 1 (Convenience)**: It is easier come out with singular names, than
  with plural ones. Objects can have irregular plurals or not plural at all, but
  will always have a singular one (with few exceptions like News).
- **Reason 2 (Aesthetic and Order)**: Especially in master-detail scenarios,
  this reads better, aligns better by name, and has more logical order (Master
  first, Detail second):
  ```
  Good:
    1.Order
    2.OrderDetail 
   
  Compared to:
    1.OrderDetails
    2.Orders
  ```

See more: https://stackoverflow.com/questions/338156/table-naming-dilemma-singular-vs-plural-names

## 4. Naming Functions and Methods
Commonly used verbs for naming functions and methods:

- **Alteration:**

| Verb     | Definition                                                                                                               | Examples         |
|----------|--------------------------------------------------------------------------------------------------------------------------|------------------|
| set      | Often used to put data in an existing resource such as an attribute of an object.                                        | set_name()       |
| change   | Often used when a whole thing, such as image, is replaced by something else. 	                                           | change_image()   |
| edit     | Often used same as change. It could be used especially when action is responsible for rendering the view.                | edit_record()    |
| update   | Often used when one or more of the components is updated as a result, and something new could also be added.             | update_file()    |
| add      | Often used to add something into a group of the things.                                                                  | add_item()       |
| append   | Often used same as add. It could be used when it does not modify the original group of things but produce the new group. | append_item()    |
| remove   | Often used when a given thing is removed from a group of the things.                                                     | remove_item()    |
| delete   | Often used same as remove, but it could also render nonrecoverable.                                                      | delete_item()    |
| write    | Often used when preserving data to an external source. Use together with `read`.                                         | save_json()      |
| store    | Often used the same way as save.                                                                                         | store_json()     |
| disable  | Often used to configure a resource an unavailable or inactive state.                                                     | disable_user()   |
| hide     | Often has the same intention as disable, by hiding it.                                                                   | hide_field()     |
| split    | Used when separating parts of a resource.                                                                                | split_table()    |
| separate | Often used the same way as the verb split.                                                                               | separate_table() |
| merge    | Often used when creating a single resource from multiple resource.                                                       | merge_records()  |
| join     | It can be used in a same way as merge.                                                                                   | join_records()   |

- **Conversion:**

| Verb         | Definition                                                                          | Examples             |
|--------------|-------------------------------------------------------------------------------------|----------------------|
| to           | Used when converting a variable from any types to a desired type.                   | to_list()            |
| a_to_b       | Used when converting a variable from type `a` to type `b`.                          | string_to_int()      |
| `Class`.from | Used when creating an instance of a `Class` from a value.                           | List.from_string()   |

- **Creation:**

| Verb     | Definition                                                                          | Examples             |
|----------|-------------------------------------------------------------------------------------|----------------------|
| create   | Used when creating a resource.                                                      | create_directory()   |
| make     | Often used in a same way as create.                                                 | make_package()       |
| generate | Often used in a same way as create.                                                 | generate_directory() |
| copy     | Used when creating a resource with the same structure and data as the original one. | copy_file()          |

- **Establishment:**

| Verb  | Definition                                                              | Examples          |
|-------|-------------------------------------------------------------------------|-------------------|
| start | Generally used when initiating an operation.                            | start_listening() |
| begin | Often used in a same way as start.                                      | begin_listening() |
| open  | Used when changing state of a resource to make it accessible or usable. | open_file()       |

- **Obtainment:**

| Verb   | Definition                                                                                              | Examples         |
|--------|---------------------------------------------------------------------------------------------------------|------------------|
| get    | Generally used to obtain a resource.                                                                    | get_data()       |
| fetch  | Can be used in a same way as get.                                                                       | fetch_data()     |
| read   | Used when acquiring data from a source. Use together with `write`.                                      | read_file()      |
| search | Generally used in a same way as find. It may refer to look for an unknown data from multiple containers | search_element() |
| close  | Used when changing state of a resource to make it inaccessible or unusable.                             | close_file()     |

- **True or False Statement:**

| Verb   | Definition                                                 | Examples        |
|--------|------------------------------------------------------------|-----------------|
| is     | Used when defining state of a resource.                    | is_available()  |
| has    | Used to define whether a resource contains a certain data. | has_name()      |
| can    | Used to define a certain ability of a resource.            | can_load()      |
| should | Used to define a certain obligation of a resource.         | should_render() |


- **Using noun for function name:**
  - A function is always expected to perform an action. **If it barely returns a
  value, it should be a property**. You have a hint that the function should be
  transformed into a property when:
    - The function barely contains a `return ...` statement,
    - The function's name, which comes naturally into your mind is `get_something`,
      as in `product.get_price()` --> `product.price()`.
