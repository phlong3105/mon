# `mon` Core Guidelines

---

## Introduction 

Python is the main dynamic language used in this project. This style guide is a list of dos and don’ts for Python programs.

Most of the guidelines here are adopted from [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

---

## 0. The Zen of Python

```text
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than right now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```

---

## 1. Import

Rules:

- **Use import statements for packages and modules only, not for individual classes or functions.**
- Use `import x` for importing packages and modules.
- Use `from x import y` where `x` is the package prefix and y is the module name with no prefix.

  ```python
  from sound.effects import echo
  
  echo.EchoFilter(input, output, delay=0.7, atten=4)
  ```

- Use `from x import y as z` if two modules named `y` are to be imported, if `y` conflicts with a top-level name defined in the current module, or if `y` is an inconveniently long name.
- Use `import y as z` only when `z` is a standard abbreviation (e.g., `np` for `numpy`).
- Do not use relative names in imports. Even if the module is in the same package, use the full package name. This helps prevent unintentionally importing a package twice.

Exemption:

- `abc` module
- `typing` module, including both Python's built-in and custom typing modules.
- `typing_extensions` module
- `globals` module. This module contains all user-defined globals like constants, enum, type alias

---

## 2. Packaging Layout

Rules:

- Use the `src` layout.
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

Advantages:

- The `src` layout requires installation of the project to be able to run its code.
- The `src` layout helps prevent accidental usage of the in-development copy of the code.
- The `src` layout helps enforce that an editable installation is only able to import files that were meant to be importable.

---

## 3. Naming

### 3.1 Naming Convention

`module_name`, `package_name`, `ClassName`, `method_name`, `ExceptionName`, `function_name`, `GLOBAL_CONSTANT_NAME`, `global_var_name`, `instance_var_name`, `function_parameter_name`, `local_var_name`, `query_proper_noun_for_thing`, `send_acronym_via_https`.

### 3.2 Naming: Singular or Plural

Rules:

- For `module_name`, `package_name`, `ClassName` uses **singular nouns**, except for words that don't have singular forms (i.e., news, weights, etc.).
- For `GLOBAL_CONSTANT_NAME`, `global_var_name`, `instance_var_name`, `function_parameter_name`, `local_var_name` use proper plural nouns.

Reasons:

- **Reason 1 (Convenience)**: It is easier come out with singular names. Objects can have irregular plurals or not plural at all, but will always have a singular one (with few exceptions like news).
- **Reason 2 (Order)**: Especially in master-detail scenarios, this reads better, aligns better by name, and has more logical order (Master first, Detail second):
  ```
  Good:
    1. Order
    2. OrderDetail 
   
  Bad:
    1. OrderDetails
    2. Orders
    3. OrdersDetails
  ```

See more: [Table Naming Dilemma: Singular vs. Plural Names](https://stackoverflow.com/questions/338156/table-naming-dilemma-singular-vs-plural-names)

### 3.3 Naming: Abbreviation

Rules:

- **FOLLOW** the naming convention of the third-party library that you are trying to extend or inherit.
  - Use `input` and `output` for PyTorch forward methods. Other cases, use `in` and `out`.
  - Use `img` in OpenCV functions.
  - Use `x` for generic no-name argument. Mostly in built-in or math functions.
  - Use `y` for the second generic no-name argument. Mostly built-in or math functions.
  - Use `src` and `dst` for generic functions: `copy(src, dst)`. Mostly in file I/O functions.
  
- Avoid Python reserved keywords. See more: [Built-in Functions](https://docs.python.org/3/library/functions.html)
- Use abbreviation only if:
	- `global` for understood by everyone
	- `significant` for seen for the first time you still know what it means
	- `absolute` for not related to the context
	- `short` for taking out one letter isn't an abbreviation

### 3.4 Naming: Functions & Methods

Rules:

- **Obtainment:**
	- Use `get` to obtain a resource.
	- Use `read` when acquiring data from a source. Use together with `write`.
	- Use `search` to look for an unknown data from multiple containers.
	- Use `close` when changing state of a resource to make it inaccessible or unusable. Use together with `open`.

- **Creation:**
	- Use `create` when creating a resource. Ex: `create_dir()`.
	- Use `X.from()` when creating an instance of class `X` from a value. Ex: `List.from_string()`.
	- Use `copy` when creating a resource with the same structure and data as the original one.

- **Alteration:**
	- Use `set` to put data in an existing resource such as an attribute of an object.
	- Use `change` when a whole thing, such as image, is replaced by something else.
	- Use `edit` similar as `change`. It could be used, especially when action
	  is responsible for rendering the view.
	- Use `update` when one or more of the components is updated as a result, and something new could also be added.
	- Use `add` to add something into a group of the things.
	- Use `append` similar as `add`. It could be used when it doesn't modify the original group of things, but produce the new group.
	- Use `remove` when a given thing is removed from a group of the things.
	- Use `delete` to eliminate the object or group of things.
	- Use `write` when preserving data to an external source. Use together with `read`.
	- Use `disable` to configure a resource an unavailable or inactive state.
	- Use `split` when separating parts of a resource.
	- Use `merge` when creating a single resource from multiple resources.
	- Use `join` similar as `merge` but for data and values.

- **Conversion:**
	- Use `to` when converting a variable from arbitrary types to the desired type. Ex: `to_list()`.
	- Use `x_to_y` when converting a variable from type `a` to type `b`. Ex: `str_to_int()`.

- **Establishment:**
	- Use `start` when initiating an operation. Ex: `start_listening()`.
	- Use `stop` when ending an operation. Ex: `stop_listening()`.
	- Use `open` when changing state of a resource to make it accessible or usable
	  Use together with `close`. Ex: `open_file()`.

- **True or False Statement:**
	- Use `is` when defining state of a resource. Ex: `is_available()`.
	- Use `has` to define whether a resource contains a certain data. Ex: `has_name()`.
	- Use `can` to define a certain ability of a resource.
	- Use `should` to define a certain obligation of a resource.

- **Using noun for function name:**
	- function is always expected to perform an action. **If it barely returns a value, it should be a property**.
	- You have a hint that the function should be transformed into a property
	  when:
		- The function barely contains a `return ...` statement,
		- The function's name, which comes naturally into your mind is `get_something`, as in `product.get_price()` --> `product.price()`.

### 3.5 Naming: Package

Rules:

- Package names should be all lower cases.
- It is usually preferable to stick to one word name. When multiple words are needed, an underscore should separate them.
- Use a noun as the name of the package because a package contains related modules and represents a thing or concept, which can be easily represented by a noun.
- Use a verb as the name of the package if the package provides a specific action or operation. For example, the package "pytest" is a testing framework for Python and uses the verb "test" in its name to clearly show its purpose.
- When stuck with coming up with a name for a package. Follow Apple's framework naming. See more: [Apple Documentation](https://developer.apple.com/documentation/technologies).

### 3.6 Naming: Module

Rules:

- Module names should be all lower cases.
- It is usually preferable to stick to one word name. When multiple words are needed, an underscore should separate them.

---

## 4. List and Tuple

Rules:

- Use `list` because it is the standard data type used in many libraries. When in doubt, use `list`.
- `list` supports many useful functions such as `sort()`, `reverse()`, etc.

---

## 5. Type Hint

Rules:

- DO NOT overuse it when coding. Think about it while refactoring your code.
- Use the suffix `Like` for union type that representing objects that can be coerced into a certain type. Example: `DictLike`.

---

## 6. `assert`, `raise`, `except`

Rules:

- Use `raise` to raise an exception.
	- Use `raise` to deal with value checking.
- Use `assert` to raise an exception if a given condition is meet.
	- Use `assert` for runtime debugging than normal runtime error detection.
	- Place assertions at the beginning of functions to check if the input is valid (preconditions).
	- Place assertions before functions’ return values to check if the output is valid (postconditions).
	- The best scenario is not using `assert`. If you name your function or method good enough, you don't need to assert.
- Use `try` to execute some code that might raise an exception, and if so, catch it.
	- Use `try` for files and database I/O operations. 

---

## 7. `protected` and `private`

Rules:

- Use a single underscore `_` prefix for protected methods and attributes.
- Use a double underscore `__` prefix for private methods and attributes.

**Recommendations:** 

- Since Python doesn't have real protected or private methods, it is better to use **public** **methods** and **attributes**. 
- If you require some custom attribute assignment with check/assert/transform, you can use single underscore `_` prefix attribute with `@property` and `@setter`.
- Our motto is keep thing simple and easy to fix. We prefer having debugging/running/testing errors quickly and fix them rather than trying to think of a way to avoid them.
