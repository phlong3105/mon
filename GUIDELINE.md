# Developer Guidelines

Python is the main dynamic language used in this project. This style guide is a
list of dos and don’ts for Python programs.

Most of the guidelines here are adopted from [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

## 1. Python Language

## 1.1 Import
Use import statements for packages and modules only, not for individual classes
or functions.

### 1.1.1 Definition
Reusability mechanism for sharing code from one module to another.

### 1.1.2 Pros
The namespace management convention is simple. The source of each identifier is
indicated in a consistent way; `x.Obj` says that object `Obj` is defined in
module `x`.

### 1.1.3 Cons
Module names can still collide. Some module names are inconveniently long.

### 1.1.4 Decision
- Use `import x` for importing packages and modules.
- Use `from x import y` where `x` is the package prefix and y is the module name
  with no prefix.
- Use `from x import y as z` if two modules named `y` are to be imported, if `y`
conflicts with a top-level name defined in the current module, or if `y` is an
inconveniently long name.
- Use `import y as z` only when `z` is a standard abbreviation (e.g., np for
  numpy).

For example the module `sound.effects.echo` may be imported as follows:
```python
from sound.effects import echo
...
echo.EchoFilter(input, output, delay=0.7, atten=4)
```

Do not use relative names in imports. Even if the module is in the same package,
use the full package name. This helps prevent unintentionally importing a
package twice.

### 1.1.5 Exemption
Exemptions from this rule:

Symbols from the following modules are used to support static analysis and type
checking:

- typing module
- collections.abc module
- typing_extensions module


## 1.2 Packages
Import each module using the full pathname location of the module.

### 1.2.1 Pros
Avoids conflicts in module names or incorrect imports due to the module search
path not being what the author expected. Makes it easier to find modules.

### 1.2.2 Cons
Makes it harder to deploy code because you have to replicate the package
hierarchy. Not really a problem with modern deployment mechanisms.

### 1.2.3 Decision
All new code should import each module by its full package name.

Imports should be as follows:

Yes:
```python
  # Reference absl.flags in code with the complete name (verbose).
  import absl.flags
  from doctor.who import jodie

  _FOO = absl.flags.DEFINE_string(...)
```

Yes:
```python
  # Reference flags in code with just the module name (common).
  from absl import flags
  from doctor.who import jodie

  _FOO = flags.DEFINE_string(...)
```
(assume this file lives in doctor/who/ where jodie.py also exists)

No:
```python
  # Unclear what module the author wanted and what will be imported.  The actual
  # import behavior depends on external factors controlling sys.path.
  # Which possible jodie module did the author intend to import?
  import jodie
```

The directory the main binary is located in should not be assumed to be in
sys.path despite that happening in some environments. This being the case, code
should assume that import jodie refers to a third party or top level package
named jodie, not a local jodie.py.
