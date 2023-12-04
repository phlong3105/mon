# API Explanation

This section explains `🐈 mon`'s API.

---

## Framework Structure

`🐈 mon` is designed to keep everything within the framework, including data,
libraries, scripts, and more. It works optimally with an IDE like 
__[PyCharm](https://www.jetbrains.com/)__, but can also be easily used in any 
console environment.

```text
mon
 |_ bin                  # Executable, main() files, CLI.
 |_ data                 # Default location to store working datasets.
 |_ docs                 # Documentation.
 |_ run                  # Running results for train/val/test/predict.
 |_ src                  # Source code.
 |   |_ app              # Code for specific applications/projects.
 |   |_ cmon             # C++ code.
 |   |_ lib              # Third-party libraries. Place code from other sources here.
 |   |_ mon              # Python code.
 |       |_ core         # Base functionality for other packages.
 |       |_ nn           # Machine learning package.
 |       |_ vision       # Computer vision package.
 |_ test                 # Testing code.
 |_ zoo                  # Model zoo.
 |_ dockerfile           # Docker setup.
 |_ linux.yaml           # ``conda`` setup for Linux.
 |_ linux-test.yaml      # ``conda`` testing setup for Linux.
 |_ mac.yaml             # ``conda`` setup for macOS.
 |_ server.yaml          # ``conda`` setup for deep learning workstation.
 |_ pycharm.env          # Local environment variables.
 |_ .gitignore           # 
 |_ README.md            # Github Readme
 |_ install.sh           # Installation script
 |_ pyproject.toml  
 |_ LICENSE  
 |_ mkdocs.yaml          # mkdocs setup
```

---

## Core API

`🐈 mon`'s core API includes several packages built on top of 
__[Python](https://www.python.org/)__ and __[PyTorch](https://pytorch.org/)__.

| Package                           | Description                           |
|-----------------------------------|---------------------------------------|
| __[core](mon/core/index.md)__     | Base functionality for other packages |
| __[coreml](mon/coreml/index.md)__ | Machine learning code                 |
| __[vision](mon/vision/index.md)__ | Computer vision code                  |

---

## Optional API

- `🐈 mon`'s functionality can be extended by incorporating third-party code to
__[lib](explanation/lib/index.md)__ package. 
- Also, projects built on top of __[mon](explanation/mon/index.md)__, can be
  placed inside __[app](explanation/app/index.md)__ package for better
  management.

| Package                   | Description         |
|---------------------------|---------------------|
| __[app](app/index.md)__   | Application code    |
| __[lib](lib/index.md)__   | Third-party library |
