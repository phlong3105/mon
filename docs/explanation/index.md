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
 |_ bin                  # Executable, main() files, CLI
 |_ data                 # Default location to store working datasets
 |_ docs                 # Documentation
 |_ run                  # Running results: train/val/test/predict
 |_ src                  # Source code 
 |   |_ app              # Code for specific applications/projects
 |   |_ lib              # Third-party libraries
 |   |_ mon              # Main source root 
 |       |_ coreml       # Machine learning package
 |       |_ createml     # Machine learning training package
 |       |_ foundation   # Base functionality for other packages
 |       |_ vision       # Computer vision package
 |_ test                 # Testing code
 |_ zoo                  # Model zoo
 |_ dockerfile           # Docker setup
 |_ pycharm.env          # Local environment variables
 |_ CHANGELOG.md         # Changelogs
 |_ GUIDELINE.md  
 |_ README.md            # Github Readme
 |_ install.sh           # Installation script
 |_ pyproject.toml  
 |_ LICENSE  
 |_ linux.yaml           # conda setup for Linux
 |_ mac.yaml             # conda setup for macOS
 |_ mkdocs.yaml          # mkdocs setup
```

---

## Core API

`🐈 mon`'s core API includes several packages built on top of 
__[Python](https://www.python.org/)__ and __[PyTorch](https://pytorch.org/)__.

| Package                                   | Description                           |
|-------------------------------------------|---------------------------------------|
| __[foundation](mon/foundation/index.md)__ | Base functionality for other packages |
| __[coreml](mon/coreml/index.md)__         | Machine learning code                 |
| __[createml](mon/createml/index.md)__     | Training code                         |
| __[vision](mon/vision/index.md)__         | Computer vision code                  |

---

## Optional API

`🐈 mon`'s functionality can be extended by incorporating third-party code to
__[lib](explanation/lib/index.md)__ package. In addition, projects, which are 
built on top of __[mon](explanation/mon/index.md)__, can be placed inside 
__[app](explanation/app/index.md)__ package for better management.

| Package                   | Description         |
|---------------------------|---------------------|
| __[app](app/index.md)__   | Application code    |
| __[lib](lib/index.md)__   | Third-party library |
