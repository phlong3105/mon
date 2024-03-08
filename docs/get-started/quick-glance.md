# Quick Glance

---

## Framework Structure

`🐈 mon` is designed to keep everything within the framework, including data,
libraries, scripts, and more. It works optimally with an IDE like 
[PyCharm](https://www.jetbrains.com/), but can also be easily used in any 
console environment.

```text
code
 |_ mon
     |_ bin                  # Executable, main() files, CLI.
     |_ data                 # Default location to store working datasets.
     |_ docs                 # Documentation.
     |_ env                  # Environment variables.
     |_ src                  # Source code.
     |   |_ mon              # Python code.
     |       |_ config       # Configuration functionality.
     |       |_ core         # Base functionality for other packages.
     |       |_ data         # Data processing package.
     |       |_ nn           # Machine learning package.
     |       |_ proc         # Basic processing package. 
     |       |_ vision       # Computer vision package.
     |_ zoo                  # Model zoo.
     |_ .gitignore           # 
     |_ install.sh           # Installation script.
     |_ LICENSE              #
     |_ mkdocs.yaml          # mkdocs setup.
     |_ pyproject.toml       # 
     |_ README.md            # Github Readme.
```
