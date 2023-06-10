# API Explanation

This section explains `🐈 mon`'s API.

---

## Framework Structure

🐈 mon is designed to keep everything within the framework, including data,
libraries, scripts, and more. It works optimally with an IDE like 
[PyCharm](https://www.jetbrains.com/), but can also be easily used in any 
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
 |_ dockerfile       
 |_ pycharm.env  
 |_ CHANGELOG.md  
 |_ GUIDELINE.md  
 |_ README.md  
 |_ install.sh  
 |_ pyproject.toml  
 |_ LICENSE  
 |_ linux.yaml  
 |_ mac.yaml  
 |_ mkdocs.yaml  
```

---

## Core API

<div class="grid cards" markdown>

- :fontawesome-brands-html5: __HTML__ for content and structure
- :fontawesome-brands-js: __JavaScript__ for interactivity
- :fontawesome-brands-css3: __CSS__ for text running out of boxes
- :fontawesome-brands-internet-explorer: __Internet Explorer__ ... huh?

</div>

<div class="grid cards" markdown>

-   __foundation__

    ---

    Base functionality for other packages

    [:octicons-arrow-right-24: Go](explanation/foundation.md)

-   __coreml__

    ---

    Core functionality to build machine learning models

    [:octicons-arrow-right-24: Go](explanation/coreml.md)

-   __vision__

    ---

    Computer vision research

    [:octicons-arrow-right-24: Go](explanation/vision.md)

</div>
