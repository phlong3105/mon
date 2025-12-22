# Useful Commands

---

## Setup

- Install `mon`: 
    ```commandline
    sudo chmod +x install.sh
    ./install.sh
    ```

- Install `ssh`:
    ```commandline
    sudo apt update
    sudo apt install openssh-server
    sudo systemctl status ssh
    ```

---

## GitHub

- Add sub-module: `git submodule add <repository_url> <path>`
    ```commandline
    git submodule add https://github.com/phlong3105/
    ```

- Remove sub-module: ` git submodule deinit -f <path/to/submodule>`
    ```commandline
    git submodule deinit -f 
    ```

## Poetry

- Install: `poetry install`
  ```commandline
  poetry install --extras "docs gui"
  ```

---

## Useful Prompts

- Generate docstring:
    ```text
    Act as a Senior Python Architect. Revise or generate docstrings for the provided code following these strict structural and grammatical rules:
  
    1. FORMATTING:
    - Use Google-style docstrings, imperative-style.
    - Wrap all text to a maximum width of 80 characters.
    - The first line must be a single, short summary (<79 chars) ending with a period.
  
    2. GRAMMATICAL HIERARCHY:
    - Package (__init__.py): First line MUST be a descriptive Noun Phrase. Follow with ONE sentence of what the package contains, starting with "This package contains...". Nothing else.
    - Module (.py file): First line MUST be a descriptive Noun Phrase. Follow with ONE sentence of what module's provides, starting with "This module provides...". Nothing else.
    - Class: First line MUST be a Noun Phrase (e.g., "A container for...", "An implementation of..."). The rest MUST use imperative-style.
    - Method/Function: MUST use with an Imperative Verb (e.g., "Calculate...", "Resize...", "Return...").
    
    3. ARGUMENTS & ATTRIBUTES:
    - Inside Classes: 
        - "Attributes:" Every attribute MUST include inline type hints (e.g., `data (np.ndarray): Description.`).
        - Omit the "Attributes:" section if there are no attributes.
        - Omit any "Args:" or "Returns:" sections.
    - Inside Functions/Methods:
        - "Args:" and "Returns:" sections MUST NOT contain inline type hints (rely on the code signature).
        - Omit the "Returns:" section entirely if the function returns None.
        - Include a "Raises:" section for any explicitly raised exceptions.
  
    Please process the following code:
    ```
