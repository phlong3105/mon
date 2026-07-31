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
