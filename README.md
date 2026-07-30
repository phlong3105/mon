<div align="center">
	<h1 align="center">🐈 MON</h1>
</div>

`🐈 mon` is a research **[Monorepo](https://github.com/matakanobu/python-monorepo/tree/main)** for computer vision, built using [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/).

<details>
  <summary></summary>
  Yes, you are right. It's named after my cat 🐈 (Mon).
</details>

## Installation

```shell
git clone https://phlong3105@github.com/phlong3105/mon
cd mon/scripts
sudo chmod +x bootstrap.sh
./bootstrap.sh
```

The code is fully compatible with [PyTorch](https://pytorch.org/) >= 2.0.

## Repo Structure

`🐈 mon` is a **[Monorepo](https://github.com/matakanobu/python-monorepo/tree/main)** with multiple projects (``projects``) and shared common libs (``libs``).

<details>
  <summary>Directory Structure</summary>

  ```text
  mon
  ├── docs                      # Documentation.
  ├── libs                      # Common code shared across multiple projects is located.
  │   ├── mon                   # My lib package.
  │   │   ├── src/              # Adopt src/ layout.
  │   │   └── pyproject.toml
  │   └── ...                   # Other 3rd-party packages.
  ├── projects                  # Projects.
  │   ├── project_A             # Adopt src/ layout.
  │   ├── project_B             # Adopt src/ layout.
  │   └── ...
  ├── scripts                   # General-purpose scripts, CI/CD helpers, local environment setup, testing runners.
  │   ├── bootstrap.sh          # Bootstrap script for local development.
  │   └── ...
  ├── tools                     # Custom CLI tools, internal automation systems, scaffolding utilities.
  ├── zoo                       # Model zoo (i.e., pre-trained weights).
  ├── .gitignore
  ├── .gitmodules
  ├── LICENSE
  ├── pyproject.toml            # Root configuration file.
  └── README.md                 # Readme file.
  ```
</details>

## Cite
If you find our work useful, please consider citing the following:
```text
@misc{Pham2022,
    author       = {Long Hoang Pham, Duong Nguyen-Ngoc Tran, Quoc Pham-Nam Ho},
    title        = {🐈 mon},
    publisher    = {GitHub},
    journal      = {GitHub repository},
    howpublished = {https://github.com/phlong3105/mon},
    year         = {2022},
}
```

## Contact
If you have any questions, feel free to contact `Long H. Pham` ([longpham3105@gmail.com](longpham3105@gmail.com) or [phlong@skku.edu](phlong@skku.edu))
