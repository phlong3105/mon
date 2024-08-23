<div align="center">
	<h1 align="center">🐈 MON</h1>
</div>

<div align = center>
	<a align="center" href="https://github.com/phlong3105/mon/docs">Documentation</a>
	<br>
	<p></p>
</div>

- `🐈 mon` is an all-in-one research framework built using [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/). 
- It covers a wide range of research topics in computer vision and machine learning.
- The development guidelines of the framework [can be found here](styleguide.md) (still work-in-progress).

## Installation

```shell
git clone https://phlong3105@github.com/phlong3105/mon
cd mon
chmod +x install.sh

# On Linux
conda init bash
bash -i install.sh

# On Mac
conda init zsh
zsh -i install.sh
```

The code is fully compatible with [PyTorch](https://pytorch.org/) >= 2.0.

## Directory Organization

```text
code
 |_ mon
     |_ data                 # Default location to store working datasets.
     |_ docs                 # Documentation.
     |_ env                  # Environment variables.
     |_ project              # Project-specific code.
     |_ src                  # Source code.
     |   |_ mon              # Python code.
     |       |_ config       # Configuration functionality.
     |       |_ core         # Base functionality for other packages.
     |       |_ data         # Data processing package.
     |       |_ nn           # Machine learning package.
     |       |_ vision       # Computer vision package.
     |_ tools                # Tools.
     |_ zoo                  # Model zoo.
     |_ .gitignore           # 
     |_ install.sh           # Installation script.
     |_ LICENSE              #
     |_ mkdocs.yaml          # mkdocs setup.
     |_ pyproject.toml       # 
     |_ README.md            # Github Readme.
```

## Cite
If you find our work useful, please cite the following:
```text
@misc{Pham2022,  
    author       = {Long Hoang Pham, Duong Nguyen-Ngoc Tran, Quoc Pham-Nam Ho},  
    title        = {🐈 mon},  
    publisher    = {GitHub},
    journal      = {GitHub repository},
    howpublished = {https://github.com/phlong3105/mon},
    year         = {2024},
}
```

## Contact
If you have any questions, feel free to contact `Long H. Pham` ([longpham3105@gmail.com](longpham3105@gmail.com) or [phlong@skku.edu](phlong@skku.edu))


<script type="text/javascript" id="clustrmaps" src="//clustrmaps.com/map_v2.js?d=mDDi2z1vAnHUyVPYInDSCoHgluvZPEfpCcbRFeggx3o&cl=ffffff&w=a"></script>
