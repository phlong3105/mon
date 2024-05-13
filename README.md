<div align="center">
	<h1 align="center">🐈 MON</h1>
</div>

<div align = center>
	<a align="center" href="http://phlong.net/mon/">Documentation</a>
	<br>
	<p></p>
</div>

- `🐈 mon` is an all-in-one research framework built using [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/). 
- It covers a wide range of research topics in computer vision and machine learning.
- The full documentation of the framework [can be found here](http://phlong.net/mon/) (still work-in-progress).

## Installation

```shell
git clone https://phlong3105@github.com/phlong3105/mon
cd mon
chmod +x install.sh
bash -i install.sh
```

## Directory Organization

- Look at "private.docx" for more information.

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
     |       |_ vision       # Computer vision package.
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
    year         = {2022},
}
```

## Contact
If you have any questions, feel free to contact `Long H. Pham` 
([longpham3105@gmail.com](longpham3105@gmail.com) or [phlong@skku.edu](phlong@skku.edu))
