![One](data/one.png) 

---

# One Research Framework
`One` is a comprehensive research repository for code, framework, and knowledge base of my work related to computer vision, machine learning, and deep learning.

[Getting Started](#getting-started) &nbsp;
[Knowledge Base](#knowledge-base) &nbsp;
[Cite](#cite) &nbsp;
[Contact](#contact) &nbsp;

---

## Getting Started
### Prerequisite

|            | Requirement                                                                                                                                                                                                                                          |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **OS**     | [**Ubuntu 20.04 / 22.04**](https://ubuntu.com/download/desktop) (fully supports), `Windows 10` and `MacOS` (partially supports)                                                                                                                      |
| **Env**    | [**Python>=3.9.0**](https://www.python.org/), [**PyTorch>=1.11.0**](https://pytorch.org/get-started/locally/), [**cudatoolkit=11.3**](https://pytorch.org/get-started/locally/), with [**anaconda**](https://www.anaconda.com/products/distribution) |
| **Editor** | [**PyCharm**](https://www.jetbrains.com/pycharm/download)                                                                                                                                                                                            |

### Directory
```text
one                  # Root directory
 |__ data            # Contains data
 |__ docs
 |__ install         # Helpful installation scripts       
 |__ pretrained      # Pretrained models weights
 |__ projects        # 
 |     |__ project1
 |     |__ project2
 |__ src
 |     |__ one       # Main source code
 |__ tests			 # Testing scripts
 |__ third_party     # Third-party libraries
```

### Installation using `conda`
```shell
cd <to-where-you-want-to-save-one-dir>
mkdir -p one
mkdir -p one/data
cd one

# Install `aic22_track4` package
git clone git@github.com:phlong3105/one
cd one/install
chmod +x install.sh
conda init bash

# Install package. When prompt to input the dataset directory path, you should 
# enter: <some-path>/one/datasets
bash -i install.sh
cd ..
pip install --upgrade -e .
```

## Cite
If you find our work useful, please cite the following:
```text
@misc{Pham2022,  
    author       = {Long Hoang Pham},  
    title        = {One: One Research Framework},  
    publisher    = {GitHub},
    journal      = {GitHub repository},
    howpublished = {\url{https://github.com/phlong3105/one}},
    year         = {2022},
}
```

## Contact
If you have any questions, feel free to contact `Long Pham` 
([longpham3105@gmail.com](longpham3105@gmail.com) or [phlong@skku.edu](phlong@skku.edu))
