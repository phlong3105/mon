
## Getting Started
### Prerequisite

|            | Requirement                                                                                                                                                                                                                                          |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **OS**     | [**Ubuntu 20.04 / 22.04**](https://ubuntu.com/download/desktop) (fully supports), `Windows 10` and `MacOS` (partially supports)                                                                                                                      |
| **Env**    | [**Python>=3.9.0**](https://www.python.org/), [**PyTorch>=1.11.0**](https://pytorch.org/get-started/locally/), [**cudatoolkit=11.3**](https://pytorch.org/get-started/locally/), with [**anaconda**](https://www.anaconda.com/products/distribution) |
| **Editor** | [**PyCharm**](https://www.jetbrains.com/pycharm/download)                                                                                                                                                                                            |

### Directory
```text
zero_adce            # Root directory
 |__ data            # Contains data
 |__ runs            # Run dir
 |__ weights         # Pretrained models weights
```

### Installation
```shell
cd <to-where-you-want-to-save>
mkdir -p zero_adce
mkdir -p zero_adce/data
cd zero_adce

git clone git@github.com:phlong3105/one
cd one/install
chmod +x install.sh
conda init bash
bash -i install.sh
cd ../..

conda activate one
```

### Prepare Data
