## train uncertainty model
python train.py -uncertainty --config config/train/train-uncertainty.json --dataset ./config/data/lol-v1.yml

## train global structure-aware diffusion
python train.py --config config/tain/lol-v1-train.json --dataset ./config/data/lol-v1.yml
