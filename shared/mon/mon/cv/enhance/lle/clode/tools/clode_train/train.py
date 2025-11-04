import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import os
import sys
from tqdm import tqdm
from omegaconf import OmegaConf

from losses import *
from misc import *
from fsp.trainer import Trainer
from fsp.data import LOLDataset
from network.conv_node import NODE

def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_torch(42)
cudnn.benchmark = True

os.environ["CUDA_VISIBLE_DEVICES"] = '5'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load configuration
config = OmegaConf.load(sys.argv[1])

model = NODE(device, (3, 128, 128), 32, augment_dim=0, time_dependent=True, adjoint=True)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
trainer = Trainer(model, optimizer, device, config)

dataset = LOLDataset()
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)

for epoch in tqdm(range(config.num_epochs)):
    trainer.train(dataloader)
    trainer.validation()