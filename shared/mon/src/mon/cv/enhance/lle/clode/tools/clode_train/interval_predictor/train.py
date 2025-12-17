import os
from tqdm import tqdm
from pathlib import Path
import wandb
import yaml
import argparse

import torch
import torch.optim as optim
import numpy as np
import random
import sys

sys.path.append(str(Path(__file__).parent.parent))

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
np.random.seed(random_seed)
random.seed(random_seed)

from data import *
from model import *


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, required=True, help='run name in wandb')
parser.add_argument('--data', type=str, default='LOL', help='dataset name')
parser.add_argument('--model', type=str, default='v1', help='model name')
parser.add_argument('--hidden_dim', type=int, default=32, help='model hidden dimension')
parser.add_argument('--cuda', type=str, default='1', help='CUDA device id')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

models = {
    'v1': Regressor_v1,
    'v2': Regressor_v2,
    'v3': Regressor_v3,
    'v4': Regressor_v4,
    'v5': Regressor_v5,
    'v6': Regressor_v6,
    'v7': Regressor_v7,
    'v8': Regressor_v8,
    'v9': Regressor_v9,
    'v10': Regressor_v10,
}

data_fn = {
    'v1': get_dataloader_v1,
    'v2': get_dataloader_v1,
    'v3': get_dataloader_v1,
    'v4': get_dataloader_v2,
    'v5': get_dataloader_v2,
    'v6': get_dataloader_v2,
    'v7': get_dataloader_v3,
    'v8': get_dataloader_v3,
    'v9': get_dataloader_v2,
    'v10': get_dataloader_v2,
}

num_epochs = 500
batch_size = 64
train_losses = []
val_losses = []
best_val_loss = float(1e5)
best_epoch = 0

train_loader, val_loader = data_fn[args.model](args.data, batch_size=batch_size, val_size=0.2, device=device)
regressor = models[args.model](hidden_dim=args.hidden_dim).to(device)
criterion = torch.nn.HuberLoss(delta=1.0)

learning_rate = 1e-4
optimizer = optim.AdamW(regressor.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)


config = {
    'model': args.model,
    'hidden_dim': args.hidden_dim,
    'learning_rate': learning_rate,
    'epochs': num_epochs,
    'batch_size': batch_size,
    'optimizer': 'AdamW',
    'loss_function': 'HuberLoss',
    'cuda': args.cuda,
}
wandb.init(project="interval_predictor", name=args.run, config=config)

for epoch in tqdm(range(num_epochs)):
    # Train
    regressor.train()
    train_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        y_pred = regressor(X_batch)
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validate
    regressor.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = regressor(X_batch)
            val_loss += criterion(y_pred, y_batch).item()
            
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    scheduler.step(avg_val_loss)
    
    # Log metrics to wandb
    wandb.log({
        "train_loss": train_loss / len(train_loader),
        "val_loss": val_loss / len(val_loader),
        "learning_rate": optimizer.param_groups[0]['lr']
    })
    
    # Save (best epoch)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model = regressor.state_dict()
        best_epoch = epoch

model_path = Path(__file__).parent / 'model' / args.run
model_path.mkdir(exist_ok=True, parents=True)

with open(model_path / 'config.yaml', 'w') as f:
    yaml.dump(config, f)
    
torch.save(best_model, (model_path / f'att_regression_{best_epoch}.pth'))
wandb.finish()