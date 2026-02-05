import torch
import torch.nn as nn
import torch.nn.functional as F
from .regressor_v2 import CBAM1D

class Regressor(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=64):
        super(Regressor, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.cbam = CBAM1D(channel=hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # [B, 15, 768]
        batch_size, _, input_dim = x.shape
        
        # [B * 15, 768]
        x = x.view(-1, input_dim)
        # [B * 15, hidden_dim]
        x = self.mlp(x)
        # [B, 15, hidden_dim]
        x = x.view(batch_size, -1, self.hidden_dim)
        # [B, hidden_dim]
        emb = self.cbam(x)[:, -1, :]
        
        # [B,]
        out = self.fc(emb).squeeze(-1)
        return out
