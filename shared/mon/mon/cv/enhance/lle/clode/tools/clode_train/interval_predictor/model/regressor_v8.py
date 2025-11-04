import torch
import torch.nn as nn
from .regressor_v3 import ECALayer

class Regressor(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=32, extra_dim=5):
        super(Regressor, self).__init__()
        
        self.mlp_extra_feat = nn.Sequential(
            nn.Linear(extra_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mlp_image_emb = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        # self.cbam = CBAM1D(channel=hidden_dim)
        self.eca = ECALayer(channel=hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.hidden_dim = hidden_dim
        self.extra_dim = extra_dim

    def forward(self, x):        
        image_feat = x[:, :-self.extra_dim]  # [B, 768]
        extra_feat = x[:, -self.extra_dim:]  # [B, 5]
        image_feat = self.mlp_image_emb(image_feat)   # [B, hidden_dim]
        extra_feat = self.mlp_extra_feat(extra_feat)  # [B, hidden_dim]
        
        # [B, 2, hidden_dim]
        x = torch.stack([image_feat, extra_feat], dim=1)
        x = self.eca(x)
        
        # [B,]
        out = self.fc(x.view(x.size(0), -1)).squeeze(-1)
        return out
