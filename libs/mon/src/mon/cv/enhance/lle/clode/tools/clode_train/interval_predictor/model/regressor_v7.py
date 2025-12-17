import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, C]
        weight = self.fc(x)  # [B, C]
        return x * weight

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
        self.gate_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.se = SELayer(channel=hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.hidden_dim = hidden_dim
        self.extra_dim = extra_dim

    def forward(self, x):        
        image_feat = x[:, :-self.extra_dim]  # [B, 768]
        extra_feat = x[:, -self.extra_dim:]  # [B, 5]
        image_feat = self.mlp_image_emb(image_feat)   # [B, hidden_dim]
        extra_feat = self.mlp_extra_feat(extra_feat)  # [B, hidden_dim]
        
        # [B, 2 * hidden_dim]
        x = torch.cat([image_feat, extra_feat], dim=-1)
        
        # [B, hidden_dim]
        gate = torch.sigmoid(self.gate_fc(x))
        fusion = gate * image_feat + (1 - gate) * extra_feat
        # [B, hidden_dim]
        x = self.se(fusion)
        
        # [B,]
        out = self.fc(x).squeeze(-1)
        return out
