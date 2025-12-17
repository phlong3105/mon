import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
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
        
        image_emb = x[:, -1, :].unsqueeze(1)    # [B, 1, hidden_dim]
        prompt_emb = x[:, :-1, :]               # [B, 14, hidden_dim]
        
        # [B, 1, hidden_dim]
        attn_output, _ = self.cross_attention(query=image_emb, key=prompt_emb, value=prompt_emb)
        
        # [B,]
        out = self.fc(attn_output.squeeze(1)).squeeze(-1)
        return out
