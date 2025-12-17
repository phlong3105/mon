import torch
import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=64):
        super(Regressor, self).__init__()

        self.vision_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU()
        )

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        # [B, 4, 64]
        x = self.vision_mlp(x)
        
        image_emb = x[:, -1, :].unsqueeze(dim=1)
        prompt_emb = x[:, :-1, :]

        # [B, 64]
        attn_output, _ = self.cross_attention(image_emb, prompt_emb, prompt_emb)
        attn_output = attn_output.squeeze(1)
        
        # [B,]
        T_pred = self.fc(attn_output).squeeze(-1)
        return T_pred