import torch
import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=64):
        super(Regressor, self).__init__()

        self.vision_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # [B, num_p + 1, 512] -> [num_p + B, 512] -> [num_p + B, 64]
        x = torch.cat([x[0, :-1, :], x[:, -1, :]], dim=0)
        x = self.vision_mlp(x)
        
        image_emb = x[-batch_size:, :].unsqueeze(dim=1)             # [B, 1, 64]
        prompt_emb = x[:-batch_size, :].repeat(batch_size, 1, 1)    # [B, num_p, 64]

        # [B, 64]
        attn_output, _ = self.cross_attention(image_emb, prompt_emb, prompt_emb)
        attn_output = attn_output.squeeze(1)
        
        # [B,]
        T_pred = self.fc(attn_output).squeeze(-1)
        return T_pred