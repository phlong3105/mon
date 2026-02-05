import torch
import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self, input_dim=768, score_dim=3, hidden_dim=32):
        super(Regressor, self).__init__()
        
        self.mlp_feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        
        self.mlp_score = nn.Sequential(
            nn.Linear(score_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        scores = x[:, -3:]      # [B, 3]
        features = x[:, :-3]    # [B, 768]
        
        # MLP: [B, 1, hidden dim]
        score_emb = self.mlp_score(scores).unsqueeze(1)
        
        # MLP + self attention: [B, 1, hidden dim]
        feature_emb = self.mlp_feature(features)
        feature_emb = self.transformer(feature_emb).unsqueeze(1)
        
        # cross attention
        attn_output, _ = self.cross_attention(query=score_emb, key=feature_emb, value=feature_emb)     
        T_pred = self.fc(attn_output.squeeze(1)).squeeze(-1)
        
        return T_pred