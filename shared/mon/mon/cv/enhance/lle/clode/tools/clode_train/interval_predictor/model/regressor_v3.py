import torch
import torch.nn as nn
import torch.nn.functional as F

class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECALayer, self).__init__()
        # Adaptive kernel size based on channel size
        k_size = int(abs((torch.log2(torch.tensor(channel, dtype=torch.float32)) + b) / gamma))
        k_size = k_size if k_size % 2 else k_size + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        # x = [B, T, C]
        x_perm = x.permute(0, 2, 1)  # [B, C, T]
        y = self.avg_pool(x_perm)    # [B, C, 1]
        y = y.permute(0, 2, 1)       # [B, 1, C]
        y = self.conv(y)             # [B, 1, C]
        y = torch.sigmoid(y)        # [B, 1, C]
        y = y.permute(0, 2, 1)      # [B, C, 1]
        x_out = x_perm * y          # [B, C, T]
        return x_out.permute(0, 2, 1)  # [B, T, C]

class Regressor(nn.Module):
    def __init__(self, input_dim=768, score_dim=3, hidden_dim=64):
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
        
        self.eca = ECALayer(channel=hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        scores = x[:, -3:]      # [B, 3]
        features = x[:, :-3]    # [B, 768]
        
        # [B, 1, hidden_dim]
        score_emb = self.mlp_score(scores).unsqueeze(1)
        feature_emb = self.mlp_feature(features).unsqueeze(1)
        # [B, 2, hidden_dim]
        emb = torch.cat((score_emb, feature_emb), dim=1)  
        emb = self.eca(emb)
        # [B, 2 * hidden_dim]
        emb = emb.view(batch_size, -1)  
        # [B,]
        T_pred = self.fc(emb).squeeze(-1)  
        
        return T_pred