import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_planes, in_planes // ratio)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_planes // ratio, in_planes)

    def forward(self, x):
        # x = [B, T, C]
        
        # [B, C]
        avg_out = torch.mean(x, dim=1)
        # [B, C]
        max_out, _ = torch.max(x, dim=1)
        avg_out = self.fc2(self.relu1(self.fc1(avg_out)))
        max_out = self.fc2(self.relu1(self.fc1(max_out)))
        # [B, 1, C]
        scale = torch.sigmoid(avg_out + max_out).unsqueeze(1)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        # x = [B, T, C]
        
        # [B, C, T]
        x_perm = x.permute(0, 2, 1)
        # [B, 2, T]
        avg_out = torch.mean(x_perm, dim=1, keepdim=True)
        max_out, _ = torch.max(x_perm, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # [B, 1, T]
        attention = torch.sigmoid(self.conv1(x_cat))
        # [B, C, T]
        x_out = x_perm * attention
        # [B, T, C]
        return x_out.permute(0, 2, 1)

class CBAM1D(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=3):
        super(CBAM1D, self).__init__()
        self.ca = ChannelAttention(channel, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


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
        
        self.cbam = CBAM1D(channel=hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim ),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        scores = x[:, -3:]      # [B, 3]
        features = x[:, :-3]    # [B, 768]
        
        # [B, 2, hidden_dim]
        score_emb = self.mlp_score(scores).unsqueeze(1)
        feature_emb = self.mlp_feature(features).unsqueeze(1)
        emb = torch.cat((score_emb, feature_emb), dim=1)
        emb = self.cbam(emb)
        # [B, 2 * hidden_dim]
        emb = emb.view(batch_size, -1)
        # [B,]
        T_pred = self.fc(emb).squeeze(-1)
        
        return T_pred