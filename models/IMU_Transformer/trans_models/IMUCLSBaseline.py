import torch.nn as nn
import torch
from copy import deepcopy


class IMUCLSBaseline(nn.Module):
    def __init__(self, config):

        super(IMUCLSBaseline, self).__init__()

        input_dim = config.get("input_dim")
        feature_dim = config.get("transformer_dim")
        window_size = config.get("window_size")

        self.input_proj = nn.Sequential(
            nn.Conv1d(config.get("input_dim"), feature_dim//16, (1,)),
            nn.Conv1d(feature_dim//16, feature_dim//8, (1,)),
            nn.MaxPool1d(2),
            nn.GELU(),

            nn.Conv1d(feature_dim//8, feature_dim//8, (1,)),
            nn.Conv1d(feature_dim//8, feature_dim//8, (1,)),
            nn.MaxPool1d(2),
            nn.GELU(),

            nn.Conv1d(feature_dim//8, feature_dim//4, (1,)),
            nn.Conv1d(feature_dim//4, feature_dim//4, (1,)),
            nn.Conv1d(feature_dim//4, feature_dim//4, (1,)),
            nn.MaxPool1d(2),
            nn.GELU(),

            nn.Conv1d(feature_dim//4, feature_dim//2, (1,)),
            nn.Conv1d(feature_dim//2, feature_dim//2, (1,)),
            nn.Conv1d(feature_dim//2, feature_dim, (1,)),
            nn.MaxPool1d(2),
            nn.GELU(),

            nn.Conv1d(feature_dim, feature_dim, (1,)),
            nn.Conv1d(feature_dim, feature_dim, (1,)),
            nn.Conv1d(feature_dim, feature_dim, (1,)),
            nn.MaxPool1d(2),
            nn.GELU()
            )
        num_classes = config.get("num_classes")
        self.imu_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim,  feature_dim//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim//4,  num_classes)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        """
        Forward pass
        :param x:  B X M x T tensor reprensting a batch of size B of  M sensors (measurements) X T time steps (e.g. 128 x 6 x 100)
        :return: B X N weight for each mode per sample
        """
        x = self.input_proj(data.transpose(1, 2))
        x = torch.mean(x, dim=2)
        x = self.imu_head(x)
        x = self.log_softmax(x)
        return x # B X N