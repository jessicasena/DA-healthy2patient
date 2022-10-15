import torch.nn as nn
import torch
from copy import deepcopy


class IMUCLSBaseline(nn.Module):
    def __init__(self, config):

        super(IMUCLSBaseline, self).__init__()

        cnn_blocks = config.get("cnn_blocks")

        self.input_proj = nn.Sequential()
        dim = 16
        for i in range(1, cnn_blocks):
            input_dim = config.get("input_dim") if i == 1 else dim
            self.input_proj.add_module(f"conv_1_{i}", nn.Conv1d(input_dim, dim, (1,)))
            self.input_proj.add_module(f"conv_2_{i}", nn.Conv1d(dim, dim * 2, (1,)))
            self.input_proj.add_module(f"maxpool_{i}", nn.MaxPool1d(2))
            self.input_proj.add_module(f"gelu_{i}", nn.GELU())
            dim = dim * 2

        num_classes = config.get("num_classes")
        dim = config.get("sample_size") // (2 ** (cnn_blocks - 1))
        self.imu_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,  dim//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim//4,  num_classes)
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
        x = self.input_proj(data)
        x = torch.mean(x, dim=1)
        x = self.imu_head(x)
        x = self.log_softmax(x)
        return x # B X N