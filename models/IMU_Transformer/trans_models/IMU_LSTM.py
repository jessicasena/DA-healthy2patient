"""
IMUTransformerEncoder model
"""

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class IMU_LSTM(nn.Module):

    def __init__(self, config):
        """
        config: (dict) configuration of the model
        """
        super().__init__()

        self.hidden = config.get("transformer_dim")

        self.input_proj = nn.Sequential(
            nn.Conv1d(config.get("input_dim"), self.hidden // 16, (1,)),
            nn.Conv1d(self.hidden // 16, self.hidden // 8, (1,)),
            nn.MaxPool1d(2),
            nn.GELU(),
            nn.BatchNorm1d(self.hidden // 8),

            nn.Conv1d(self.hidden // 8, self.hidden // 8, (1,)),
            nn.Conv1d(self.hidden // 8, self.hidden // 8, (1,)),
            nn.MaxPool1d(2),
            nn.GELU(),
            nn.BatchNorm1d(self.hidden // 8),
            nn.Dropout(0.1),

            nn.Conv1d(self.hidden // 8, self.hidden // 4, (1,)),
            nn.Conv1d(self.hidden // 4, self.hidden // 4, (1,)),
            nn.Conv1d(self.hidden // 4, self.hidden // 4, (1,)),
            nn.MaxPool1d(2),
            nn.GELU(),
            nn.BatchNorm1d(self.hidden // 4),
            nn.Dropout(0.1),

            nn.Conv1d(self.hidden // 4, self.hidden // 2, (1,)),
            nn.Conv1d(self.hidden // 2, self.hidden // 2, (1,)),
            nn.Conv1d(self.hidden // 2, self.hidden, (1,)),
            nn.MaxPool1d(2),
            nn.GELU(),
            nn.BatchNorm1d(self.hidden),
            nn.Dropout(0.1),

            nn.Conv1d(self.hidden, self.hidden, (1,)),
            nn.Conv1d(self.hidden, self.hidden, (1,)),
            nn.Conv1d(self.hidden, self.hidden, (1,)),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.BatchNorm1d(self.hidden)
        )
        lstm_hidden = 256
        self.lstm = nn.LSTM(input_size=self.hidden, hidden_size=lstm_hidden, num_layers=3, batch_first=True)

        num_classes=config.get("num_classes")
        self.imu_head = nn.Sequential(
            nn.Linear(lstm_hidden,  lstm_hidden//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(lstm_hidden//4,  num_classes)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
         # (seq, batch, feature)

        # Embed in a high dimensional space and reshape to Transformer's expected shape
        src = self.input_proj(data).permute(2, 0, 1)

        src = self.lstm(src)[0]
        src = torch.mean(src, dim=1)

        # Class probability
        target = self.log_softmax(self.imu_head(src))
        return target

def get_activation(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    raise RuntimeError("Activation {} not supported".format(activation))