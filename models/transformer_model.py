import torch
from torch import nn, Tensor
import math
import torchvision


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        loss = torchvision.ops.sigmoid_focal_loss(inputs.float(), targets.float(),
                                           reduction=self.reduction, gamma=self.gamma, alpha=self.alpha)
        return loss


class TimeSeriesTransformer(nn.Module):
    """
    code from: https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
    https://github.com/yolish/har-with-imu-transformer/
    """

    def __init__(self,
                 n_classes,
                 batch_first: bool,
                 dim_val: int = 512,
                 n_encoder_layers: int = 4,
                 n_heads: int = 8,
                 dropout_encoder: float = 0.2,
                 dim_feedforward_encoder: int = 2048
                 ):

        super().__init__()
        self.window_size = 272

        self.encoder_input_layer = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=10),
            nn.SiLU(True),
            nn.BatchNorm1d(16),

            nn.Conv1d(16, 32, kernel_size=10),
            nn.MaxPool1d(2),
            nn.SiLU(True),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=10),
            nn.MaxPool1d(2),
            nn.SiLU(True),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 128, kernel_size=10),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(128),
            nn.SiLU(True),

            nn.Conv1d(128, 256, kernel_size=10),
            nn.MaxPool1d(2),
            nn.SiLU(True),
            nn.BatchNorm1d(256),

            nn.Conv1d(256, 512, kernel_size=10),
            nn.MaxPool1d(2),
            nn.SiLU(True),
            nn.BatchNorm1d(512),
        )

        self.cls_token = nn.Parameter(torch.zeros((1, dim_val)), requires_grad=True)
        self.position_embed = nn.Parameter(torch.randn(self.window_size + 1, 1, dim_val))#if batch_first else nn.Parameter(torch.randn(1, self.window_size + 1, dim_val))

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_val, nhead=n_heads,
                                                   dim_feedforward=dim_feedforward_encoder, dropout=dropout_encoder,
                                                   batch_first=batch_first)

        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_encoder_layers, norm=None)
        self.imu_head = nn.Sequential(
            nn.LayerNorm(dim_val),
            nn.Linear(dim_val, dim_val // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim_val // 4, n_classes)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor) -> Tensor:
        src = self.encoder_input_layer(src)
        src = src.permute(2, 0, 1)

        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])
        src += self.position_embed

        src = self.encoder(src=src)[0]

        logits = self.imu_head(src)

        return logits


class PositionalEncoder(nn.Module):
    def __init__(
            self,
            dropout: float = 0.1,
            max_seq_len: int = 18000,
            d_model: int = 512,
            batch_first: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0
        # copy pasted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_seq_len, d_model)

        pe[0, :, 0::2] = torch.sin(position * div_term)

        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or
               [enc_seq_len, batch_size, dim_val]
        """

        x = x + self.pe[:, :x.size(self.x_dim), :]

        return self.dropout(x)