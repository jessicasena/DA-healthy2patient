import torch.nn as nn
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math


class TimeSeriesTransformer(nn.Module):
    """
    code from: https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
    """

    def __init__(self,
                 n_classes,
                 batch_first: bool,
                 dim_val: int = 512,
                 n_encoder_layers: int = 4,
                 n_heads: int = 8,
                 dropout_encoder: float = 0.2,
                 dropout_pos_enc: float = 0.1,
                 dim_feedforward_encoder: int = 2048
                 ):

        super().__init__()
        self.window_size = 553

        # self.encoder_input_layer = nn.Sequential(nn.Conv1d(3, dim_val, 1), nn.GELU(),
        #                                 nn.Conv1d(dim_val, dim_val, 1), nn.GELU(),
        #                                 nn.Conv1d(dim_val, dim_val, 1), nn.GELU(),
        #                                 nn.Conv1d(dim_val, dim_val, 1), nn.GELU())
        self.encoder_input_layer = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=10),
            nn.ReLU(True),
            nn.BatchNorm1d(16),

            nn.Conv1d(16, 32, kernel_size=10),
            nn.MaxPool1d(2),
            nn.ReLU(True),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=10),
            nn.MaxPool1d(2),
            nn.ReLU(True),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 128, kernel_size=10),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Conv1d(128, 256, kernel_size=10),
            nn.MaxPool1d(2),
            nn.ReLU(True),
            nn.BatchNorm1d(256),

            nn.Conv1d(256, 512, kernel_size=10),
            nn.MaxPool1d(2),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
        )

        #self.positional_encoding_layer = PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc)

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

        #self.linear_mapping = nn.Linear(in_features=dim_val, out_features=n_classes)

    def forward(self, src: Tensor) -> Tensor:
        #print("Size of src as given to forward(): {}".format(src.size()))

        # src shape: [batch_size, src length, dim_val] regardless of number of input features
        src = self.encoder_input_layer(src)

        #print("Size of src after input layer: {}".format(src.size()))

        src = src.permute(2, 0, 1)

        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])
        src += self.position_embed

        #print("From model.forward(): Size of src after permute: {}".format(src.size()))

        # src shape: [batch_size, src length, dim_val] regardless of number of input features
        #src = self.positional_encoding_layer(src)
        #print("From model.forward(): Size of src after pos_enc layer: {}".format(src.size()))

        # src shape: if batch first: [batch_size, enc_seq_len, dim_val] else: [enc_seq_len, batch_size, dim_val]

        src = self.encoder(src=src)[0]
        #print("From model.forward(): Size of src after encoder: {}".format(src.size()))

        #src = torch.mean(src, dim=1)
        # shape [batch_size, target seq len]
        #output = self.linear_mapping(src)
        output = self.log_softmax(self.imu_head(src))
        #print("From model.forward(): decoder_output size after linear_mapping = {}".format(output.size()))

        return output


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