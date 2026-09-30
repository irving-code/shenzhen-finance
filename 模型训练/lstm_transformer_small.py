import torch
from torch import nn


class LSTMTransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim=27,
        sequence_length=20,
        projection_dim=64,
        lstm_hidden_dim=64,
        lstm_layers=1,
        transformer_layers=1,
        transformer_heads=4,
        transformer_feedforward_dim=128,
        dropout=0.1
    ):
        super().__init__()
        if projection_dim % transformer_heads != 0:
            raise ValueError('projection_dim 必须能被 transformer_heads 整除')
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU()
        )
        self.lstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        if lstm_hidden_dim != projection_dim:
            self.lstm_projection = nn.Linear(lstm_hidden_dim, projection_dim)
        else:
            self.lstm_projection = nn.Identity()
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, projection_dim)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=projection_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_feedforward_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(projection_dim),
            nn.Linear(projection_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self.sequence_length = sequence_length

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError('输入必须是三维张量(batch, sequence, feature)')
        if x.shape[1] != self.sequence_length:
            raise ValueError('输入序列长度与模型配置不一致')
        x = self.input_projection(x)
        x, _ = self.lstm(x)
        x = self.lstm_projection(x)
        x = x + self.position_embedding[:, :x.shape[1], :]
        x = self.transformer(x)
        return self.output_head(x[:, -1, :]).squeeze(-1)
