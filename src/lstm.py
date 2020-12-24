import torch
import torch.nn as nn



class LSTMEncoder(nn.Module):

    def __init__(self, d_model: int, num_layers: int, dropout: float, *args, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_model, bidirectional=True, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        output, _ = self.lstm(x)
        # output - [batch, seq_len, 2 * d_model]
        return self.proj(output)
