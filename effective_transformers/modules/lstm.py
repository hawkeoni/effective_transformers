import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from effective_transformers.modules.encoder_base import Encoder


class LSTMEncoder(Encoder):

    def __init__(self, d_model: int, num_layers: int, dropout: float, *args, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(d_model,
                            d_model,
                            bidirectional=True,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        if mask is not None:
            x_packed = pack_padded_sequence(x, mask.sum(dim=1), batch_first=True, enforce_sorted=False)
            output, final_states = self.lstm(x_packed)
            output, lengths = pad_packed_sequence(output, batch_first=True)
        else:
            output, final_states = self.lstm(x)
        # output - [batch, seq_len, 2 * d_model]
        output = self.proj(output)
        return output
