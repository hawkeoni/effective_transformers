import torch
import torch.nn as nn


def check_encoder_dimensions_hook(self, inputs, res):
    x = inputs[0]
    assert x.shape == res.shape
    if len(inputs) == 2:
        mask = inputs[1]
        assert mask.shape == x.shape[:2]


class Encoder(nn.Module):
    """
    Base encoder class.
    Takes a tensor of shape [batch, seq_len, d_model]
    and returns a tensor of shape [batch, seq_len, d_model]
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.register_forward_hook(check_encoder_dimensions_hook)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError
