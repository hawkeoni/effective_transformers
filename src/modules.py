import math

import torch
import torch.nn as nn
import einops as ein


def gen_wavelength_embedding(d_model: int, max_length: int) -> torch.Tensor:
    """
    Taken from https://nlp.seas.harvard.edu/2018/04/03/attention.html.
    Returns an embedding matrix of shape [1, max_length, d_model]
    """
    pos_embedding = torch.zeros(max_length, d_model)
    position = torch.arange(0, max_length).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
    )
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)
    return pos_embedding.unsqueeze(0)


class Embedder(nn.Module):

    def __init__(self, d_model: int, vocab_size: int, max_length: int, use_sin_pos: bool = False):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.use_sin_pos = use_sin_pos
        if use_sin_pos:
            self.register_buffer("pos_emb", gen_wavelength_embedding(d_model, max_length))
        else:
            self.pos_emb = nn.Embedding(max_length, d_model)

    def forward(self, x: torch.LongTensor):
        """
        x - [batch_size, seq_len]
        return [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = x.shape
        emb = self.word_emb(x)
        if self.use_sin_pos:
            pos_emb = self.pos_emb[:, :seq_len]
        else:
            pos_emb = torch.arange(seq_len)
            pos_emb = ein.repeat(positions, "seq_len -> batch seq_len", batch=batch_size)
            pos_emb = self.pos_emb(pos_emb)
        return emb + pos_emb

