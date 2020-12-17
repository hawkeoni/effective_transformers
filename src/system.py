import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.transformer import TransformerEncoder
from src.modules import Embedder
from src.dataset import Vocab, ListOpsDataset

TRANSFORMER_FACTORY = {"default": TransformerEncoder}


class ListOpsSystem(pl.LightningModule):

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        transformer_type: str = "default",
        use_sin_pos: bool = False,
        max_length: int = 2010
    ):
        super().__init__()
        self.save_hyperparameters()
        vocab_len = len(Vocab().idx2word)
        self.embedder = Embedder(d_model, vocab_len, max_length, use_sin_pos)
        transformer_cls = TRANSFORMER_FACTORY[transformer_type]
        self.encoder = transformer_cls(d_model, num_layers, num_heads, ff_dim, dropout)
        self.out = nn.Linear(d_model, 10)

    def forward(self, x: torch.LongTensor):
        """
        x - [batch, seq_len]
        return class distribution [batch, 10]
        """
        embedded = self.embedder(x)
        encoded = self.encoder(embedded)
        # x - [batch, seq_len, d_model]
        pooled = encoded[:, 0]
        # pooled - [batch, d_model]
        pred = self.out(pooled)
        # pred - [batch, 10]
        return pred
