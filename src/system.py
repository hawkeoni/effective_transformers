import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.transformer import TransformerEncoder
from src.modules import Embedder

TRANSFORMER_FACTORY = {"default": TransformerEncoder}

class ListOpsSystem(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        transformer_type: str = "default",
    ):
        super().__init__(self)
        self.save_hyperparameters()
        self.embedder = Embedder(params)
        self.encoder = TRANSFORMER_FACTORY[transformer_type](d_model, num_layers, num_heads)
        self.out = nn.Linear(d_model, 10)

    def forward(self, x: torch.LongTensor):
        """
        x - [batch, seq_len]
        return class distribution [batch, 10]
        """
        embedded = self.embedder(x)
        encoded = self.encoder(x)
        # x - [batch, seq_len, d_model]
        pooled = x[:, 0]
        # pooled - [batch, d_model]
        pred = self.out(pooled)
        # pred - [batch, 10]
        return pred
