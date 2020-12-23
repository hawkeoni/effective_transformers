from argparse import ArgumentParser
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

from src.transformer import TransformerEncoder
from src.modules import Embedder
from src.dataset import Vocab, ListOpsDataset

TRANSFORMER_FACTORY = {"default": TransformerEncoder}


class ListOpsSystem(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--d_model", type=int, default=512)
        parser.add_argument("--num_layers", type=int, default=6)
        parser.add_argument("--num_heads", type=int, default=8)
        parser.add_argument("--ff_dim", type=int, default=2048)
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--transformer_type", type=str, default="default")
        parser.add_argument("--use_sin_pos", type=bool, default=False)
        parser.add_argument("--max_length", type=int, default=2010)
        parser.add_argument("--warmup_steps", type=int, default=1000)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument('--encoder_layers', type=int, default=12)
        return parser

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        transformer_type: str = "default",
        use_sin_pos: bool = False,
        max_length: int = 2010,
        warmup_steps: int = 2000,
        batch_size: int = 32,
        *args, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        vocab_len = len(Vocab().idx2word)
        self.embedder = Embedder(d_model, vocab_len, max_length, use_sin_pos)
        transformer_cls = TRANSFORMER_FACTORY[transformer_type]
        self.encoder = transformer_cls(d_model, num_layers, num_heads, ff_dim, dropout)
        self.out = nn.Linear(d_model, 10)
        self.base_lr = 2e-5
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = batch_size

    def forward(self, x: torch.LongTensor, mask: torch.Tensor = None):
        """
        x - [batch, seq_len]
        return class distribution [batch, 10]
        """
        if mask is None:
            mask = (x != 0)
        embedded = self.embedder(x)
        encoded = self.encoder(embedded, mask)
        # x - [batch, seq_len, d_model]
        pooled = encoded[:, 0]
        # pooled - [batch, d_model]
        pred = self.out(pooled)
        # pred - [batch, 10]
        return pred

    def training_step(self, batch, batch_idx):
        # x - [batch, seq_len] y - [batch]
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        p = pred.argmax(dim=1)
        true = (p == y).sum().item()
        total = y.size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss, "total": total, "true": true}

    def validation_step(self, batch, batch_idx):
        # x - [batch, seq_len] y - [batch]
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        p = pred.argmax(dim=1)
        true = (p == y).sum().item()
        total = y.size(0)
        self.log("valid_loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss, "total": total, "true": true}

    def training_epoch_end(self, outputs):
        total = sum([x["total"] for x in outputs])
        correct = sum([x["true"] for x in outputs])
        accuracy = correct / (total + 1e-12)
        self.log("train_accuracy", accuracy, on_epoch=True)

    def validation_epoch_end(self, outputs):
        total = sum([x["total"] for x in outputs])
        correct = sum([x["true"] for x in outputs])
        accuracy = correct / (total + 1e-12)
        self.log("valid_accuracy", accuracy, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.base_lr)
        return optimizer

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        super().optimizer_step(current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu, using_native_amp, using_lbfgs)
        warmup_steps = self.hparams["warmup_steps"]
        lr = min(batch_nb / warmup_steps * self.base_lr, 1.) * self.base_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def train_dataloader(self) -> DataLoader:
        dataset = ListOpsDataset("dataset/basic_train.csv")
        loader = DataLoader(dataset, self.batch_size, collate_fn=self.collate_fn)#, num_workers=8)
        return loader

    def val_dataloader(self) -> DataLoader:
        dataset = ListOpsDataset("dataset/basic_val.csv")
        loader = DataLoader(dataset, self.batch_size, collate_fn=self.collate_fn)#, num_workers=8)
        return loader

    def test_dataloader(self) -> DataLoader:
        dataset = ListOpsDataset("dataset/basic_test.csv")
        loader = DataLoader(dataset, self.batch_size, collate_fn=self.collate_fn)#, num_workers=8)
        return loader

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = list(zip(*batch))
        y = torch.stack(y, dim=0).squeeze(1)
        x = pad_sequence(x, batch_first=True, padding_value=0)
        return x, y
