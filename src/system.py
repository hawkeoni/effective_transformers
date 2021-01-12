from argparse import ArgumentParser
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

from src.lstm import LSTMEncoder
from src.transformer import TransformerEncoder
from src.modules import Embedder
from src.dataset import Vocab, ListOpsDataset

TRANSFORMER_FACTORY = {
    "transformer": TransformerEncoder,
    "lstm": LSTMEncoder,
    "performer": None,
    "linear": None
    }


class ListOpsSystem(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--d_model", type=int, default=512)
        parser.add_argument("--num_layers", type=int, default=6)
        parser.add_argument("--num_heads", type=int, default=8)
        parser.add_argument("--ff_dim", type=int, default=2048)
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--model_type", type=str, default="transformer")
        parser.add_argument("--use_sin_pos", type=bool, default=False)
        parser.add_argument("--max_length", type=int, default=2010)
        parser.add_argument("--warmup_steps", type=int, default=1000)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--base_lr", type=float, default=2e-5)
        return parser

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        model_type: str = "default",
        use_sin_pos: bool = False,
        max_length: int = 2010,
        warmup_steps: int = 2000,
        batch_size: int = 32,
        base_lr: float = 2e-5,
        *args, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.step = 0
        vocab_len = len(Vocab().idx2word)
        self.embedder = Embedder(d_model, vocab_len, max_length, use_sin_pos)
        transformer_cls = TRANSFORMER_FACTORY[model_type]
        self.encoder = transformer_cls(
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout)
        self.out = nn.Linear(d_model, 10)
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.training_acc = pl.metrics.Accuracy()
        self.validation_acc = pl.metrics.Accuracy()

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
        self.log("train_acc", self.training_acc(p, y),
                 on_step=True, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # x - [batch, seq_len] y - [batch]
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        p = pred.argmax(dim=1)
        self.log("valid_loss", loss)
        self.log("valid_acc", self.validation_acc(p, y),
                 on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.base_lr)
        return optimizer

    def optimizer_step(
            self,
            current_epoch,
            batch_nb,
            optimizer,
            optimizer_idx,
            closure,
            on_tpu=False,
            using_native_amp=False,
            using_lbfgs=False):
        super().optimizer_step(
                current_epoch,
                batch_nb,
                optimizer,
                optimizer_idx,
                closure,
                on_tpu,
                using_native_amp,
                using_lbfgs)
        self.step += 1
        warmup_steps = self.hparams.get("warmup_steps", 1)
        lr = min(self.step / warmup_steps, 1.) * self.base_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        self.log("lr", lr, on_step=True, prog_bar=True)

    def train_dataloader(self) -> Optional[DataLoader]:
        try:
            dataset = ListOpsDataset("dataset/basic_train.csv")
        except:
            return
        loader = DataLoader(dataset, self.batch_size, collate_fn=self.collate_fn, num_workers=4)
        return loader

    def val_dataloader(self) -> Optional[DataLoader]:
        try:
            dataset = ListOpsDataset("dataset/basic_val.csv")
        except:
            return
        loader = DataLoader(dataset, self.batch_size, collate_fn=self.collate_fn, num_workers=4)
        return loader

    def test_dataloader(self) -> Optional[DataLoader]:
        try:
            dataset = ListOpsDataset("dataset/basic_test.csv")
        except:
            return
        loader = DataLoader(dataset, self.batch_size, collate_fn=self.collate_fn, num_workers=4)
        return loader

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = list(zip(*batch))
        y = torch.stack(y, dim=0).squeeze(1)
        x = pad_sequence(x, batch_first=True, padding_value=0)
        return x, y
