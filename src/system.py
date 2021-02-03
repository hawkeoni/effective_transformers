import logging
from argparse import ArgumentParser
from typing import List, Tuple, Optional, Dict, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

from src.lstm import LSTMEncoder
from src.transformer import TransformerEncoder, PytorchTransformerEncoder
from src.modules import Embedder
from src.dataset import Vocab, ListOpsDataset

logger = logging.getLogger(__name__)
TRANSFORMER_FACTORY = {
    "transformer": TransformerEncoder,
    "pytorch_transformer": PytorchTransformerEncoder,
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
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--dataset_dir", type=str, default="dataset")
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
        weight_decay: float = 0.,
        dataset_dir: str = "dataset",
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
        self.weight_decay = weight_decay
        self.dataset_path = Path(dataset_dir)

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
        return {"loss": loss, "total": y.size(0), "correct": (p == y).sum().item()}

    def validation_step(self, batch, batch_idx):
        # x - [batch, seq_len] y - [batch]
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        p = pred.argmax(dim=1)
        return {"loss": loss, "total": y.size(0), "correct": (p == y).sum().item()}

    def validation_epoch_end(self, outs: List[Dict[str, float]]):
        total = 0
        correct = 0
        loss = 0
        for out in outs:
            total += out["total"]
            correct += out["correct"]
            loss += out["loss"].item()
        self.log("valid_acc_epoch", correct / (total + 1e-12))
        self.log("valid_loss_epoch", loss / (len(outs) + 1))

    def training_epoch_end(self, outs: List[Dict[str, float]]):
        loss = 0
        total = 0
        correct = 0
        for out in outs:
            total += out["total"]
            correct += out["correct"]
            loss += out["loss"].item()
        self.log("train_acc_epoch", correct / (total + 1e-12))
        self.log("train_loss_epoch", loss / (len(outs) + 1))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        return optimizer

    def optimizer_step(
            self,
            epoch: int = None,
            batch_idx: int = None,
            optimizer: "Optimizer" = None,
            optimizer_idx: int = None,
            optimizer_closure: Optional[Callable] = None,
            on_tpu: bool = None,
            using_native_amp: bool = None,
            using_lbfgs: bool = None,
    ):
        self.step += 1
        warmup_steps = self.hparams.get("warmup_steps", 1)
        lr = self.base_lr
        lr *= min(self.step / warmup_steps, 1.)
        lr /= max(self.step, warmup_steps) ** 0.5
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        self.log("lr", lr)
        optimizer.step()
        optimizer.zero_grad()

    def train_dataloader(self) -> Optional[DataLoader]:
        dataset = ListOpsDataset(self.dataset_path / "train.tsv")
        loader = DataLoader(dataset, self.batch_size, collate_fn=self.collate_fn)
        return loader

    def val_dataloader(self) -> Optional[DataLoader]:
        path = self.dataset_path / "val.tsv"
        if not path.exists():
            return
        dataset = ListOpsDataset(path)
        loader = DataLoader(dataset, self.batch_size, collate_fn=self.collate_fn)
        return loader

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = list(zip(*batch))
        y = torch.stack(y, dim=0).squeeze(1)
        x = pad_sequence(x, batch_first=True, padding_value=0)
        return x, y
