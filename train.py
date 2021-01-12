from pathlib import Path
from argparse import ArgumentParser

import pytorch_lightning as pl

from src.system import ListOpsSystem


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--serialization_dir", type=str, default="model")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ListOpsSystem.add_model_specific_args(parser)
    args = parser.parse_args()
    Path(args.serialization_dir).mkdir(exist_ok=True, parents=True)

    system = ListOpsSystem(**vars(args))
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.serialization_dir,
        verbose=True,
        #monitor='valid_acc_epoch',
        mode='max',
        prefix='',
        save_top_k=-1,
        save_last=True)
    trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback, log_every_n_steps=1)
    trainer.fit(system)
