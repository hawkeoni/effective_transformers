import os
from pathlib import Path
from argparse import ArgumentParser
from typing import List

import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, Callback

from src.system import ListOpsSystem



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--serialization_dir", type=str, default="model")
    parser.add_argument("--neptune", action="store_true", default=False,
                        help="Set to log to neptune")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ListOpsSystem.add_model_specific_args(parser)
    args = parser.parse_args()

    Path(args.serialization_dir).mkdir(exist_ok=True, parents=True)
    system = ListOpsSystem(**vars(args))


    early_stopping_callback = EarlyStopping(
        "valid_acc_epoch",
        patience=3,
        mode="max"
    )
    lr_callback = LearningRateMonitor("step")
    checkpoint_callback = ModelCheckpoint(
        filepath=args.serialization_dir,
        verbose=True,
        monitor='valid_acc_epoch',
        mode='max',
        prefix='',
        save_top_k=-1,
        save_last=True)
    callbacks = [early_stopping_callback, lr_callback, checkpoint_callback]

    loggers = [
        NeptuneLogger(
            os.environ["NEPTUNE_API_TOKEN"],
            "hawkeoni/effective-transformers",
            experiment_id=args.serialization_dir
        )
    ]

    trainer = pl.Trainer.from_argparse_args(args, log_every_n_steps=1, logger=loggers, callbacks=callbacks)
    trainer.fit(system)
