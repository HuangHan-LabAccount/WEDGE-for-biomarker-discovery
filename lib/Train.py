import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.callbacks import LearningRateMonitor
import time
import os
import sys
from pytorch_lightning.callbacks import EarlyStopping
sys.path.append('lib/')
def create_trainer(max_epochs=100,patience=10,min_delta=1e-4,
                   accumulate_grad_batches=1,check_val_every_n_epoch=1,
                   log_dir="lightning_logs",experiment_name="heterogcn",
                   save_dir="checkpoints",**trainer_kwargs):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=experiment_name,
        default_hp_metric=False
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(save_dir, experiment_name),
            filename='{epoch}-{val_total_loss:.4f}',
            monitor='val_total_loss',
            mode='min',
            save_top_k=5,
            save_last=True,
            verbose=True
        ),

        EarlyStopping(
            monitor='val_total_loss',
            min_delta=min_delta,
            patience=patience,
            verbose=True,
            mode='min'
        ),

        LearningRateMonitor(logging_interval='epoch')
    ]

    # create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
        check_val_every_n_epoch=check_val_every_n_epoch,
        precision='16-mixed',
        accelerator='auto',
        devices='auto',
        strategy='auto',
        enable_progress_bar=True,
        **trainer_kwargs
    )

    return trainer



