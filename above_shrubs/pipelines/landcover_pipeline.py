import os
import logging
import torch
import lightning as pl

from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchgeo.trainers import SemanticSegmentationTask

from above_shrubs.pipelines.base_pipeline import BasePipeline
from above_shrubs.datamodules.landcover_datamodule import LandCoverSegmentationDataModule
# from above_shrubs.utils.callbacks_utils import get_callbacks
# from above_shrubs.utils.logger_utils import get_loggers

from above_shrubs.config import LandCoverConfig as Config
from above_shrubs.utils.callbacks_utils import get_callbacks
from above_shrubs.utils.logger_utils import get_loggers

class LandcoverSegmentationPipeline(BasePipeline):

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(
                self,
                config_filename: str,
                model_filename: str = None,
                output_dir: str = None,
                inference_regex_list: list = None,
                default_config: str = 'templates/chm_cnn_default.yaml',
                logger=None
            ):
        """Constructor method
        """

        # Configuration file intialization
        if config_filename is None:
            config_filename = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                default_config)
            logging.info(f'Loading default config: {config_filename}')

        # Read configuration file
        self.conf = self._read_config(config_filename, Config)
        logging.info(f'Logged configuration file: {config_filename}')

        # Set working directory if not available in configuration file
        if self.conf.work_dir is None:
            self.conf.work_dir = os.path.join(
                Path.home(), 'above-shrubs-output')

        # Set working directory if not available in configuration file
        if self.conf.model_dir is None:
            self.conf.model_dir = os.path.join(
                Path.home(), 'above-shrubs-output')

        # Create directories by default
        os.makedirs(self.conf.work_dir, exist_ok=True)
        os.makedirs(self.conf.model_dir, exist_ok=True)

        # Set logger
        self.logger = logger if logger is not None else self._set_logger()
        logging.info(f'Log output available at {self.conf.model_dir}')

        # Seed everything for repeatability
        self.seed_everything(self.conf.seed)

    def train(self):

        logging.info("Starting training stage")

        # Data module
        datamodule = LandCoverSegmentationDataModule(
            csv_path=self.conf.csv_path,
            tile_size=self.conf.tile_size,
            batch_size=self.conf.batch_size,
            num_workers=self.conf.num_workers,
            nodata_threshold=self.conf.nodata_threshold,
            # train_transforms=self.conf.train_transforms
        )

        # Model / Task
        task = SemanticSegmentationTask(
            model=self.conf.decoder_name,          # e.g., 'deeplabv3+'
            backbone=self.conf.backbone_name,      # e.g., 'resnet18'
            in_channels=len(self.conf.output_bands),     # number of image bands
            num_classes=self.conf.num_classes,     # number of classes
            lr=self.conf.learning_rate,
            loss=self.conf.loss_func,                             # CrossEntropyLoss
            # ignore_index=self.conf.ignore_index    # optional: class to ignore
        )

        # Callbacks
        callbacks = get_callbacks(self.conf.callbacks)
        loggers = get_loggers(self.conf.loggers)

        # Trainer
        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            callbacks=callbacks,
            logger=loggers,
            max_epochs=self.conf.num_epochs,
            strategy="ddp_find_unused_parameters_true",
            log_every_n_steps=1,
        )

        # Train
        trainer.fit(task, datamodule=datamodule)

        # Optional: test
        # trainer.test(task, datamodule=datamodule)
        
    def predict(self):
        logging.info("Prediction not implemented in this example.")
        # You can adapt your `sliding_window_tiler_multiclass` logic here
        raise NotImplementedError("Prediction needs to be implemented.")
