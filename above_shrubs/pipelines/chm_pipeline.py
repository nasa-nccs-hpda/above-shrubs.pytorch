import os
import logging

import timm
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import repeat
from torch.utils.data import DataLoader
from multiprocessing import Pool, cpu_count
from above_shrubs.datasets.chm_dataset import CHMDataset
from above_shrubs.datamodules.chm_datamodule import CHMDataModule
from above_shrubs.pipelines.base_pipeline import BasePipeline


# temporary additions

import os
import tempfile

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from torchgeo.datamodules import EuroSAT100DataModule
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import PixelwiseRegressionTask


CHUNKS = {'band': 'auto', 'x': 'auto', 'y': 'auto'}
xp = np

__status__ = "Development"


class CHMPipeline(BasePipeline):

    # -------------------------------------------------------------------------
    # setup
    # -------------------------------------------------------------------------
    def setup(self):
        """
        Convert .tif into numpy files.
        """
        # Get data and label filenames for training
        if self.conf.train_data_dir is not None:
            p = Pool(processes=cpu_count())
            train_data_filenames = self.get_dataset_filenames(
                self.conf.train_data_dir, ext='*.tif')
            train_label_filenames = self.get_dataset_filenames(
                self.conf.train_label_dir, ext='*.tif')
            assert len(train_data_filenames) == len(train_label_filenames), \
                'Number of data and label filenames do not match'
            logging.info(f'{len(train_data_filenames)} files from TIF to NPY')
            p.starmap(
                self._tif_to_numpy,
                zip(
                    train_data_filenames,
                    train_label_filenames,
                    repeat(self.conf.train_tiles_dir)
                )
            )

        # Get data and label filenames for training
        if self.conf.test_data_dir is not None:
            p = Pool(processes=cpu_count())
            test_data_filenames = self.get_dataset_filenames(
                self.conf.test_data_dir, ext='*.tif')
            test_label_filenames = self.get_dataset_filenames(
                self.conf.test_label_dir, ext='*.tif')
            assert len(test_data_filenames) == len(test_label_filenames), \
                'Number of data and label filenames do not match'
            logging.info(f'{len(test_data_filenames)} files from TIF to NPY')
            p.starmap(
                self._tif_to_numpy,
                zip(
                    test_data_filenames,
                    test_label_filenames,
                    repeat(self.conf.test_tiles_dir)
                )
            )
        return

    # -------------------------------------------------------------------------
    # preprocess
    # -------------------------------------------------------------------------
    def preprocess(self):
        """
        Perform general preprocessing.
        """
        logging.info('Starting preprocessing stage')

        self._set_train_test_dirs()

        # Calculate mean and std values for training
        data_filenames = self.get_dataset_filenames(
            self.conf.train_data_dir, ext='*.tif')
        logging.info(f'Mean and std values from {len(data_filenames)} files.')

        # Temporarily disable standardization and augmentation
        current_standardization = self.conf.standardization
        self.conf.standardization = None
        metadata_output_filename = os.path.join(
            self.metadata_dir, 'mean-std-values.csv')

        # Set main data loader
        chm_train_dataset = CHMDataset(
            self.conf.train_data_dir,
            self.conf.train_label_dir,
            img_size=(self.conf.tile_size, self.conf.tile_size),
        )

        train_dataloader = DataLoader(
            chm_train_dataset,
            batch_size=self.conf.batch_size, shuffle=False
        )

        # Get mean and std array
        mean, std = self.get_mean_std_dataset(
            train_dataloader, metadata_output_filename)
        logging.info(f'Mean: {mean.numpy()}, Std: {std.numpy()}')

        # Re-enable standardization for next pipeline step
        self.conf.standardization = current_standardization

        logging.info('Done with preprocessing stage')

    # -------------------------------------------------------------------------
    # preprocess
    # -------------------------------------------------------------------------
    def train(self):
        """
        Perform general training.
        """
        logging.info('Starting training stage')

        self._set_train_test_dirs()

        batch_size = 10
        num_workers = 2
        max_epochs = 50
        fast_dev_run = False

        datamodule = CHMDataModule(
            train_data_dir=self.conf.train_data_dir,
            train_label_dir=self.conf.train_label_dir,
            test_data_dir=self.conf.test_data_dir,
            test_label_dir=self.conf.test_label_dir,
            batch_size=16,
            num_workers=8,
        )

        # Set main data loader
        #chm_train_dataset = CHMDataset(
        #    os.path.join(self.conf.train_tiles_dir, 'images'),
        #    os.path.join(self.conf.train_tiles_dir, 'labels'),
        #    img_size=(self.conf.tile_size, self.conf.tile_size),
        #)

        #chm_test_dataset = CHMDataset(
        #    os.path.join(self.conf.test_tiles_dir, 'images'),
        #    os.path.join(self.conf.test_tiles_dir, 'labels'),
        #    img_size=(self.conf.tile_size, self.conf.tile_size),
        #)

        # start dataloader
        #train_dataloader = DataLoader(
        #    chm_train_dataset,
        #    batch_size=self.conf.batch_size, shuffle=True
        #)

        #test_dataloader = DataLoader(
        #    chm_test_dataset,
        #    batch_size=self.conf.batch_size, shuffle=False
        #)

        #print(timm.list_models("prithvi*"))
        # Build model
        #     model = build_finetune_model(config, logger)

        # Build optimizer
        # optimizer = build_optimizer(config,
        #                            model,
        #                            is_pretrain=False,
        #                            logger=logger)

        return

    # -------------------------------------------------------------------------
    # get_mean_std_dataset
    # -------------------------------------------------------------------------
    def get_mean_std_dataset(self, dataloader, output_filename: str):

        test_sample_shape = next(iter(dataloader))['image'].shape
        if test_sample_shape[1] < test_sample_shape[2]:
            mean_dims = (0, 2, 3)
        else:
            mean_dims = (1, 2, 3)

        for index, data_dict in tqdm(enumerate(dataloader)):
            data = data_dict['image'].cuda()
            channels_sum, channels_squared_sum, num_batches = 0, 0, 0
            channels_sum += torch.mean(data, dim=mean_dims)
            channels_squared_sum += torch.mean(data**2, dim=mean_dims)
            num_batches += 1

        mean = channels_sum / num_batches
        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

        mean, std = mean.cpu(), std.cpu()

        if output_filename is not None:
            mean_std = np.stack(
                [mean.numpy(), std.numpy()], axis=0)
            pd.DataFrame(mean_std).to_csv(
                output_filename, header=None, index=None)
        return mean, std
