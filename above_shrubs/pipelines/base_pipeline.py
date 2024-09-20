import os
import sys
import torch
import logging
import omegaconf
import numpy as np
import pandas as pd
import rioxarray as rxr

from glob import glob
from pathlib import Path
from datetime import datetime
from above_shrubs.config import Config

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


class BasePipeline(object):
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

        # Set logger
        self.logger = logger if logger is not None else self._set_logger()
        logging.info(f'Log output available at {self.conf.work_dir}')

        # Seed everything for repeatability
        self.seed_everything(self.conf.seed)

        # output directory to store metadata and artifacts
        self.metadata_dir = os.path.join(self.conf.work_dir, 'metadata')
        os.makedirs(self.metadata_dir, exist_ok=True)
        logging.info(f'Metadata dir: {self.metadata_dir}')

        """
        # rewrite model filename option if given from CLI
        if model_filename is not None:
            assert os.path.exists(model_filename), \
                f'{model_filename} does not exist.'
            self.conf.model_filename = model_filename

        # rewrite output directory if given from CLI
        if output_dir is not None:
            self.conf.inference_save_dir = output_dir
            os.makedirs(self.conf.inference_save_dir, exist_ok=True)

        # rewrite inference regex list
        if inference_regex_list is not None:
            self.conf.inference_regex_list = inference_regex_list



        # Set output directories and locations
        self.images_dir = os.path.join(self.conf.data_dir, 'images')
        self.logger.info(f'Images dir: {self.images_dir}')

        self.labels_dir = os.path.join(self.conf.data_dir, 'labels')
        self.logger.info(f'Labels dir: {self.labels_dir}')

        self.model_dir = self.conf.model_dir
        self.logger.info(f'Model dir: {self.model_dir}')

        # Create output directories
        for out_dir in [
                self.images_dir, self.labels_dir,
                self.metadata_dir, self.model_dir]:
            os.makedirs(out_dir, exist_ok=True)

        logging.info(f'Output dir: {self.conf.inference_save_dir}')

        # save configuration into the model directory
        try:
            OmegaConf.save(
                self.conf, os.path.join(self.model_dir, 'config.yaml'))
        except PermissionError:
            logging.info('No permissions to save config, skipping step.')


        """

    # -------------------------------------------------------------------------
    # _read_config
    # -------------------------------------------------------------------------
    def _read_config(self, filename: str, config_class=Config):
        """
        Read configuration filename and initiate objects
        """
        # Configuration file initialization
        schema = omegaconf.OmegaConf.structured(config_class)
        conf = omegaconf.OmegaConf.load(filename)
        try:
            conf = omegaconf.OmegaConf.merge(schema, conf)
        except BaseException as err:
            sys.exit(f"ERROR: {err}")
        return conf

    # -------------------------------------------------------------------------
    # _set_logger
    # -------------------------------------------------------------------------
    def _set_logger(self):
        """
        Set logger configuration.
        """
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        # set console output
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # set filename output
        log_filename = f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'
        os.makedirs(os.path.join(self.conf.work_dir, 'logs'), exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(self.conf.work_dir, 'logs', log_filename))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # remove the root logger
        logger.handlers.pop(0)
        return logger

    # -------------------------------------------------------------------------
    # seed_everything
    # -------------------------------------------------------------------------
    def seed_everything(self, seed: int = 42) -> None:
        """
        Seeds starting randomization from libraries.
        Args:
            seed (int): integer to seed libraries with.
        Returns:
            None.
        """
        np.random.seed(seed)
        if HAS_GPU:
            try:
                cp.random.seed(seed)
            except RuntimeError:
                return
        return

    # -------------------------------------------------------------------------
    # seed_everything
    # -------------------------------------------------------------------------
    def get_dataset_filenames(self, data_dir: str, ext: str = '*.npy') -> list:
        """
        Get dataset filenames for training.
        """
        data_filenames = sorted(glob(os.path.join(data_dir, ext)))
        assert len(data_filenames) > 0, f'No files under {data_dir}.'
        return data_filenames

    # -------------------------------------------------------------------------
    # _tif_to_numpy
    # -------------------------------------------------------------------------
    def _tif_to_numpy(self, data_filename, label_filename, output_dir):
        """
        Convert TIF to NP.
        """
        # open the imagery
        image = rxr.open_rasterio(data_filename).values
        label = rxr.open_rasterio(label_filename).values

        if np.isnan(label).any():
            return

        # get output filenames
        image_output_dir = os.path.join(output_dir, 'images')
        os.makedirs(image_output_dir, exist_ok=True)

        label_output_dir = os.path.join(output_dir, 'labels')
        os.makedirs(label_output_dir, exist_ok=True)

        # save the new arrays
        np.save(
            os.path.join(
                image_output_dir,
                f'{Path(data_filename).stem}.npy'
            ), image)
        np.save(
            os.path.join(
                label_output_dir,
                f'{Path(label_filename).stem}.npy'
            ), label)
        return

    # -------------------------------------------------------------------------
    # _set_train_test_dirs
    # -------------------------------------------------------------------------
    def _set_train_test_dirs(self):

        # Set output directories and locations
        if self.conf.train_tiles_dir is not None:
            self.train_images_dir = os.path.join(
                self.conf.train_tiles_dir, 'images')
            self.train_labels_dir = os.path.join(
                self.conf.train_tiles_dir, 'labels')
        self.logger.info(f'Train images dir: {self.train_images_dir}')

        # Set output directories and locations
        if self.conf.test_tiles_dir is not None:
            self.test_images_dir = os.path.join(
                self.conf.test_tiles_dir, 'images')
            self.test_labels_dir = os.path.join(
                self.conf.test_tiles_dir, 'labels')
        self.logger.info(f'Test images dir: {self.test_images_dir}')

        return

    # -------------------------------------------------------------------------
    # get_mean_std_dataset
    # -------------------------------------------------------------------------
    def get_mean_std_dataset(self, dataloader, output_filename: str):

        for index, (data, _) in enumerate(dataloader):
            channels_sum, channels_squared_sum, num_batches = 0, 0, 0
            channels_sum += torch.mean(data, dim=(1, 2, 3))
            channels_squared_sum += torch.mean(data**2, dim=(1, 2, 3))
            num_batches += 1

        mean = channels_sum / num_batches
        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

        if output_filename is not None:
            mean_std = np.stack([mean.numpy(), std.numpy()], axis=0)
            pd.DataFrame(mean_std).to_csv(
                output_filename, header=None, index=None)
        return mean, std
