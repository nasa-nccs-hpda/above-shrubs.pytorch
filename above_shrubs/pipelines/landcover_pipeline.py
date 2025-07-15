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
from above_shrubs.inference import sliding_window_tiler_multiclass



from above_shrubs.inference import sliding_window_tiler_multiclass
import timm

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput

import torch
import torch.nn as nn
from transformers import AutoModel

from above_shrubs.utils.callbacks_utils import get_callbacks
from above_shrubs.utils.logger_utils import get_loggers

from pytorch_lightning.strategies import DDPStrategy

from above_shrubs.utils.model_utils import CHMPixelwiseRegressionTask


from above_shrubs.encoders.segformer_module import SegFormerLightningModule


from glob import glob
import os
import logging

import time
import timm
import torch
import torchgeo
import numpy as np
import pandas as pd
import multiprocessing

from tqdm import tqdm
from itertools import repeat
from torch.utils.data import DataLoader
from multiprocessing import Pool, cpu_count
from above_shrubs.datasets.chm_dataset import CHMDataset
from above_shrubs.datamodules.chm_datamodule import CHMDataModule
from above_shrubs.pipelines.base_pipeline import BasePipeline

from typing import List
import logging
import rasterio
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr

from tqdm import tqdm
from glob import glob
from pathlib import Path
from itertools import repeat
from omegaconf import OmegaConf
from pygeotools.lib import iolib, warplib
from multiprocessing import Pool, cpu_count
from omegaconf.listconfig import ListConfig

import torchvision.transforms as transforms
from transformers import AutoConfig, AutoModel

from transformers import Dinov2Model



# temporary additions

import os
import tempfile

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

#from torchgeo.datamodules import EuroSAT100DataModule
#from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import PixelwiseRegressionTask


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
            num_classes=self.conf.num_classes
            # train_transforms=self.conf.train_transforms
        )

        # Model / Task
        if self.conf.decoder_name == 'deeplabv3+':
            task = SemanticSegmentationTask(
                model=self.conf.decoder_name,          # e.g., 'deeplabv3+'
                backbone=self.conf.backbone_name,      # e.g., 'resnet18'
                in_channels=len(self.conf.output_bands),     # number of image bands
                num_classes=self.conf.num_classes,     # number of classes
                lr=self.conf.learning_rate,
                loss=self.conf.loss_func,                             # CrossEntropyLoss
                # ignore_index=self.conf.ignore_index    # optional: class to ignore
            )
        elif self.conf.decoder_name == 'segformer':
            task = SegFormerLightningModule(
                num_classes=self.conf.num_classes,
                learning_rate=self.conf.learning_rate,
                in_channels=len(self.conf.output_bands)
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
    def modify_bands(
            self, xraster: xr.core.dataarray.DataArray, input_bands: List[str],
            output_bands: List[str], drop_bands: List[str] = []):
        """
        Drop multiple bands to existing rasterio object
        """
        # Drop any bands from input that should not be on output
        for ind_id in list(set(input_bands) - set(output_bands)):
            drop_bands.append(input_bands.index(ind_id)+1)

        # print(drop_bands)

        if isinstance(xraster, (np.ndarray, np.generic)):
            # Do not modify if image has the same number of output bands
            if xraster.shape[-1] == len(output_bands):
                return xraster
            elif xraster.shape[-1] == 4:
                return xraster
            xraster = np.delete(
                xraster[:][:], [x - 1 for x in drop_bands], axis=0)
            return xraster
        else:
            # print("DROP BANDS",drop_bands )
            # Do not modify if image has the same number of output bands
            if xraster['band'].shape[0] == len(output_bands):
                return xraster
            elif xraster['band'].shape[0] == 4:
                return xraster
            return xraster.drop(dim="band", labels=drop_bands)


    # -------------------------------------------------------------------------
    # predict
    # -------------------------------------------------------------------------
    def predict(self, force_cleanup: bool = False) -> None:

        logging.info('Starting prediction stage')

        # if the model is not specified, take the last one
        if self.conf.model_filename is not None:
            latest_model_file = self.conf.model_filename
        else:
            latest_model_file = max(
                (glob(os.path.join(self.conf.model_dir, 'epoch*.ckpt'))),
                key=os.path.getmtime
            )

        # task to predict with the latest model
        task = SemanticSegmentationTask.load_from_checkpoint(
            latest_model_file)

        # Gather filenames to predict
        if len(self.conf.inference_regex_list) > 0:
            data_filenames = self.get_filenames(
                self.conf.inference_regex_list)
        else:
            data_filenames = self.get_filenames(self.conf.inference_regex)
        logging.info(f'{len(data_filenames)} files to predict')

        # iterate files, create lock file to avoid predicting the same file
        for filename in sorted(data_filenames):

            # start timer
            start_time = time.time()

            output_directory = self.conf.inference_save_dir
            os.makedirs(output_directory, exist_ok=True)

            # set prediction output filename
            output_filename = os.path.join(
                output_directory,
                f'{Path(filename).stem}.{self.conf.experiment_type}.tif')

            # lock file for multi-node, multi-processing
            lock_filename = f'{output_filename}.lock'

            # delete lock file and overwrite prediction if force_cleanup
            logging.warning(
                'You have selected to force cleanup files. ' +
                'This option disables lock file tracking, which' +
                'Could lead to processing the same file multiple times.'
            )
            if force_cleanup and os.path.isfile(lock_filename):
                try:
                    os.remove(lock_filename)
                except FileNotFoundError:
                    logging.info(f'Lock file not found {lock_filename}')
                    continue

            # predict only if file does not exist and no lock file
            if not os.path.isfile(output_filename) and \
                    not os.path.isfile(lock_filename):

                try:

                    logging.info(f'Starting to predict {filename}')

                    # create lock file
                    open(lock_filename, 'w').close()

                    # open filename
                    image = rxr.open_rasterio(filename)
                    logging.info(f'Prediction shape: {image.shape}')

                except rasterio.errors.RasterioIOError:
                    logging.info(f'Skipped {filename}, probably corrupted.')
                    continue

                # Modify the bands to match inference details
                logging.info('Modifying bands')
                image = self.modify_bands(
                    xraster=image, input_bands=self.conf.input_bands,
                    output_bands=self.conf.output_bands)
                logging.info(f'Prediction shape after modf: {image.shape}')

                # add DTM if necessary
                if 'dtm' in list(map(str.lower, self.conf.output_bands)):
                    logging.info('Adding DTM layer')
                    image = self.add_dtm(filename, image, self.conf.dtm_path)
                    logging.info(f'Prediction shape after modf: {image.shape}')

                # Transpose the image for channel last format
                image = image.transpose("y", "x", "band")

                # Remove no-data values to account for edge effects
                # TODO: consider replacing this with the new parameters
                # for better padding.
                temporary_tif = xr.where(image > -100, image, 2000)

                # TODO: This needs some comments explaining what each of these is doing
                # Sliding window prediction
                # prediction = regression_inference.sliding_window_tiler(
                #    xraster=temporary_tif,
                #    model=task.model, # PMM edit model --> task.model
                #    n_classes=self.conf.n_classes,
                #    overlap=self.conf.inference_overlap,
                #    batch_size=self.conf.pred_batch_size,
                #    standardization=self.conf.standardization,
                #    mean=self.conf.preprocessing_mean_vector,
                #    std=self.conf.preprocessing_std_vector,
                #    normalize=self.conf.normalize,
                #    window=self.conf.window_algorithm
                # ) * self.conf.normalize_label

                temporary_tif = temporary_tif / 10000

                prediction = sliding_window_tiler_multiclass(
                    image,
                    model=task.model,
                    n_classes=self.conf.num_classes,
                    img_size=self.conf.tile_size,
                    standardization=None,#self.conf.standardization,
                    batch_size=self.conf.batch_size,
                    mean=None,#self.conf.preprocessing_mean_vector,
                    std=None,#self.conf.preprocessing_std_vector
                )

                # Set to 0 for negative predictions
                prediction[prediction < 0] = 0
                prediction = np.argmax(prediction, axis=-1)

                # Drop image band to allow for a merge of mask
                image = image.drop(
                    dim="band",
                    labels=image.coords["band"].values[1:],
                )

                # Get metadata to save raster
                prediction = xr.DataArray(
                    np.expand_dims(prediction, axis=-1),
                    name=self.conf.experiment_type,
                    coords=image.coords,
                    dims=image.dims,
                    attrs=image.attrs
                )

                # Add metadata to raster attributes
                prediction.attrs['long_name'] = (self.conf.experiment_type)
                # prediction.attrs['model_name'] = (self.conf.model_filename)
                prediction = prediction.transpose("band", "y", "x")

                # Add cloudmask to the prediction
                # TODO: cloudmasking seemingly not working? unclear b/c not 
                # all cloudmasks are in specified cloudmask dir - confusing
                if self.conf.cloudmask_path is not None:

                    # get the corresponding file that matches the
                    # cloudmask regex
                    cloudmask_filename = self.get_filenames(
                        os.path.join(
                            self.conf.cloudmask_path,
                            f'{Path(filename).stem.split("-")[0]}*.tif'
                        ),
                        allow_empty=True # TODO: check this; this was causing an error at one point 
                    )

                    # if we found cloud mask filename, proceed
                    if len(cloudmask_filename) > 0:
                        prediction = self.add_cloudmask(
                            prediction, cloudmask_filename[0])
                    else:
                        logging.info(
                            'No cloud mask filename found, ' +
                            'skipping cloud mask step.')

                # Set nodata values on mask
                nodata = prediction.rio.nodata
                prediction = prediction.where(image != nodata)
                prediction.rio.write_nodata(
                    self.conf.prediction_nodata, encoded=True, inplace=True)

                # Save output raster file to disk
                prediction.rio.to_raster(
                    output_filename,
                    BIGTIFF="IF_SAFER",
                    compress=self.conf.prediction_compress,
                    driver=self.conf.prediction_driver,
                    dtype=self.conf.prediction_dtype
                )
                del prediction

                # delete lock file
                try:
                    os.remove(lock_filename)
                except FileNotFoundError:
                    logging.info(f'Lock file not found {lock_filename}')
                    continue

                logging.info(f'Finished processing {output_filename}')
                logging.info(f"{(time.time() - start_time)/60} min")

            # This is the case where the prediction was already saved
            else:
                logging.info(f'{output_filename} already predicted.')

        return