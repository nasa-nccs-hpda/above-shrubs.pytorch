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

from above_shrubs.config import CHMConfig as Config

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

CHUNKS = {'band': 'auto', 'x': 'auto', 'y': 'auto'}
xp = np

__status__ = "Development"


class CHMPipeline(BasePipeline):

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

        # train data directories
        self.train_data_dir = os.path.join(
            self.conf.train_tiles_dir, 'images')
        self.train_label_dir = os.path.join(
            self.conf.train_tiles_dir, 'labels')

        # test data directories
        self.test_data_dir = os.path.join(
            self.conf.test_tiles_dir, 'images')
        self.test_label_dir = os.path.join(
            self.conf.test_tiles_dir, 'labels')

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
            logging.info(
                f'Training tiles saved at {self.conf.train_tiles_dir}')

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
            logging.info(f'Test tiles saved at {self.conf.test_tiles_dir}')
        return

    # -------------------------------------------------------------------------
    # preprocess
    # -------------------------------------------------------------------------
    def preprocess(self):
        """
        Perform general preprocessing.
        """
        logging.info('Starting preprocessing stage')

        # Set output directories
        self._set_train_test_dirs()

        # Calculate mean and std values for training
        data_filenames = self.get_dataset_filenames(
            self.train_data_dir, ext='*.npy')
        logging.info(f'Mean and std values from {len(data_filenames)} files.')

        # Temporarily disable standardization and augmentation
        current_standardization = self.conf.standardization
        self.conf.standardization = None

        # Set main data loader
        chm_train_dataset = CHMDataset(
            self.train_data_dir,
            self.train_label_dir
        )

        # Set data loader to iterate over the tiles
        # Some elements here are fixed based on benchmarking
        # experiments done in the past
        train_dataloader = DataLoader(
            chm_train_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
            prefetch_factor=4
        )

        # Get mean and std array
        mean, std = self.get_mean_std_dataset(
            train_dataloader, self.conf.metadata_filename)
        logging.info(f'Mean: {mean.numpy()}, Std: {std.numpy()}')

        # Re-enable standardization for next pipeline step
        self.conf.standardization = current_standardization

        logging.info('Done with preprocessing stage')

        return

    # -------------------------------------------------------------------------
    # train
    # -------------------------------------------------------------------------
    def train(self):
        """
        Perform general training.
        """
        logging.info('Starting training stage')

        # Set working directories for training
        self._set_train_test_dirs()

        # Setup transforms
        transform_images, transform_labels = self.get_transforms(
            self.conf.backbone_name)

        # Load data module
        datamodule = CHMDataModule(
            train_data_dir=self.train_data_dir,
            train_label_dir=self.train_label_dir,
            test_data_dir=self.test_data_dir,
            test_label_dir=self.test_label_dir,
            batch_size=self.conf.batch_size,
            num_workers=multiprocessing.cpu_count(),
            transform_images=transform_images,
            transform_labels=transform_labels,
            n_images=self.conf.num_train_images
        )
        logging.info('Loaded data module.')

        # Set available accelerator to use
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

        # Set callbacks
        callbacks = get_callbacks(self.conf.callbacks)

        # Set logger
        loggers = get_loggers(self.conf.loggers)

        # Create a task to train with
        # MetaDinoV2RS_Lightning
        task = CHMPixelwiseRegressionTask(
            loss=self.conf.loss_func,
            model=self.conf.decoder_name,  # 'fcn' 'deeplabv3+',
            backbone=self.conf.backbone_name,
            weights=self.conf.weights,
            in_channels=len(self.conf.output_bands),
            num_outputs=1,  # since pixelwise, we get a single layer of output
            lr=self.conf.learning_rate,
            patience=self.conf.patience
        )

        # Set trainer
        trainer = Trainer(
            accelerator=accelerator,
            callbacks=callbacks,
            fast_dev_run=self.conf.fast_dev_run,
            log_every_n_steps=1,
            logger=loggers,
            min_epochs=1,
            max_epochs=self.conf.num_epochs,
            strategy="ddp_find_unused_parameters_true"
        )

        # train our model
        trainer.fit(model=task, datamodule=datamodule)

        # test our model
        trainer.test(model=task, datamodule=datamodule)

        return

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
        task = CHMPixelwiseRegressionTask.load_from_checkpoint(
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
                prediction = sliding_window_tiler_multiclass(
                    image,
                    model=task.model,
                    n_classes=1,
                    img_size=224,
                    standardization=self.conf.standardization,
                    batch_size=self.conf.batch_size,
                    mean=self.conf.preprocessing_mean_vector,
                    std=self.conf.preprocessing_std_vector
                )

                # Set to 0 for negative predictions
                prediction[prediction < 0] = 0

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

    # -------------------------------------------------------------------------
    # validate
    # -------------------------------------------------------------------------
    def validate(self):
        raise NotImplementedError("This function is not implemented yet.")

    # -------------------------------------------------------------------------
    # add_dtm
    # -------------------------------------------------------------------------
    def add_dtm(self, raster_filename: str, raster, dtm_filename: str):

        # warp dtm to match raster
        warp_ds_list = warplib.memwarp_multi_fn(
            [raster_filename, dtm_filename], res=raster_filename,
            extent=raster_filename, t_srs=raster_filename, r='average',
            dst_ndv=int(raster.rio.nodata), verbose=False
        )
        dtm_ma = iolib.ds_getma(warp_ds_list[1])

        # Drop image band to allow for a merge of mask
        dtm = raster.drop(
            dim="band",
            labels=raster.coords["band"].values[1:],
        )
        dtm.coords['band'] = [raster.shape[0] + 1]

        # Get metadata to save raster
        dtm = xr.DataArray(
            np.expand_dims(dtm_ma, axis=0),
            name='dtm',
            coords=dtm.coords,
            dims=dtm.dims,
            attrs=dtm.attrs
        ).fillna(raster.rio.nodata)
        dtm = dtm.where(raster[0, :, :] > 0, int(raster.rio.nodata))

        # concatenate the bands together
        dtm = xr.concat([raster, dtm], dim="band")
        dtm = dtm.where(dtm > 0, int(raster.rio.nodata))

        # additional clean-up for the imagery
        dtm.where(
            dtm.any(dim='band') != True,  # noqa: E712
            int(raster.rio.nodata)
        )
        dtm = dtm.where(dtm > 0, int(raster.rio.nodata))
        dtm.attrs['long_name'] = dtm.attrs['long_name'] + ("DTM",)
        return dtm

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

    # -------------------------------------------------------------------------
    # get_transforms
    # -------------------------------------------------------------------------
    def get_transforms(self, model_name: str):

        if model_name == 'dinov2_rs':
            # Setup transforms, currently fixed, will need them to be dynamic
            # Check on how to do this later
            # Ensure compatibility with MetaDinoV2RS, but what about the other
            # models we will need to support
            images_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=self.conf.preprocessing_mean_vector,
                    std=self.conf.preprocessing_std_vector)  # Normalize
            ])
            # Ensure compatibility with MetaDinoV2RS
            labels_transform = transforms.Compose([
                transforms.Resize((224, 224)),
            ])
        else:
            images_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=self.conf.preprocessing_mean_vector,
                    std=self.conf.preprocessing_std_vector)  # Normalize
            ])
            # Ensure compatibility with MetaDinoV2RS
            labels_transform = transforms.Compose([
                transforms.Resize((224, 224)),
            ])
        return images_transform, labels_transform
