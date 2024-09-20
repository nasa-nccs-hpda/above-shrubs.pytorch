import os
import logging
import numpy as np
from itertools import repeat
from multiprocessing import Pool, cpu_count
from above_shrubs.pipelines.base_pipeline import BasePipeline

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
        data_filenames = self.get_dataset_filenames(self.train_images_dir)
        label_filenames = self.get_dataset_filenames(self.train_labels_dir)
        logging.info(f'Mean and std values from {len(data_filenames)} files.')

        # Temporarily disable standardization and augmentation
        current_standardization = self.conf.standardization
        self.conf.standardization = None
        metadata_output_filename = os.path.join(
            self.metadata_dir, f'mean-std-values.csv')

        # Set main data loader
        main_data_loader = RegressionDataLoaderSRLite(
            data_filenames, label_filenames, self.conf, False
        )


        """

        # Get mean and std array
        mean, std = get_mean_std_dataset(
            main_data_loader.train_dataset, metadata_output_filename)
        logging.info(f'Mean: {mean.numpy()}, Std: {std.numpy()}')

        # Re-enable standardization for next pipeline step
        self.conf.standardization = current_standardization

        logging.info('Done with preprocessing stage')
        """

