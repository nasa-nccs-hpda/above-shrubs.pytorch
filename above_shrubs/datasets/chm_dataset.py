import os
import sys
import torch
import numpy as np
import rioxarray as rxr
from typing import Any
from pathlib import Path
from typing import List
from torchgeo.datasets import NonGeoDataset

__status__ = "Production"


class CHMDataset(NonGeoDataset):
    """
    CHM Regression dataset from NonGeoDataset.
    """

    # n_images: default is -1 for all, set a value
    # depending on the number of images you are interested
    # in using for training

    def __init__(
        self,
        image_paths: list,
        mask_paths: list,
        transform=None,
        transform_labels=None,
        n_images: int = -1,
        band_indices: List[int] = []
    ) -> None:
        super().__init__()

        # transform
        self.transform = transform
        self.transform_labels = transform_labels

        # images and labels path
        if isinstance(image_paths, str):
            self.image_paths = [image_paths]
        else:
            self.image_paths = image_paths

        if isinstance(mask_paths, str):
            self.mask_paths = [mask_paths]
        else:
            self.mask_paths = mask_paths

        # images and labels list
        self.image_list = []
        self.mask_list = []

        # get filename paths
        for image_path, mask_path in zip(self.image_paths, self.mask_paths):
            self.image_list.extend(self.get_filenames(image_path))
            self.mask_list.extend(self.get_filenames(mask_path))

        # get first n_images if needed
        if n_images != -1:
            self.image_list = self.image_list[:n_images]
            self.mask_list = self.mask_list[:n_images]

        # rgb indices for some plots
        self.rgb_indices = [0, 1, 2]
        self.band_indices = band_indices

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int) -> dict[str, Any]:

        image = self._load_file(self.image_list[index])

        # select bands if needed
        if len(self.band_indices) > 0:

            # select particular bands from the training tile
            image = image[self.band_indices]

        image = torch.from_numpy(
            image.astype(np.float32))
        label = torch.from_numpy(self._load_file(
            self.mask_list[index]).astype(np.float32))

        if self.transform is not None:
            image = self.transform(image)
        
        if self.transform_labels is not None:
            label = self.transform_labels(label)

        return {'image': image, 'mask': label}

    def _load_file(self, path: Path):

        # loading the file
        if Path(path).suffix == '.npy':
            data = np.load(path)
        elif Path(path).suffix == '.tif':
            data = rxr.open_rasterio(path).to_numpy()
        else:
            sys.exit('Non-recognized dataset format. Expects npy or tif.')
        return data

    def get_filenames(self, path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = []
        for filename in sorted(os.listdir(path)):
            files_list.append(os.path.join(path, filename))
        return files_list
