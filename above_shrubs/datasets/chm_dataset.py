import os
import sys
import numpy as np
import rioxarray as rxr
from typing import Any
from pathlib import Path
from torchgeo.datasets import NonGeoDataset


class CHMDataset(NonGeoDataset):
    """
    CHM Regression dataset from NonGeoDataset.
    """

    def __init__(
        self,
        image_paths: list,
        mask_paths: list,
        img_size: tuple = (256, 256),
        transform=None,
    ) -> None:
        super().__init__()

        # image size
        self.image_size = img_size

        # transform
        self.transform = transform

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

        # rgb indices for some plots
        self.rgb_indices = [0, 1, 2]

    def __len__(self) -> int:
        return len(self.image_list)

    # def __getitem__(self, idx, transpose=True):
    #
    #    # load image
    #    img = np.load(self.image_list[idx])
    #
    #    # load mask
    #    mask = np.load(self.mask_list[idx])
    #    # perform transformations
    #    if self.transform is not None:
    #        img = self.transform(img)
    #
    #    return img, mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        output = {
            "image": self._load_file(
                self.image_list[index]).astype(np.float32),
            "mask": self._load_file(
                self.mask_list[index]).astype(np.int64),
        }
        return output

    def _load_file(self, path: Path):
        if Path(path).suffix == '.npy':
            data = np.load(path)
        elif Path(path).suffix == '.tif':
            data = rxr.open_rasterio(path)
        else:
            sys.exit('Non-recognized dataset format. Expects npy or tif.')
        return data.to_numpy()

    def get_filenames(self, path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = []
        for filename in sorted(os.listdir(path)):
            files_list.append(os.path.join(path, filename))
        return files_list
