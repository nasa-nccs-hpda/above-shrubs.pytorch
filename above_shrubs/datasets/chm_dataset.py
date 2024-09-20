import os
import numpy as np
from torch.utils.data import Dataset


class CHMDataset(Dataset):
    """
    CHM Regression dataset.
    """

    def __init__(
        self,
        image_paths: list,
        mask_paths: list,
        img_size: tuple = (256, 256),
        transform=None,
    ):

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

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx, transpose=True):

        # load image
        img = np.load(self.image_list[idx])

        # load mask
        mask = np.load(self.mask_list[idx])

        # perform transformations
        if self.transform is not None:
            img = self.transform(img)

        return img, mask

    def get_filenames(self, path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = []
        for filename in sorted(os.listdir(path)):
            files_list.append(os.path.join(path, filename))
        return files_list
