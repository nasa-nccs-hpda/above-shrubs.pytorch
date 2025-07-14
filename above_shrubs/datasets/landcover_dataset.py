import torch
import logging
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
from torchgeo.datasets import NonGeoDataset


class LandCoverSegmentationDataset(NonGeoDataset):
    def __init__(
        self,
        csv_path: str,
        tile_size: int = 256,
        transforms=None,
        nodata_threshold: float = 0.5,
    ):
        """
        Args:
            csv_path (str): CSV with 'data','label','ntiles'.
            tile_size (int): Tile size.
            transforms: Optional (image,label) transforms.
            nodata_threshold (float): Fraction of nodata pixels allowed.
        """
        super().__init__()

        self.df = pd.read_csv(csv_path)
        logging.info(f'Found {self.df.shape[0]} files for training.')

        # Validate file paths immediately
        for i, row in self.df.iterrows():
            data_path = Path(row["data"])
            label_path = Path(row["label"])
            if not data_path.is_file():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            if not label_path.is_file():
                raise FileNotFoundError(f"Label file not found: {label_path}")

        # set some default values
        self.tile_size = tile_size
        self.transforms = transforms
        self.nodata_threshold = nodata_threshold

        # get images and labels
        self.images = []
        self.labels = []
        self.ntiles_per_raster = []

        for _, row in self.df.iterrows():

            logging.info(f'Setting metadata for {Path(row["data"]).stem}')
            img_ds = rasterio.open(row["data"])
            lbl_ds = rasterio.open(row["label"])
            self.images.append(img_ds)
            self.labels.append(lbl_ds)
            self.ntiles_per_raster.append(int(row["ntiles"]))

        # Cumulative lengths for indexing
        self.cumulative_counts = np.cumsum(self.ntiles_per_raster)
        logging.info(f'Getting ready to train with {self.cumulative_counts} tiles.')

    def __len__(self):
        return int(self.cumulative_counts[-1])

    def __getitem__(self, idx):

        raster_idx = np.searchsorted(self.cumulative_counts, idx, side="right")
        img_ds = self.images[raster_idx]
        lbl_ds = self.labels[raster_idx]

        max_x = img_ds.width - self.tile_size
        max_y = img_ds.height - self.tile_size

        while True:

            x0 = np.random.randint(0, max_x)
            y0 = np.random.randint(0, max_y)

            window = rasterio.windows.Window(x0, y0, self.tile_size, self.tile_size)

            image = img_ds.read(window=window).astype(np.float32)
            label = lbl_ds.read(1, window=window).astype(np.int64)

            # If more than 4 bands, keep only the first 4
            if image.shape[0] > 4:
                image = image[:4, :, :]

            # Normalize
            image /= 10000.0

            # Nodata mask: any negative value in any band
            img_nodata = np.any(image < 0, axis=0)
            lbl_nodata = (label > 200)
            combined_nodata = np.logical_or(img_nodata, lbl_nodata)

            #print(
            #    f"Nodata fraction: {combined_nodata.mean():.4f}, "
            #    f"Negative pixels: {img_nodata.sum()}, "
            #    f"High label pixels: {lbl_nodata.sum()}"
            #)


            if combined_nodata.mean() <= self.nodata_threshold:
                image = torch.from_numpy(image)
                label = torch.from_numpy(label)
                #if self.transforms:
                #    image, label = self.transforms(image, label)
                print(image.shape, label.shape)
                return image, label
