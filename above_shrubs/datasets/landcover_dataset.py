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
        num_classes: int = 10
    ):
        super().__init__()

        self.df = pd.read_csv(csv_path)
        logging.info(f'Found {self.df.shape[0]} files for training.')

        # Validate file paths immediately
        for i, row in self.df.iterrows():
            if not Path(row["data"]).is_file():
                raise FileNotFoundError(f"Data file not found: {row['data']}")
            if not Path(row["label"]).is_file():
                raise FileNotFoundError(f"Label file not found: {row['label']}")

        self.tile_size = tile_size
        self.transforms = transforms
        self.nodata_threshold = nodata_threshold
        self.num_classes = num_classes

        # Save paths instead of open datasets
        self.image_paths = self.df["data"].tolist()
        self.label_paths = self.df["label"].tolist()
        self.ntiles_per_raster = self.df["ntiles"].astype(int).tolist()

        self.cumulative_counts = np.cumsum(self.ntiles_per_raster)
        logging.info(f'Getting ready to train with {self.cumulative_counts[-1]} tiles.')

    def __len__(self):
        return int(self.cumulative_counts[-1])

    def __getitem__(self, idx):
        raster_idx = np.searchsorted(self.cumulative_counts, idx, side="right")
        img_path = self.image_paths[raster_idx]
        lbl_path = self.label_paths[raster_idx]

        # Open inside worker
        with rasterio.open(img_path) as img_ds, rasterio.open(lbl_path) as lbl_ds:
            max_x = img_ds.width - self.tile_size
            max_y = img_ds.height - self.tile_size

            while True:
                x0 = np.random.randint(0, max_x)
                y0 = np.random.randint(0, max_y)
                window = rasterio.windows.Window(x0, y0, self.tile_size, self.tile_size)

                image = img_ds.read(window=window).astype(np.float32)
                label = lbl_ds.read(1, window=window).astype(np.int64)

                if image.shape[0] > 4:
                    image = image[:4, :, :]

                image /= 10000.0

                img_nodata = np.any(image < 0, axis=0)
                lbl_nodata = (label > 200)
                combined_nodata = np.logical_or(img_nodata, lbl_nodata)

                # Subtract 1 if not Clustered
                if not Path(lbl_path).name.startswith("Clustered"):
                    label -= 1

                if combined_nodata.mean() <= self.nodata_threshold and label.max() < self.num_classes:
                    assert label.min() >= 0, f"Negative labels found in {lbl_path}: {np.unique(label)}"
                    image = torch.from_numpy(image)
                    label = torch.from_numpy(label)

                    if self.transforms:
                        image, label = self.transforms(image, label)

                    return {"image": image, "mask": label}
