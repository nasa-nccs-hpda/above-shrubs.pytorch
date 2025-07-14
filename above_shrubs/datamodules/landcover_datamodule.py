import torch
import pytorch_lightning as pl
from torchgeo.datamodules import NonGeoDataModule
from above_shrubs.utils.transform_utils import basic_augmentations
from above_shrubs.datasets.landcover_dataset import LandCoverSegmentationDataset


class LandCoverSegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        tile_size: int = 256,
        batch_size: int = 8,
        num_workers: int = 4,
        nodata_value: float = 0,
        nodata_threshold: float = 0.5,
        train_transforms=None,
        val_transforms=None,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.nodata_value = nodata_value
        self.nodata_threshold = nodata_threshold
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

    def setup(self, stage=None):
        self.train_dataset = LandCoverSegmentationDataset(
            csv_path=self.csv_path,
            tile_size=self.tile_size,
            transforms=self.train_transforms,
            nodata_threshold=self.nodata_threshold,
        )
        self.val_dataset = LandCoverSegmentationDataset(
            csv_path=self.csv_path,
            tile_size=self.tile_size,
            transforms=self.val_transforms,
            nodata_threshold=self.nodata_threshold,
        )
        self.test_dataset = LandCoverSegmentationDataset(
            csv_path=self.csv_path,
            tile_size=self.tile_size,
            transforms=None,
            nodata_threshold=self.nodata_threshold,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == '__main__':

    dm = LandCoverSegmentationDataModule(
        csv_path="/explore/nobackup/people/jacaraba/development/above-shrubs.pytorch/projects/landcover/configs/landcover_v2.0.csv",
        tile_size=256,
        batch_size=4,
        num_workers=0,
        nodata_threshold=0.0,
        train_transforms=None,
    )

    dm.setup()

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    x, y = batch
    print("Image shape:", x.shape)
    print("Label shape:", y.shape)
