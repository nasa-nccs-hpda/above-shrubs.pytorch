import torch
from typing import Any
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule
from above_shrubs.datasets.chm_dataset import CHMDataset

__status__ = "Production"


class CHMDataModule(NonGeoDataModule):
    """NonGeo CHM data module implementation"""

    def __init__(
                self,
                train_data_dir: str,
                train_label_dir: str,
                test_data_dir: str = None,
                test_label_dir: str = None,
                batch_size: int = 16,
                num_workers: int = 8,
                transform_images=None,
                transform_labels=None,
                n_images: int = -1,
                prefetch_factor: int = 4,
                **kwargs: Any
            ) -> None:

        super().__init__(CHMDataset, batch_size, num_workers, **kwargs)

        # Ensure dataset_class is defined
        self.dataset_class = CHMDataset

        # training paths
        self.train_data_dir = train_data_dir
        self.train_label_dir = train_label_dir

        # test paths
        self.test_data_dir = test_data_dir
        self.test_label_dir = test_label_dir

        # transforms and number of images to use
        self.transform_images = transform_images
        self.transform_labels = transform_labels
        self.n_images = n_images
        self.prefetch_factor = prefetch_factor

    def setup(self, stage: str = None) -> None:
        if stage in ["fit"] or stage is None:
            self.train_dataset = self.dataset_class(
                self.train_data_dir,
                self.train_label_dir,
                self.transform_images,
                self.transform_labels,
                self.n_images
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                self.test_data_dir,
                self.test_label_dir,
                self.transform_images,
                self.transform_labels,
                self.n_images
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                self.test_data_dir,
                self.test_label_dir,
                self.transform_images,
                self.transform_labels,
                self.n_images
            )

    def prepare_data(self):
        # This method can be empty if no downloading
        # or data preparation is required
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor
        )


def collate_fn(inputs):
    batch = dict()
    pixel_values = torch.stack(
        [torch.tensor(i[0]).float() for i in inputs], dim=0)
    labels = torch.stack([torch.tensor(i[1]).long() for i in inputs], dim=0)
    original_images = [torch.tensor(i[2]).float() for i in inputs]
    original_segmentation_maps = [torch.tensor(i[3]).long() for i in inputs]

    # Uncomment this if your pixel_values are in
    # (batch_size, height, width, channels) format
    # pixel_values = pixel_values.permute(0, 3, 1, 2)

    batch["pixel_values"] = pixel_values
    batch["labels"] = labels
    batch["original_images"] = original_images
    batch["original_segmentation_maps"] = original_segmentation_maps

    return batch
