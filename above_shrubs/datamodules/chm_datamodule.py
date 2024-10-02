import torch
import lightning as L
#import kornia.augmentation as K  # noqa: N812

from typing import Any
from torchvision import transforms
from torchgeo.datamodules import NonGeoDataModule
from torch.utils.data import random_split, DataLoader
from torchgeo.transforms import AugmentationSequential
from above_shrubs.datasets.chm_dataset import CHMDataset


MEANS = [325.04178, 518.01135, 393.07028, 2660.147, 343.5341]
STDS = [80.556175, 133.02502, 135.68076, 822.97205, 116.81135]


class CHMDataModule(NonGeoDataModule):
    """NonGeo Fire Scars data module implementation"""

    def __init__(
                self,
                train_data_dir: str,
                train_label_dir: str,
                test_data_dir: str = None,
                test_label_dir: str = None,
                batch_size: int = 16,
                num_workers: int = 8,
                tile_size: tuple = (224, 224),
                **kwargs: Any
            ) -> None:

        super().__init__(CHMDataset, batch_size, num_workers, **kwargs)
        # applied for training
        #self.train_aug = AugmentationSequential(
        #    K.Normalize(MEANS, STDS),
        #    K.RandomCrop(tile_size),
        #    data_keys=["image", "mask"],
        #)
        #self.aug = AugmentationSequential(
        #    K.Normalize(MEANS, STDS),
        #    data_keys=["image", "mask"]
        #)

        # tile size
        self.tile_size = tile_size

        # training paths
        self.train_data_dir = train_data_dir
        self.train_label_dir = train_label_dir

        # test paths
        self.test_data_dir = test_data_dir
        self.test_label_dir = test_label_dir

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                self.train_data_dir,
                self.train_label_dir,
                img_size=self.tile_size,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                self.test_data_dir,
                self.test_label_dir,
                img_size=self.tile_size,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                self.test_data_dir,
                self.test_label_dir,
                img_size=self.tile_size,
            )



"""

class CHMDataModule(L.LightningDataModule):

    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        CHMDataset(self.data_dir, train=True, download=True)
        CHMDataset(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)
"""