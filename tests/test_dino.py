
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from above_shrubs.encoders.metachm_dino_encoder import SSLVisionTransformer
from above_shrubs.datasets.chm_dataset import CHMDatasetTorch
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms

#model = torch.load('/explore/nobackup/projects/ilab/data/MetaCHM/saved_checkpoints/SSLhuge_satellite.pth')
#print(model)

weights_path = '/explore/nobackup/projects/ilab/data/MetaCHM/saved_checkpoints/SSLhuge_satellite.pth'

#tile = np.load('/explore/nobackup/people/mmacande/srlite/chm_model/20231014_chm/train_merged_nodtm_npy/images/chip_WV03_20190723_M1BS_10400100506ED100_CHM_fairbanks_ql2_2017_04295_v20231006.npy')
#print(tile.shape)


# Convert to PyTorch tensor and add a channel dimension (assuming single-channel data)
#tile = torch.tensor(tile).unsqueeze(0)  # Shape: (4, 1, 64, 64)

# Resize using interpolate
#tile = F.interpolate(tile, size=(224, 224), mode='bilinear', align_corners=False)
#print(tile.shape)

from above_shrubs.decoders.meta_rpt_head import MetaDinoV2RS


model = MetaDinoV2RS(pretrained=weights_path, huge=True, input_bands=4)
#model = model.eval()
#print("WHOLEU")

#dummy_input = torch.randn(1, 4, 224, 224)
#output = model(tile)

#print("POPOTE")

#output = output.squeeze()

#print(output.min(), output.max())

# Define transformations
means = [368.7239,547.4674,528.48615,2144.9368]
stds = [115.4657,157.63426,231.98418,1246.9503]

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure compatibility with MetaDinoV2RS
    transforms.Normalize(mean=means, std=stds)  # Normalize
])

transform_labels = transforms.Compose([
    #transforms.ToTensor(), 
    transforms.Resize((224, 224)),  # Ensure compatibility with MetaDinoV2RS
])

dataset = CHMDatasetTorch(
    image_paths='/explore/nobackup/people/mmacande/srlite/chm_model/20231014_chm/train_merged_nodtm_npy/images',
    mask_paths='/explore/nobackup/people/mmacande/srlite/chm_model/20231014_chm/train_merged_nodtm_npy/labels',
    transform=transform,
    transform_labels=transform_labels
)
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=40,         # Number of CPU workers for loading
    pin_memory=True,       # Enable pinned memory
    prefetch_factor=4
)

criterion = torch.nn.L1Loss() #MAE loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Wrap the model with DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    #torch.save(model.state_dict(), f"fine_tuned_metadino_{epoch}.pth")

# Save the fine-tuned model
torch.save(model.state_dict(), "/explore/nobackup/projects/ilab/scratch/jacaraba/above-shrubs/fine_tuned_metadino.pth")

print("HELLO")


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from above_shrubs.encoders.metachm_dino_encoder import SSLVisionTransformer
from above_shrubs.decoders.meta_rpt_head import MetaDinoV2RS
from above_shrubs.datasets.chm_dataset import CHMDatasetTorch

# Path to Pretrained Model
weights_path = '/explore/nobackup/projects/ilab/data/MetaCHM/saved_checkpoints/SSLhuge_satellite.pth'

# Define Mean and Std for Normalization
means = [368.7239, 547.4674, 528.48615, 2144.9368]
stds = [115.4657, 157.63426, 231.98418, 1246.9503]

# Define Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure compatibility with MetaDinoV2RS
    transforms.Normalize(mean=means, std=stds)  # Normalize
])

transform_labels = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize labels
])

# -------------------------
# PyTorch Lightning DataModule
# -------------------------
class CHMDataModule(pl.LightningDataModule):
    def __init__(self, image_dir, mask_dir, batch_size=256, num_workers=40):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """Load dataset for training."""
        self.train_dataset = CHMDatasetTorch(
            image_paths=self.image_dir,
            mask_paths=self.mask_dir,
            transform=transform,
            transform_labels=transform_labels
        )

    def train_dataloader(self):
        """Return DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=4  # Enable prefetching
        )

# -------------------------
# PyTorch Lightning Model
# -------------------------
class MetaDinoV2RSLightning(pl.LightningModule):
    def __init__(self, pretrained_path, input_bands=4, lr=1e-4, weight_decay=1e-5):
        super().__init__()
        self.model = MetaDinoV2RS(pretrained=pretrained_path, huge=True, input_bands=input_bands)
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

# -------------------------
# Training Setup
# -------------------------
if __name__ == "__main__":
    # Define DataModule
    data_module = CHMDataModule(
        image_dir='/explore/nobackup/people/mmacande/srlite/chm_model/20231014_chm/train_merged_nodtm_npy/images',
        mask_dir='/explore/nobackup/people/mmacande/srlite/chm_model/20231014_chm/train_merged_nodtm_npy/labels'
    )

    # Define Model
    model = MetaDinoV2RSLightning(pretrained_path=weights_path)

    # Define Trainer (Auto Multi-GPU)
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",  # Automatically use GPUs
        devices=-1,  # Use all available GPUs
        precision=16,  # Use mixed precision for speed
        log_every_n_steps=10,
        deterministic=True,
        enable_checkpointing=True
    )

    # Start Training
    trainer.fit(model, data_module)

    # Save Final Model
    model_path = "/explore/nobackup/projects/ilab/scratch/jacaraba/above-shrubs/fine_tuned_metadino.pth"
    trainer.save_checkpoint(model_path)
"""