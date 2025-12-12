import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import segmentation_models_pytorch as smp


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        probs = torch.softmax(logits, dim=1)
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score.mean()

class SwinUNetLightningModule(pl.LightningModule):
    def __init__(self, num_classes: int, learning_rate: float, in_channels: int = 4, dice_weight: float = 0.5):
        super().__init__()
        self.save_hyperparameters()

        # Swin Transformer UNet
        self.model = smp.Unet(
            encoder_name="mit_b5",
            encoder_weights="imagenet",
            in_channels=in_channels,   # <- THIS IS YOUR 4 BANDS
            classes=num_classes
        )

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"].long()

        logits = self(images)

        ce = self.ce_loss(logits, masks)
        dice = self.dice_loss(logits, masks)
        loss = self.dice_weight * dice + (1 - self.dice_weight) * ce

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"].long()

        logits = self(images)

        ce = self.ce_loss(logits, masks)
        dice = self.dice_loss(logits, masks)
        loss = self.dice_weight * dice + (1 - self.dice_weight) * ce

        self.log("val_loss", loss, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
