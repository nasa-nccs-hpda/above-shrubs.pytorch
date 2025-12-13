import os
import torch
import lightning
import segmentation_models_pytorch as smp
from torchgeo.models import FCN, get_weight
from torchvision.models._api import WeightsEnum
from torchgeo.trainers import PixelwiseRegressionTask
from torchgeo.trainers.utils import extract_backbone
from above_shrubs.decoders.meta_rpt_head import MetaDinoV2RS
from above_shrubs.decoders.meta_fcn_head import MetaDinoV2RSFCN
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from above_shrubs.decoders.custom_dpt_head import DINOv3DepthHead

class CHMPixelwiseRegressionTask(PixelwiseRegressionTask):

    def configure_models(self) -> None:
        """Initialize the model."""
        weights = self.weights

        if self.hparams['model'] == 'unet':
            self.model = smp.Unet(
                encoder_name=self.hparams['backbone'],
                encoder_weights='imagenet' if weights is True else None,
                in_channels=self.hparams['in_channels'],
                classes=1,
            )
        elif self.hparams['model'] == 'deeplabv3+':
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hparams['backbone'],
                encoder_weights='imagenet' if weights is True else None,
                in_channels=self.hparams['in_channels'],
                classes=1,
            )
        elif self.hparams['model'] == 'fcn':
            self.model = FCN(
                in_channels=self.hparams['in_channels'],
                classes=1,
                num_filters=self.hparams['num_filters'],
            )
        elif self.hparams['model'] == 'dinov2_rs' \
                or self.hparams['model'] == 'dinov2_rs_rpt':
            self.model = MetaDinoV2RS(
                pretrained=weights,
                huge=True,
                input_bands=self.hparams['in_channels']
            )
        elif self.hparams['model'] == 'dinov2_fcn_rs':
            self.model = MetaDinoV2RSFCN(
                pretrained=weights,
                huge=True,
                input_bands=self.hparams['in_channels']
            )
        else:
            raise ValueError(
                f"Model type '{self.hparams['model']}' is not valid. " +
                "Currently, only supports 'unet', 'deeplabv3+', " +
                " 'dinov2_rs', 'dinov2_rs_rpt', and 'fcn'."
            )

        if self.hparams['model'] != 'fcn' \
                and self.hparams['model'] != 'dinov2_rs' \
                and self.hparams['model'] != 'dinov2_rs_rpt' \
                and self.hparams['model'] != 'dinov2_fcn_rs':
            if weights and weights is not True:
                if isinstance(weights, WeightsEnum):
                    state_dict = weights.get_state_dict(progress=True)
                elif os.path.exists(weights):
                    _, state_dict = extract_backbone(weights)
                else:
                    state_dict = get_weight(weights).get_state_dict(
                        progress=True)
                self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone
        if self.hparams.get('freeze_backbone', False) \
            and self.hparams['model'] in [
            'unet',
            'deeplabv3+',
        ]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams.get('freeze_decoder', False) \
                and self.hparams['model'] in [
            'unet',
            'deeplabv3+',
        ]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    def configure_optimizers(
        self,
    ) -> 'lightning.pytorch.utilities.types.OptimizerLRScheduler':
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        optimizer = AdamW(self.parameters(), lr=self.hparams['lr'])
        # scheduler = ReduceLROnPlateau(optimizer, patience=self.hparams['patience'])

        def warmup(epoch):
            if epoch < 5:  # Warmup for first 5 epochs
                return epoch / 5
            return 1

        scheduler = LambdaLR(optimizer, warmup)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': self.monitor},
        }


def load_flexible_model(checkpoint_path, device='cuda', input_bands='nrg'):
    """Load a model checkpoint regardless of which model type it is"""
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint to check model type
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = checkpoint['model_config']  # This line was missing!
    
    print(f"Loading {model_config['description']}")
    print(f"  Model name: {model_config['model_name']}")
    print(f"  Embedding dim: {model_config['embed_dim']}")
    print(f"  Total tokens: {model_config['total_tokens']}")

    # Create model with correct configuration
    token = os.getenv("HF_TOKEN")
    if token is None or token.strip() == "":
        raise RuntimeError(
            "Hugging Face token not found.\n"
            "Please set the environment variable HF_TOKEN "
            "before loading the model.\n\n"
            "Example:\n"
            "  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx\n"
        )
    model = DINOv3DepthHead(
        model_name=model_config['model_name'],
        freeze_backbone=True,
        token=token,
        input_bands=input_bands
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"  Best epoch: {checkpoint['epoch']}")
    print(f"  Best validation loss: {checkpoint['val_loss']:.4f}")
    
    return model, checkpoint


if __name__ == "__main__":

    checkpoint_path = "/explore/nobackup/people/mfrost2/projects/dino_boreal_final/best_dinov3_fulldataset_customloss.pth"
    load_flexible_model(checkpoint_path, device='cuda', input_bands='nrg')
