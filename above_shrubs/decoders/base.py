import warnings
import torch.nn as nn
import torch.nn.functional as F
from above_shrubs.encoders.base import Encoder
from torchgeo.trainers import PixelwiseRegressionTask

class Decoder(nn.Module):
    """Base class for decoders."""

    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
    ) -> None:
        """Initialize the decoder.

        Args:
            encoder (Encoder): encoder used.
            num_classes (int): number of classes of the task.
            finetune (bool): whether the encoder is finetuned.
        """
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.finetune = finetune

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')

    return F.interpolate(input, size, scale_factor, mode, align_corners)


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
        elif self.hparams['model'] == 'dinov2':
            self.model = MetaDinoV2RS(
                pretrained=weights,
                huge=True,
                input_bands=self.hparams['in_channels']
            )
        else:
            raise ValueError(
                f"Model type '{self.hparams['model']}' is not valid. "
                "Currently, only supports 'unet', 'deeplabv3+', 'dinov2' and 'fcn'."
            )

        if self.hparams['model'] != 'fcn' and self.hparams['model'] != 'dinov2':
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
        if self.hparams.get('freeze_backbone', False) and self.hparams['model'] in [
            'unet',
            'deeplabv3+',
        ]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams.get('freeze_decoder', False) and self.hparams['model'] in [
            'unet',
            'deeplabv3+',
        ]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False
