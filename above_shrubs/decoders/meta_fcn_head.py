import torch
import torch.nn as nn
import torch.nn.functional as F

from above_shrubs.encoders.metachm_dino_encoder import SSLVisionTransformer


class SimpleFCNDecoder(nn.Module):
    def __init__(self, in_channels=(1280, 1280, 1280, 1280), output_range=30):
        super(SimpleFCNDecoder, self).__init__()

        self.output_range = output_range  # Maximum height value (30 meters)

        self.conv1 = nn.Conv2d(in_channels[-1], 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=1)  # 1 output channel for regression

    def forward(self, features):
        """
        Features come from SSLVisionTransformer (list of feature maps).
        We only use the last feature map (highest resolution).
        """
        x = features[-1]  # Use the last feature map from the transformer

        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = F.relu(self.conv3(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = F.relu(self.conv4(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = self.conv5(x)  # Final linear layer (no activation)
        x = torch.relu(x) * self.output_range  # Ensure output is in range [0, 30]

        return x

class MetaDinoV2RSFCN(nn.Module):
    def __init__(self, pretrained=None, huge=False, input_bands=3):
        super().__init__()
        if huge == True:

            self.backbone = SSLVisionTransformer(
                embed_dim=1280,
                num_heads=20,
                out_indices=(9, 16, 22, 29),
                depth=32,
                pretrained=pretrained
            )

            if input_bands > 3:

                # Get the original Conv2d layer
                old_conv = self.backbone.patch_embed.proj

                # Create a new Conv2d layer with in_channels=4
                new_conv = nn.Conv2d(
                    in_channels=input_bands,  # Change from 3 → 4
                    out_channels=old_conv.out_channels,  # Keep output channels the same
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )

                # Copy weights from the old Conv2d
                with torch.no_grad():
                    new_conv.weight[:, :3, :, :] = old_conv.weight  # Copy existing 3-channel weights
                    new_conv.weight[:, 3:, :, :] = old_conv.weight[:, :1, :, :]  # Duplicate first channel for the 4th band

                # Replace the old projection layer
                self.backbone.patch_embed.proj = new_conv

            # self.decode_head = RPTHead(
            #    in_channels=(1280, 1280, 1280, 1280),
            #    embed_dims=1280,
            #    post_process_channels=[160, 320, 640, 1280],
            # )
            # self.decode_head = DPTHead(
            #    in_channels=(1280, 1280, 1280, 1280),
            #    embed_dims=1280,
            #    post_process_channels=[160, 320, 640, 1280],
            #    classify=False
            # )
            self.decode_head = SimpleFCNDecoder(
                in_channels=(1280, 1280, 1280, 1280), output_range=30
            )
        else:
            self.backbone = SSLVisionTransformer(init_cfg=pretrained)
            self.decode_head = RPTHead()

    def forward(self, x):
        x = self.backbone(x)
        x = self.decode_head(x)
        return x
