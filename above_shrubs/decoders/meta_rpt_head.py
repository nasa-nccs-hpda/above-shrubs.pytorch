# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from typing import Any
from above_shrubs.decoders.base import resize
from above_shrubs.encoders.metachm_dino_encoder import SSLVisionTransformer
from above_shrubs.decoders.meta_dpt_head import DPTHead
from torchvision.models._api import WeightsEnum
from torchgeo.trainers.utils import extract_backbone, load_state_dict
from torchgeo.trainers import PixelwiseRegressionTask


def kaiming_init(module: nn.Module,
                 a: float = 0,
                 mode: str = 'fan_out',
                 nonlinearity: str = 'relu',
                 bias: float = 0,
                 distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class ConvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 act_cfg=dict(type='ReLU'),
                 inplace=True):
        super().__init__()
        self.with_activation = act_cfg is not None
        self.with_bias = bias if bias != 'auto' else not self.with_activation
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=self.with_bias)
        
        if self.with_activation:
            self.activate = nn.ReLU()  # You can modify this for other activations
            
        self.init_weights()

    def init_weights(self):
        kaiming_init(self.conv)
    
    def forward(self, x):
        x = self.conv(x)
        if self.with_activation:
            x = self.activate(x)
        return x

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

class HeadDepth(nn.Module):
    def __init__(self, features):
        super(HeadDepth, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),  # Output 1 for regression
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.head(x)

class RPTHead(nn.Module):
    def __init__(self,
                 in_channels=(1024, 1024, 1024, 1024),
                 channels=256,
                 embed_dims=1024,
                 post_process_channels=[128, 256, 512, 1024],
                 expand_channels=False,
                 **kwargs):
        super(RPTHead, self).__init__(**kwargs)
        self.channels = channels
        self.expand_channels = expand_channels
        self.reassemble_blocks = nn.ModuleList()

        # Create ReassembleBlocks according to input dimensions
        for embed_dim in in_channels:
            self.reassemble_blocks.append(
                ConvModule(embed_dim, embed_dim, kernel_size=1, act_cfg=None)
            )

        self.convs = nn.ModuleList()
        for channel in post_process_channels:
            self.convs.append(
                ConvModule(channel, self.channels, kernel_size=3, padding=1)
            )

        self.project = ConvModule(self.channels, self.channels, kernel_size=3, padding=1)
        self.conv_depth = HeadDepth(self.channels)

    def forward(self, inputs):
        x = [self.reassemble_blocks[i](inp) for i, inp in enumerate(inputs)]
        x = [self.convs[i](feature) for i, feature in enumerate(x)]

        out = x[-1]
        for i in range(len(x) - 1):
            out = self.project(out + x[-(i + 2)])  # Simple addition, can be modified

        depth_output = self.conv_depth(out)
        return depth_output


class MetaDinoV2RS(nn.Module):
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
            self.decode_head = DPTHead(
                in_channels=(1280, 1280, 1280, 1280),
                embed_dims=1280,
                post_process_channels=[160, 320, 640, 1280],
                classify=False
            )
        else:
            self.backbone = SSLVisionTransformer(init_cfg=pretrained)
            self.decode_head = RPTHead()

    def forward(self, x):
        x = self.backbone(x)
        x = self.decode_head(x)
        return x


