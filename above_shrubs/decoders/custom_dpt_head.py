# Standard library imports
import os
import warnings
from datetime import datetime
import glob


warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.parallel.parallel_apply")

# PyTorch and related
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Computer vision and image processing
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

# Hugging Face
from transformers import AutoModel
from datasets import load_dataset, load_from_disk

# Visualization
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm

# Progress bars
from tqdm import tqdm



class ChannelAttention(nn.Module):
    """Channel attention mechanism to focus on important features"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.attention(x)


class DINOv3DepthHead(nn.Module):
    def __init__(self, model_name="facebook/dinov3-vitl16-pretrain-sat493m", 
             freeze_backbone=True, output_channels=1, token=None, input_bands='rgb',
             training_config=None):
        
        super().__init__()
        
        # Store training config for use in decoder
        self.training_config = training_config or {}
        self.dropout_rate = self.training_config.get('dropout_rate', 0.1)  # Default to 0.1
        
        # Store output_channels for use in decoder
        self.output_channels = output_channels
        
        # Store output_channels for use in decoder
        self.output_channels = output_channels
        
        # Load DINOv3 backbone
        if token:
            self.backbone = AutoModel.from_pretrained(model_name, token=token)
        else:
            self.backbone = AutoModel.from_pretrained(model_name)
    
        # Modify input weights for NRG if specified
        if input_bands == 'nrg':
            self._modify_input_weights_for_nrg()
        
        # Get actual model dimensions dynamically
        self._determine_model_dimensions()
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self._freeze_backbone = True
        
        print(f"DINOv3 DepthHead initialized:")
        print(f"  Model: {model_name}")
        print(f"  Input bands: {input_bands}")
        print(f"  Embedding dim: {self.embed_dim}")
        print(f"  Total tokens: {self.total_tokens}")
        print(f"  Spatial patches: {self.num_spatial_patches}")
        print(f"  Using tokens {self.spatial_start}:{self.spatial_end}")
        
        # Build improved decoder
        self.depth_head = self._build_improved_decoder()
    
        # Initialize the final layer - ADD THIS LINE:
        self._initialize_output_layer(target_percentile=8.42)  # 70% - 1.72, 90% = 8.42 Use your actual mean or percentile here
    
    def _determine_model_dimensions(self):
        """Dynamically determine model dimensions by running a test forward pass"""
        # Use 64x64 dummy input to match your actual data
        dummy_input = torch.randn(1, 3, 64, 64)
        
        with torch.no_grad():
            outputs = self.backbone(dummy_input)
            features = outputs.last_hidden_state
        
        # Get actual dimensions
        self.total_tokens = features.shape[1]
        self.embed_dim = features.shape[2]
        
        # Calculate spatial patch info for 64x64 input with 16x16 patches
        self.patches_per_side = 64 // 16  # = 4
        self.num_spatial_patches = self.patches_per_side ** 2  # = 16
        
        # Determine token structure
        non_spatial_tokens = self.total_tokens - self.num_spatial_patches
        
        if non_spatial_tokens == 1:
            # Structure: [CLS] + [16 spatial]
            self.spatial_start = 1
            self.spatial_end = self.total_tokens
        elif non_spatial_tokens == 5:
            # Structure: [CLS] + [4 register] + [16 spatial]  
            self.spatial_start = 5
            self.spatial_end = self.total_tokens
        else:
            # Generic: take last 16 tokens as spatial
            self.spatial_start = self.total_tokens - self.num_spatial_patches
            self.spatial_end = self.total_tokens
        
        print(f"Detected model structure:")
        print(f"  Total tokens: {self.total_tokens}")
        print(f"  Embedding dim: {self.embed_dim}")
        print(f"  Non-spatial tokens: {non_spatial_tokens}")
        print(f"  Input size: 64x64")
        print(f"  Patch grid: {self.patches_per_side}x{self.patches_per_side}")

    def _modify_input_weights_for_nrg(self):
        """Simple weight modification for NIR-Red-Green input"""
        print("Modifying input weights for NIR-Red-Green bands...")
        
        # Find the first Conv2d layer with 3 input channels (this is the patch embedding)
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                print(f"Found patch embedding at: {name}")
                
                with torch.no_grad():
                    original_weights = module.weight.data.clone()
                    
                    # NIR=Red, Red=Red, Green=Green (assuming BGR input order)
                    new_weights = torch.zeros_like(original_weights)
                    new_weights[:, 0, :, :] = original_weights[:, 2, :, :]  # NIR <- Red
                    new_weights[:, 1, :, :] = original_weights[:, 2, :, :]  # Red <- Red  
                    new_weights[:, 2, :, :] = original_weights[:, 1, :, :]  # Green <- Green
                    
                    module.weight.data = new_weights
                    print("✅ Weights modified successfully")
                break

    def _build_improved_decoder(self):
        """
        Improved decoder adapted for 16x16 input (1x1 spatial features)
        """
        
        class ResidualBlock(nn.Module):
            """Residual block with LeakyReLU and attention"""
            def __init__(self, channels, use_attention=True, dropout_rate=0.1):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(channels)
                self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(channels)
                self.activation = nn.LeakyReLU(0.1, inplace=True)
                
                # Add dropout after activation
                self.dropout = nn.Dropout2d(dropout_rate)
                
                self.attention = ChannelAttention(channels) if use_attention else nn.Identity()
                
            def forward(self, x):
                residual = x
                out = self.activation(self.bn1(self.conv1(x)))
                out = self.dropout(out)  # Apply dropout after activation
                out = self.bn2(self.conv2(out))
                out = self.attention(out)
                out += residual  # Skip connection
                return self.activation(out)
               
        # Get dropout rate from training config
        dropout_rate = self.dropout_rate
        
        layers = nn.ModuleList()
        current_channels = self.embed_dim
        
        # Need 4 upsampling steps: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        target_channels = [512, 256, 128, 64]  # Same number, but explicit about path
        
        for i, out_channels in enumerate(target_channels):
            # Upsampling block (2x upsampling each time)
            upsample_block = nn.Sequential(
                nn.ConvTranspose2d(current_channels, out_channels, 
                                 kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout2d(dropout_rate)  # Use config dropout rate
            )
            layers.append(upsample_block)
            
            # Residual refinement blocks with dropout
            refinement_block = nn.Sequential(
                ResidualBlock(out_channels, use_attention=(i >= 2), dropout_rate=dropout_rate),
                ResidualBlock(out_channels, use_attention=(i >= 2), dropout_rate=dropout_rate)
            )
            layers.append(refinement_block)
            
            current_channels = out_channels

        # Final layers - these should NOT change spatial dimensions
        # After 4 upsampling steps, we should already be at 64x64
        final_layers = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(dropout_rate),  # Use config dropout rate
            ChannelAttention(128),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(dropout_rate),  # Use config dropout rate
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            
            # No dropout before final prediction layer
            # Final prediction - should already be 64x64
            nn.Conv2d(32, self.output_channels, kernel_size=1)
        )
        layers.append(final_layers)
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
    # Validate input size
        if x.shape[-2:] != (64, 64):
            raise ValueError(f"Expected input size 64x64, got {x.shape[-2:]}")
        
        # Get features from DINOv3
        if hasattr(self, '_freeze_backbone'):
            with torch.no_grad():
                outputs = self.backbone(x)
        else:
            outputs = self.backbone(x)
            
        features = outputs.last_hidden_state
        
        # Extract spatial tokens (should be just 1 token for 16x16 input)
        spatial_tokens = features[:, self.spatial_start:self.spatial_end, :]
        batch_size = spatial_tokens.shape[0]
        
        # Verify we have the right number of spatial patches
        expected_spatial_tokens = self.spatial_end - self.spatial_start
        if spatial_tokens.shape[1] != expected_spatial_tokens:
            raise ValueError(f"Expected {expected_spatial_tokens} spatial tokens, got {spatial_tokens.shape[1]}")
        
        # Reshape to spatial grid (4x4 for 64x64 input)  # ← CORRECTED COMMENT
        spatial_features = spatial_tokens.transpose(1, 2).reshape(
            batch_size, self.embed_dim, self.patches_per_side, self.patches_per_side  # 4x4
        )
        
        # Pass through decoder (4x4 -> 64x64)  # ← CORRECTED COMMENT
        depth_map = self.depth_head(spatial_features)
        depth_map = depth_map.squeeze(1)

        depth_map = torch.clamp(depth_map, min=0.0)
        
        return depth_map
        
    def _initialize_output_layer(self, target_mean=None, target_percentile=None):
        """
        Initialize final layer to better predict the full range with skewed data
        
        Args:
            target_mean (float, optional): The mean value of your targets
            target_percentile (float, optional): A specific percentile value from your targets
                                               (e.g., 0.7 would be the 70th percentile)
        """
        # Find the final convolutional layer
        final_conv = None
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                final_conv = module
        
        if final_conv is None:
            print("Warning: Could not find final convolutional layer for initialization")
            return
            
        with torch.no_grad():
            # Initialize weights with a slightly higher standard deviation
            # to encourage more diverse predictions
            nn.init.normal_(final_conv.weight, mean=0.0, std=0.02)
            
            # Initialize the bias to address the skew
            if final_conv.bias is not None:
                # If you've calculated statistics from your dataset:
                if target_mean is not None:
                    final_conv.bias.fill_(target_mean)
                elif target_percentile is not None:
                    final_conv.bias.fill_(target_percentile)
                else:
                    # Default initialization - adjust based on your data's scale
                    # This value should be in the same scale as your target values
                    final_conv.bias.fill_(3.0)  # Example value


if __name__ == "__main__":

    print("Testing")
