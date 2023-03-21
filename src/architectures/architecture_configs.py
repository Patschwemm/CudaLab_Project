from dataclasses import dataclass
import torch
import torch.nn as nn
from . import vanilla_modules 
from . import resnet_modules 
from .temporal_modules import Conv2dRNNCell, Conv2dGRUCell

@dataclass
class Temporal_TemplateUNetConfig:
    """Configuration for U-Net."""
    initblock: nn.Module
    downsampleblock: nn.Module
    upsampleblock: nn.Module
    temporal_cell: nn.Module = Conv2dGRUCell
    out_channels: int = 91
    encoder_blocks: list[list[int]] = [[3, 64, 64], [64, 128, 128], [128, 256, 256]],
    # these are the dimensions for concatenation, if summing is wanted, reduce the first dimension for each block
    decoder_blocks: list[list[int]] = [[512, 256, 256], [256, 128, 128], [128, 64, 64]],
    concat_hidden: bool = True
    use_pooling: bool = False
    batch_norm: bool = False

@dataclass 
class Temporal_VanillaUNetConfig:

    initblock: nn.Module = vanilla_modules.ConvBlock
    downsampleblock: nn.Module = vanilla_modules.DownsampleBlock
    upsampleblock: nn.Module = vanilla_modules.UpsampleBlock
    temporal_cell: nn.Module = Conv2dRNNCell
    out_channels: int = 91
    encoder_blocks: list[list[int]] = [[3, 64, 64], [64, 128, 128], [128, 256, 256]],
    # these are the dimensions for concatenation, if summing is wanted, reduce the first dimension for each block
    decoder_blocks: list[list[int]] = [[512, 256, 256], [256, 128, 128], [128, 64, 64]],
    concat_hidden: bool = True
    use_pooling: bool = False
    batch_norm: bool = False

@dataclass 
class Temporal_ResNetUNetConfig:

    initblock: nn.Module = resnet_modules.ResNet18Initblock
    downsampleblock: nn.Module = resnet_modules.Resnet18Downblock
    upsampleblock: nn.Module = resnet_modules.Resnet18Upblock
    temporal_cell: nn.Module = Conv2dGRUCell
    out_channels: int = 91
    encoder_blocks: list[list[int]] = [[3, 64, 64], [64, 128, 128], [128, 256, 256]],
    # these are the dimensions for concatenation, if summing is wanted, reduce the first dimension for each block
    decoder_blocks: list[list[int]] = [[512, 256, 256], [256, 128, 128], [128, 64, 64]],
    concat_hidden: bool = True
    use_pooling: bool = False
    batch_norm: bool = False