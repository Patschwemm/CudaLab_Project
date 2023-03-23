import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNextBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvNextBlock, self).__init__()
        self.dwconv = nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=3, stride=1, groups=dim, padding_mode='reflect') # depthwise conv
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.pwconv1 = nn.Linear(out_ch, 4 * out_ch)  # pointwise/1x1 convs, implemented with linear layers
        self.act1 = nn.GELU()
        self.pwconv2 = nn.Linear(4 * out_ch, out_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm1(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.norm2(x)
        x = self.act2(residual + x)

        return x

    
class ConvNextInitblock(nn.Module):

    def __init__(self, block, batch_norm = True) -> None:
        super().__init__()
        

    def forward(self, x):
        
        return x

class ConvNextBlock(nn.Module):

    def __init__(self, block, k_size=3, batch_norm = True) -> None:
        super().__init__()


    def forward(self, x):
        
        return out
    

class ConvNextUpblock(nn.Module):

    def __init__(self, block, batch_norm = True, cut_channels_on_upsample = False) -> None:
        super().__init__()
        # to be called seperately it needs to have the same self.upsample and self.conv_block names

    
    def forward(self, x):
        
        return x
    
class ConvNextDownblock(nn.Module):

    def __init__(self, block, use_pooling, batch_norm = True) -> None:
        super().__init__()

    def forward(self, x):
        
        return x





