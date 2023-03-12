import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Union
from pathlib import Path
from modules import * 
from temporal_modules import *


@dataclass
class RNN_UNetConfig:
    """Configuration for U-Net."""
    out_channels: int = 2
    encoder_blocks: list[list[int]] = [[3, 64, 64], [64, 128, 128], [128, 256, 256]],
    # these are the dimensions for concatenation, if summing is wanted, reduce the first dimension for each block
    decoder_blocks: list[list[int]] = [[512, 256, 256], [256, 128, 128], [128, 64, 64]],
    dim: int = 2
    concat_hidden: bool = True
    use_pooling: bool = False
    batch_norm: bool = False


class RNN_UNetEncoder(nn.Module):
    """Encoder part of U-Net."""

    def __init__(
        self,
        blocks: tuple[tuple[int, ...]],
        use_pooling: bool = False,
        batch_norm: bool = True,
    ) -> None:

        super(RNN_UNetEncoder, self).__init__()
        self.in_block = ConvBlock(blocks[0], batch_norm)
        self.downsample_blocks = nn.ModuleList(
            [DownsampleBlock(channels, use_pooling, batch_norm) for channels in blocks[1:]]
        )

        conv_rnns = []
        for channels in blocks:
            in_size = channels[-1]
            conv_rnns.append(Conv2dRNNCell(input_size=in_size, hidden_size=in_size, kernel_size=3))
        
        self.conv_rnn = nn.ModuleList(conv_rnns)

        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_block(x)

        hidden_states = []
        rnn_states = []
        for block, c_rnn in zip(self.downsample_blocks, self.conv_rnn[:-1]):
            # hidden_states.append(x)
            # pass through RNN 
            rnn_states.append(c_rnn(x))
            x = block(x)
            # append rnn states for concatenation
        
        # pass through the last conv rnn state
        x = self.conv_rnn[-1](x)

        return x, hidden_states, rnn_states


class RNN_UNetDecoder(nn.Module):
    """Decoder part of U-Net."""

    def __init__(
        self,
        out_channels: int,
        blocks: tuple[tuple[int, ...]],
        batch_norm: bool = True,
        concat_hidden: bool = True,
    ) -> None:
        super(RNN_UNetDecoder, self).__init__()
        self.in_block = ConvBlock(blocks[0], batch_norm)
        self.upsample_blocks = nn.ModuleList(
            [UpsampleBlock(channels, batch_norm, concat_hidden) for channels in blocks[1:]]
        )
        self.out_block = nn.Conv2d(
            in_channels=blocks[-1][-1],
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        self.concat_hidden = concat_hidden

    def forward(self, x: torch.Tensor, rnn_states: list[torch.Tensor]) -> torch.Tensor:
        # for block, h, conv_rnn in zip(self.upsample_blocks, reversed(hidden_states), rnn_states):
        for block, conv_rnn in zip(self.upsample_blocks, reversed(rnn_states)):
            if self.concat_hidden:
                x = block.upsample(x)
                # conv_rnn = center_pad(conv_rnn, x.shape[2:])
                
                x = torch.cat([x, conv_rnn], dim=1)
                x = block.conv_block(x)
            else:
                x = block.upsample(x)
                # conv_rnn = center_pad(conv_rnn, x.shape[2:])
                x = x + conv_rnn
                x = block.conv_block(x)

        return self.out_block(x)


class RNN_UNet(nn.Module):
    """U-Net segmentation model."""

    def __init__(self, config: RNN_UNetConfig) -> None:
        super(RNN_UNet, self).__init__()
        self.config = config

        self.encoder = RNN_UNetEncoder(
            config.encoder_blocks[0], 
            config.use_pooling, 
            config.batch_norm
        )
        self.decoder = RNN_UNetDecoder(
            config.out_channels,
            config.decoder_blocks[0],
            config.batch_norm,
            config.concat_hidden,
        )

        # self.encoder.apply(init_weights)
        # self.decoder.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # loop over sequence
        outputs = []
        # x is Batch x Sequence x Channel x Height x Width
        for i in range(x.shape[1]):
            out, hidden_states, rnn_states = self.encoder(x[:, i])
            out = self.decoder(out, rnn_states)
            outputs.append(out)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs[:, -1]

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)