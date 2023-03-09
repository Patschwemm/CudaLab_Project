import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Union
from Pathlib import Path
from modules import * 

# Todo:
# 1. For a RNN connections between iterations like the skip connections in the encoder have the hidden states of the decoder as well
# 1.5. Let the encoder concatenate the hidden states from the prior iteration from the decoder hidden states
# 2. implement for each layer the hidden states to be input (in trainer function most likely)
# 3. Optional: have the initial hidden state be a learnable nn.Params 

@dataclass
class RNN_UNetConfig:
    """Configuration for U-Net."""

    out_channels: int
    encoder_blocks: list[list[int]]
    decoder_blocks: list[list[int]]
    l: list[int]
    dim: int = 2
    concat_hidden: bool = False
    use_pooling: bool = False
    batch_norm: bool = False


class RNN_UNetEncoder(nn.Module):
    """Encoder part of U-Net."""

    def __init__(
        self,
        dim: int,
        blocks: tuple[tuple[int, ...]],
        l: int,
        use_pooling: bool = False,
        batch_norm: bool = True,
    ) -> None:
        super(RNN_UNetEncoder, self).__init__()

        # set the block channel up where the l is specified
        print(blocks)
        for layer in l:
            blocks[layer][0] = int(blocks[layer][0] * 2)
        print(blocks)

        # initial convolution block for the UNet architecture
        self.in_block = ConvBlock(dim, blocks[0], batch_norm)

        # downsamples followed by convolution
        self.downsample_blocks = nn.ModuleList(
            [DownsampleBlock(dim, channels, use_pooling, batch_norm) for channels in blocks[1:]]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # initial convolution block
        x = self.in_block(x)

        # extract skip connections
        hidden_states = []
        for block in self.downsample_blocks:
            hidden_states.append(x)
            x = block(x)

        return x, hidden_states


class RNN_UNetDecoder(nn.Module):
    """Decoder part of U-Net."""

    def __init__(
        self,
        dim: int,
        out_channels: int,
        blocks: tuple[tuple[int, ...]],
        batch_norm: bool = True,
        concat_hidden: bool = False,
    ) -> None:
        super(RNN_UNetDecoder, self).__init__()

        # upsample followed by convolution
        self.upsample_blocks = nn.ModuleList(
            [UpsampleBlock(dim, channels, batch_norm, concat_hidden) for channels in blocks]
        )

        # 1x1 convolution for last block of convolution - UNet Structure
        self.out_block = ConvNd(
            dim=dim,
            in_channels=blocks[-1][-1],
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
        )

        # boolean to concatenate hidden or sum op hidden state
        self.concat_hidden = concat_hidden

    def forward(self, x: torch.Tensor, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        for block, h in zip(self.upsample_blocks, reversed(hidden_states)):

            rnn_hidden = []

            # if concatenation is wanted
            if self.concat_hidden:
                x = block.upsample(x)
                h = center_pad(h, x.shape[2:])
                x = torch.cat([x, h], dim=1)
                x = block.conv_block(x)
                rnn_hidden.append(x)
            # if summing up is wanted
            else:
                x = block.upsample(x)
                h = center_pad(h, x.shape[2:])
                x = x + h
                x = block.conv_block(x)
                rnn_hidden.append(x)        

        return self.out_block(x)


class UNet(nn.Module):
    """U-Net segmentation model."""

    def __init__(self, config: RNN_UNetConfig) -> None:
        super(UNet, self).__init__()
        self.config = config

        self.encoder = RNN_UNetEncoder(
            config.dim,
            config.encoder_blocks,
            config.l,
            config.use_pooling,
            config.batch_norm
        )

        self.decoder = RNN_UNetDecoder(
            config.dim,
            config.out_channels,
            config.decoder_blocks,
            config.batch_norm,
            config.concat_hidden,
        )

        # self.encoder.apply(init_weights)
        # self.decoder.apply(init_weights)

    def forward(self, x: torch.Tensor, rnn_states: torch.Tensor) -> torch.Tensor:
        x, hidden_states = self.encoder(x, rnn_states)
        x, rnn_states = self.decoder(x, hidden_states)
        return x, rnn_states

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    # @classmethod
    # def from_pretrained(cls, path: str, config: Union[UNetConfig, str, None] = None) -> "UNet":
    #     path = Path(path)

    #     if config is None:
    #         config = UNetConfig.from_file(path.parent / "model.yaml")
    #     elif not isinstance(config, UNetConfig):
    #         config = UNetConfig.from_file(config)

    #     model = cls(config)
    #     model.load_state_dict(torch.load(path, map_location=idist.device()))
    #     return model