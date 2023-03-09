import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Union
from pathlib import Path
from modules import * 


@dataclass
class RNN_UNetConfig:
    """Configuration for U-Net."""
    out_channels: 2
    encoder_blocks: [[3, 64, 64], [64, 128, 128], [128, 256, 256]]]
    decoder_blocks: [[512, 256, 256], [256, 128, 128], [128, 64, 64]]
    dim: int = 2
    concat_hidden: bool = True
    use_pooling: bool = False
    batch_norm: bool = False


class RNN_UNetEncoder(nn.Module):
    """Encoder part of U-Net."""

    def __init__(
        self,
        dim: int,
        blocks: tuple[tuple[int, ...]],
        use_pooling: bool = False,
        batch_norm: bool = True,
    ) -> None:
        super(UNetEncoder, self).__init__()
        self.in_block = ConvBlock(blocks[0], batch_norm)
        self.downsample_blocks = nn.ModuleList(
            [DownsampleBlock(channels, use_pooling, batch_norm) for channels in blocks[1:]]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_block(x)

        hidden_states = []
        for block in self.downsample_blocks:
            hidden_states.append(x)
            x = block(x)

        return x, hidden_states


class RNN_UNetDecoder(nn.Module):
    """Decoder part of U-Net."""

    def __init__(
        self,
        out_channels: int,
        blocks: tuple[tuple[int, ...]],
        batch_norm: bool = True,
        concat_hidden: bool = True,
    ) -> None:
        super(UNetDecoder, self).__init__()
        self.in_block = ConvBlock(block[0], batch_norm)
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

    def forward(self, x: torch.Tensor, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        for block, h in zip(self.upsample_blocks, reversed(hidden_states)):
            if self.concat_hidden:
                x = block.upsample(x)
                h = center_pad(h, x.shape[2:])
                x = torch.cat([x, h], dim=1)
                x = block.conv_block(x)
            else:
                x = block(x)
                h = center_pad(h, x.shape[2:])
                x = x + h

        return self.out_block(x)


class RNN_UNet(nn.Module):
    """U-Net segmentation model."""

    def __init__(self, config: UNetConfig) -> None:
        super(UNet, self).__init__()
        self.config = config

        self.encoder = UNetEncoder(
            config.encoder_blocks, 
            config.use_pooling, 
            config.batch_norm
        )
        self.decoder = UNetDecoder(
            config.out_channels,
            config.decoder_blocks,
            config.batch_norm,
            config.concat_hidden,
        )

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, hidden_states = self.encoder(x)
        x = self.decoder(x, hidden_states)
        return x

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