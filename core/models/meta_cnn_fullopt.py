from collections import OrderedDict
from typing import List

import torch
from torch import Tensor, nn
from torch.types import Device

from core.models.layers.utils import *  # noqa: F403

from .meta_base import ConvBlock, LinearBlock, Meta_Base
from torch.nn import functional as F
from .layers.quantized_conv2d import QConv2d
from .layers.quantized_linear import QLinear
__all__ = ["FullOptMetalens"]


class FFT_FeatureExtractor(nn.Module):
    def __init__(self, encode_mode: str="mag_phase") -> None:
        super().__init__()
        self.encode_mode = encode_mode

    def forward(self, x):

        # Compute 2D FFT
        fft_result = torch.fft.rfft2(x, norm="ortho")

        if self.encode_mode == "mag":
            fft_mag = torch.abs(fft_result)
            return fft_mag
        elif self.encode_mode == "phase":
            fft_phase = torch.angle(fft_result)  # Phase is in range [-π, π]
            return fft_phase
        elif self.encode_mode == "mag_phase":
            return fft_result
        else:
            raise ValueError(f"Unknown encode_mode: {self.encode_mode}")

class Receiver1D(nn.Module):
    def __init__(self, length: int, num_classes: int, apply_softmax: bool = False):
        """
        Partitions 'length' into 'num_classes' equal segments (or as equal as possible).
        Sums the intensity in each segment => [B, num_classes].
        Optionally applies softmax.
        """
        super().__init__()
        self.length = length
        self.num_classes = num_classes
        self.apply_softmax = apply_softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, channels, L]  (channels could be 1 or outC)
        returns: [B, num_classes]
        """
        # print(x.shape)
        # print(x[1,1,:])
        B, C, L = x.shape
        if L != self.length:
            raise ValueError(f"Expected length={self.length}, got {L}.")

        # 2) Sum over channels if needed, or keep if each channel is separate
        #    Typically, if you have 1 channel, skip this. 
        #    If outC=1, then x => shape [B, 1, L]. We'll just keep it that way for now.
        #    If you have multiple channels that you want combined, do:
        # x = x.sum(dim=1)  # => [B, L]
        # then reshape => [B, 1, L] or proceed. 
        # We'll skip that for now unless your design needs it.

        # 3) Partition length dimension L into num_classes segments
        #    We'll assume L is divisible by num_classes for simplicity
        segment_size = L // self.num_classes

        # 4) Sum the intensities in each segment => shape [B, channels, num_classes]
        out_list = []
        for i in range(self.num_classes):
            start = i * segment_size
            # last segment goes to the end if not perfectly divisible
            end = start + segment_size if (i < self.num_classes - 1) else L
            seg = x[:, :, start:end]   # => [B, channels, seg_length]
            seg_sum = (torch.abs(seg) ** 2).sum(dim=-1)  # => [B, channels]
            out_list.append(seg_sum)

        # 5) Concatenate => [B, channels, num_classes]
        x_part = torch.stack(out_list, dim=-1)  # => [B, channels, num_classes]

        # print(x_part.shape)
        x_part = x_part.sum(dim=1) if x_part.shape[1] != 1 else x_part.squeeze(1) # => [B, num_classes]
        # print(x_part.shape)
        # print(x_part[1,:])
        # exit(0)
        # 6) If each channel = a separate class, you might keep channels dimension
        #    BUT typically we want exactly [B, num_classes].
        #    If channels=1, it is shape [B, 1, num_classes], so we do:
        # x_part = x_part.squeeze(1)  # => [B, num_classes] if channels=1
        # If channels != 1, adapt accordingly.

        # # 7) Optional softmax for classification
        if self.apply_softmax:
            x_part = F.softmax(x_part, dim=-1)

        return x_part

class FullOptMetalens(Meta_Base):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        in_channels: int,
        num_classes: int,
        kernel_list: List[int] = [32],
        mid_channel_list: List[int] = [32],
        kernel_size_list: List[int] = [32],
        stride_list=[1],
        padding_list=[0],
        dilation_list=[1],
        groups_list=[1],
        conv_cfg=dict(type="MetaConv2d"),
        linear_cfg=dict(type="Linear"),
        device: Device = torch.device("cuda"),
    ) -> None:
        super().__init__(conv_cfg=conv_cfg, linear_cfg=linear_cfg, device=device)
        self.conv_cfg = conv_cfg
        self.img_height = img_height
        self.img_width = img_width
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.kernel_list = kernel_list
        self.mid_channel_list = mid_channel_list
        self.kernel_size_list = kernel_size_list
        self.stride_list = stride_list
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.groups_list = groups_list

        self.device = device

        self.build_layers()
        self.reset_parameters()

    def build_layers(self):

        print(self.conv_cfg['encode_mode'], flush=True)
        self.feature_extraction = FFT_FeatureExtractor(self.conv_cfg['encode_mode'])
        
        self.features = OrderedDict()
        for idx, (mid_channels, out_channels) in enumerate(
            zip(self.mid_channel_list, self.kernel_list), 0
        ):
            layer_name = "conv" + str(idx + 1)
            in_channels = self.in_channels if (idx == 0) else self.kernel_list[idx - 1]
            self.features[layer_name] = ConvBlock(
                in_channels,
                out_channels,
                self.kernel_size_list[idx],
                self.stride_list[idx],
                self.padding_list[idx],
                self.dilation_list[idx],
                self.groups_list[idx],
                bias=False,
                mid_channels=mid_channels,
                conv_cfg=self.conv_cfg,
                norm_cfg=None,  # enable batchnorm
                act_cfg=None,  # enable relu
                device=self.device,
            )

        self.features = nn.Sequential(self.features)

        self.receiver = Receiver1D(
            length=self.kernel_size_list[-1], num_classes=self.num_classes, apply_softmax=False
        )

    def set_test_mode(self, test_mode=True):
        for name, module in self.features.named_children():
            if hasattr(module, "set_test_mode"):
                module.set_test_mode(test_mode)


    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extraction(x)
        x = x.flatten(-2)
        # linear interpolation on the last dimension to the self.kernel_size_list[0]
        # if x is a complex tensor, we need to interpolate the real and imaginary part separately
        x_real = F.interpolate(x.real, size=self.kernel_size_list[0], mode='nearest')
        x_imag = F.interpolate(x.imag, size=self.kernel_size_list[0], mode='nearest')
        x = x_real + 1j*x_imag
        # print("this is the shape of x", x.shape)
        # quit()
        x = self.features(x)
        # [bs, OutC, L]
        # print(x)
        # if self.pool1d is not None:
        #     x = self.pool1d(x)
        # x = torch.flatten(x, 1)
        # x = self.temp_classifier(x)
        # x = self.receiver(x)
        # if self.pool2d is not None:
        #     x = self.pool2d(x)
        # x = torch.flatten(x, 1)
        x = self.receiver(x)

        return x
