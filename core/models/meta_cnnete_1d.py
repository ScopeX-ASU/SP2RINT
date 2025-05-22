from collections import OrderedDict
from typing import List

import torch
from torch import Tensor, nn
from torch.types import Device

from core.models.layers.utils import *  # noqa: F403
from core.utils import insert_zeros_after_every_N_except_last
from pyutils.general import print_stat
from .meta_base import ConvBlock, LinearBlock, Meta_Base
from torch.nn import functional as F
from .layers.quantized_conv2d import QConv2d
from .layers.quantized_linear import QLinear
__all__ = ["Meta_CNNETE_1D"]
    
class LTNForClassification(nn.Module):
    def __init__(
            self, 
            soc_out_channels=3, 
            out_length=100, 
            hidden_dim=16, 
            num_classes=10, 
            pooling_kernel=4, 
            pooling_stride=1
        ):
        super().__init__()
        self.conv1x1 = nn.Conv1d(soc_out_channels, soc_out_channels, kernel_size=1, bias=False)
        if out_length != 1:
            self.pool = nn.AvgPool1d(kernel_size=pooling_kernel, stride=pooling_stride)

            pooled_L = (out_length - pooling_kernel) // pooling_stride + 1
            flat_dim = soc_out_channels * pooled_L
        else:
            self.pool = nn.Identity()
            flat_dim = soc_out_channels * out_length
        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: Tensor of shape [B, soc_out_channels, L]
        """
        # x = self.conv1x1(x)              # [B, soc_out_channels, L]
        # x = self.pool(x)                 # [B, soc_out_channels, L_pooled]
        x = x.view(x.size(0), -1)        # [B, soc_out_channels * L_pooled]
        x = F.relu(self.fc1(x))          # [B, hidden_dim]
        x = self.fc2(x)                  # [B, num_classes]
        return x

class Receiver1D(nn.Module):
    def __init__(self, in_length: int, out_length: int, apply_softmax: bool = False):
        """
        Partitions 'in_length' into 'out_length' equal segments (or as equal as possible).
        Sums the intensity in each segment => [B, out_length].
        Optionally applies softmax.
        """
        super().__init__()
        self.in_length = in_length
        self.out_length = out_length
        self.apply_softmax = apply_softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, channels, L]  (channels could be 1 or outC)
        returns: [B, out_length]
        """
        # print(x.shape)
        # print(x[1,1,:])
        B, C, L = x.shape
        if L != self.in_length:
            raise ValueError(f"Expected in_length={self.in_length}, got {L}.")

        # 2) Sum over channels if needed, or keep if each channel is separate
        #    Typically, if you have 1 channel, skip this. 
        #    If outC=1, then x => shape [B, 1, L]. We'll just keep it that way for now.
        #    If you have multiple channels that you want combined, do:
        # x = x.sum(dim=1)  # => [B, L]
        # then reshape => [B, 1, L] or proceed. 
        # We'll skip that for now unless your design needs it.

        # 3) Partition length dimension L into out_length segments
        #    We'll assume L is divisible by out_length for simplicity

        segment_size = L // self.out_length

        if L % self.out_length == 0:
            pass
        else:
            effective_length = L - (L % self.out_length)
            x = x[:, :, L//2 - effective_length // 2: L//2 - effective_length // 2 + effective_length]

        # 4) Sum the intensities in each segment => shape [B, channels, out_length]
        out_list = []
        for i in range(self.out_length):
            start = i * segment_size
            # last segment goes to the end if not perfectly divisible
            end = start + segment_size if (i < self.out_length - 1) else L
            seg = x[:, :, start:end]   # => [B, channels, seg_length]
            seg_sum = seg.sum(dim=-1)  # => [B, channels]
            out_list.append(seg_sum)

        # 5) Concatenate => [B, channels, out_length]
        x_part = torch.stack(out_list, dim=-1)  # => [B, channels, out_length]
        # print(x_part.shape)
        x_part = x_part.sum(dim=1) if x_part.shape[1] != 1 else x_part.squeeze(1) # => [B, out_length]

        return x_part


class Meta_CNNETE_1D(Meta_Base):
    def __init__(
        self,
        sequence_length: int,
        in_channels: int,
        input_wg_width: float,
        input_wg_interval: float,
        feature_dim: int,
        num_classes: int,
        kernel_list: List[int] = [32],
        mid_channel_list: List[int] = [32],
        kernel_size_list: List[int] = [32],
        stride_list=[1],
        padding_list=[0],
        dilation_list=[1],
        groups_list=[1],
        pool_out_size: int = 5,
        hidden_list: List[int] = [32],
        conv_cfg=dict(type="MetaConv2d"),
        linear_cfg=dict(type="Linear"),
        digital_norm_cfg=dict(type="BN", affine=True),
        digital_act_cfg=dict(type="ReLU", inplace=True),
        optical_norm_cfg=None,
        optical_act_cfg=None,
        device: Device = torch.device("cuda"),
        feature_extractor_type: str = "fft",
        fft_mode_1: int = 3,
        fft_mode_2: int = 3,
        window_size: int = 2,
        full_opt: bool = True,
    ) -> None:
        super().__init__(conv_cfg=conv_cfg, linear_cfg=linear_cfg, device=device)
        self.conv_cfg = conv_cfg
        self.linear_cfg = linear_cfg
        self.digital_norm_cfg = digital_norm_cfg
        self.digital_act_cfg = digital_act_cfg
        self.optical_norm_cfg = optical_norm_cfg
        self.optical_act_cfg = optical_act_cfg
        self.sequence_length = sequence_length
        self.in_channels = in_channels
        self.input_wg_width = input_wg_width
        self.input_wg_width_px = round(input_wg_width * 50)
        self.input_wg_interval = input_wg_interval
        self.input_wg_interval_px = round(input_wg_interval * 50)
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.kernel_list = kernel_list
        assert self.kernel_list == [1], "kernel_list must be [1] since we only consider 1 channel in this project"
        self.mid_channel_list = mid_channel_list
        self.kernel_size_list = kernel_size_list
        self.stride_list = stride_list
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.groups_list = groups_list
        self.feature_extractor_type = feature_extractor_type
        self.fft_mode_1 = fft_mode_1
        self.fft_mode_2 = fft_mode_2
        self.window_size = window_size

        self.pool_out_size = pool_out_size
        self.hidden_list = hidden_list

        self.device = device

        self.full_opt = full_opt
        if self.full_opt:
            assert self.feature_dim == self.num_classes, "feature_dim must be equal to num_classes when full_opt is True"

        self.build_layers()
        self.reset_parameters()

    def build_layers(self):
        if self.feature_extractor_type == "fft":
            raise NotImplementedError("FFT feature extractor is deprecated now")
            print(self.conv_cfg['encode_mode'], flush=True)
            # assert self.conv_cfg['encode_mode'] == "mag", "when using fft as feature extractor, encodee mode must be mag"
            self.feature_extraction = FFT_FeatureExtractor()
        elif self.feature_extractor_type == "none":
            self.feature_extraction = None
        else:
            raise NotImplementedError("feature extractor type has to be 'none'")

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
                norm_cfg=self.optical_norm_cfg, # optical part can't have normalization
                act_cfg=self.optical_act_cfg, # optical part can't have activation
                device=self.device,
            )

        self.features = nn.Sequential(self.features)

        # if self.pool_out_size > 0:
        #     self.pool1d = nn.AdaptiveAvgPool1d(self.pool_out_size)
        #     feature_size = (
        #         self.kernel_list[-1] * self.pool_out_size
        #     )
        # else:
        #     self.pool1d = None
        #     signal_length = self.signal_length
        #     for layer in self.modules():
        #         if isinstance(layer, self._conv):
        #             sig_length = layer.get_output_dim(signal_length)
        #     feature_size = self.kernel_list[-1] * sig_length
        
        self.receiver = Receiver1D(
            in_length=self.kernel_size_list[-1] * self.conv_cfg["resolution"] * self.conv_cfg["pixel_size_data"], 
            out_length=self.feature_dim, 
            apply_softmax=False,
        )

        self.rms_norm = torch.nn.RMSNorm(
            normalized_shape=self.feature_dim,
            eps=None,
            elementwise_affine=False,
        )
        out_length = self.sequence_length - self.window_size + 1
        if not self.full_opt:
            self.classifier = LTNForClassification(
                soc_out_channels=self.feature_dim,
                out_length=out_length,
                num_classes=self.num_classes
            )
        else:
            self.classifier = None

        # self.classifier = OrderedDict()
        # for idx, hidden_dim in enumerate(self.hidden_list, 0):
        #     layer_name = "fc" + str(idx + 1)
        #     in_features = self.feature_dim if idx == 0 else self.hidden_list[idx - 1]
        #     out_features = hidden_dim
        #     self.classifier[layer_name] = LinearBlock(
        #         in_features,
        #         out_features,
        #         bias=True,
        #         linear_cfg=self.linear_cfg,
        #         act_cfg=self.digital_act_cfg,
        #         norm_cfg=self.digital_norm_cfg,
        #         device=self.device,
        #     )

        # layer_name = "fc" + str(len(self.hidden_list) + 1)
        # self.classifier[layer_name] = LinearBlock(
        #     self.hidden_list[-1] if len(self.hidden_list) > 0 else self.feature_dim,
        #     self.num_classes,
        #     bias=True,
        #     linear_cfg=self.linear_cfg,
        #     act_cfg=None,
        #     device=self.device,
        # )
        # self.classifier = nn.Sequential(self.classifier)

    def set_test_mode(self, test_mode=True):
        for name, module in self.features.named_children():
            if hasattr(module, "set_test_mode"):
                module.set_test_mode(test_mode)

    def set_near2far_matrix(self, near2far_matrix):
        for name, module in self.features.named_children():
            if hasattr(module, "set_near2far_matrix"):
                module.set_near2far_matrix(near2far_matrix)
  
    def forward(self, x: Tensor) -> Tensor:
        total_sys_lens = round(self.kernel_size_list[-1] * self.conv_cfg["resolution"] * self.conv_cfg["pixel_size_data"])
        x = x.unsqueeze(1)  # [B, 1, L]
        bs, C, L = x.shape  # Input is [B, 1, L]
        assert C == 1, "Only 1 input channel is supported for 1D mode"

        p = self.window_size
        assert L >= p, "Input length smaller than patch size"

        # Unfold the 1D signal to sliding patches of length `p`
        patches = x.unfold(dimension=2, size=p, step=1)  # [B, C, L-p+1, p]
        out_L = patches.shape[2]

        patches = patches.contiguous().view(bs * C * out_L, p).unsqueeze(1)  # [B*out_L, 1, p]

        # Apply waveguide spacing
        source_mask = torch.ones_like(patches, dtype=torch.bool, device=patches.device)
        x = insert_zeros_after_every_N_except_last(patches, self.input_wg_width_px, self.input_wg_interval_px)
        source_mask = insert_zeros_after_every_N_except_last(source_mask, self.input_wg_width_px, self.input_wg_interval_px)

        assert x.shape[-1] <= total_sys_lens, f"Input signal too long after encoding: {x.shape[-1]} > {total_sys_lens}"

        total_pad_len = total_sys_lens - x.shape[-1]
        pad_len_1 = total_pad_len // 2
        pad_len_2 = total_pad_len - pad_len_1

        padding_1 = torch.zeros((*x.shape[:-1], pad_len_1), dtype=x.dtype, device=x.device)
        padding_2 = torch.zeros((*x.shape[:-1], pad_len_2), dtype=x.dtype, device=x.device)
        boolean_padding_1 = torch.zeros((*source_mask.shape[:-1], pad_len_1), dtype=source_mask.dtype, device=source_mask.device)
        boolean_padding_2 = torch.zeros((*source_mask.shape[:-1], pad_len_2), dtype=source_mask.dtype, device=source_mask.device)

        x = torch.cat([padding_1, x, padding_2], dim=-1)  # [B*out_L, 1, total_sys_lens]
        source_mask = torch.cat([boolean_padding_1, source_mask, boolean_padding_2], dim=-1)
        x = torch.cat([x.unsqueeze(-1), source_mask.unsqueeze(-1)], dim=-1)  # [B*out_L, 1, total_sys_lens, 2]
        # Optical propagation
        x, inner_fields = self.features((x, None))  # -> [B*out_L, feature_out_channel, L]
        inner_fields = [f.squeeze().unsqueeze(1) for f in inner_fields]
        inner_fields = torch.cat(inner_fields, dim=1)
        x = F.interpolate(
            x,
            size=total_sys_lens,
            mode="linear"
        )  # [B*out_L, feature_out_channel, total_sys_lens]

        x = x.view(bs, C, out_L, -1).permute(0, 2, 1, 3).reshape(bs * out_L, C, -1)  # [B, out_L, L] => [B, C, out_L, L] => [B, C*out_L, L]
        # Receiver
        x = self.receiver(x)  # [B, feature_dim]
        x = self.rms_norm(x)
        if self.full_opt:
            x = x.flatten(start_dim=1)
            return x, inner_fields
        x = x.reshape(bs, out_L, -1).permute(0, 2, 1) # bs, out_length, out_H, out_W
        x = self.classifier(x)  # [B, num_classes]


        return x, inner_fields


