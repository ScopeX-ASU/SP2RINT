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
__all__ = ["Meta_CNNETE", "CNN"]

class LTNForClassification(nn.Module):
    def __init__(self, soc_out_channels=3, out_H=27, out_W=27, hidden_dim=100, num_classes=10, pooling_kernel=4, pooling_stride=1):
        super().__init__()
        self.conv1x1 = nn.Conv2d(soc_out_channels, soc_out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=pooling_kernel, stride=pooling_stride)

        pooled_H = (out_H - pooling_kernel) // pooling_stride + 1
        pooled_W = (out_W - pooling_kernel) // pooling_stride + 1
        flat_dim = soc_out_channels * pooled_H * pooled_W
        
        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FeatureExtractor1D(nn.Module):
    """
    A small CNN-based feature extractor that outputs (B, 1, out_length).
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 16,
        hidden_channels: int = 8,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.out_channels = out_channels

        # Example CNN block:
        #   Conv -> BN -> ReLU -> adaptive pool => flatten => linear => [B, out_length] => [B, 1, out_length]
        self.conv = QConv2d(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            in_bit=8,
            w_bit=8,
            out_bit=8,
        )
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)

        # Example: pool to a small 2×2
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        # So after the pool, shape is [B, hidden_channels, 2, 2]

        # Then a linear to produce out_length
        # That means in_features = hidden_channels * (2 * 2)
        self.fc = QLinear(
            hidden_channels * 4, 
            out_channels,
            w_bit=8,
            in_bit=8,
            out_bit=8
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_channels, H, W]
        returns: [B, 1, out_length]
        """
        # 1) CNN extraction
        x = self.conv(x)         # => [B, hidden_channels, H, W]
        x = self.bn(x)           
        x = self.relu(x)         
        x = self.pool(x)         # => [B, hidden_channels, 2, 2]

        # 2) Flatten
        x = x.flatten(start_dim=1)  # => [B, hidden_channels * 4]

        # 3) Map to out_length => [B, out_length]
        x = self.fc(x)

        # 4) Reshape to single channel [B, 1, out_length]
        x = x.unsqueeze(-1)       # => [B, 1, out_length]
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
        # print(x_part.shape)
        # print(x_part[1,:])
        # exit(0)
        # 6) If each channel = a separate class, you might keep channels dimension
        #    BUT typically we want exactly [B, out_length].
        #    If channels=1, it is shape [B, 1, out_length], so we do:
        # x_part = x_part.squeeze(1)  # => [B, out_length] if channels=1
        # If channels != 1, adapt accordingly.

        # # 7) Optional softmax for classification
        # if self.apply_softmax:
        #     x_part = F.softmax(x_part, dim=-1)

        return x_part

class FMNISTFeatureExtractor(nn.Module):
    def __init__(self, in_channels = 1, feature_dim=128):
        super(FMNISTFeatureExtractor, self).__init__()
        # Convolution layers
        self.conv1 = QConv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1, 
                                w_bit=16, 
                                in_bit=16, 
                                out_bit=16
                                )
        self.conv2 = QConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, 
                                w_bit=16,  
                                in_bit=16, 
                                out_bit=16
                                )
        
        # Fully connected layers to get the desired feature dimension
        # Note: 64 * 7 * 7 = 3136, after two 2×2 pooling operations on 28×28
        self.fc1 = QLinear(64 * 7 * 7, 256, 
                            w_bit=16, 
                            in_bit=16, 
                            out_bit=16
                            )
        self.fc2 = QLinear(256, feature_dim, 
                           w_bit=16, 
                           in_bit=16, 
                           out_bit=16
                           )

    def forward(self, x):
        """
        x: [batch_size, 1, 28, 28] (FMNIST images)
        Returns a feature tensor of shape [batch_size, feature_dim].
        """
        x = F.relu(self.conv1(x))      # -> [batch_size, 32, 28, 28]
        x = F.max_pool2d(x, 2)         # -> [batch_size, 32, 14, 14]
        x = F.relu(self.conv2(x))      # -> [batch_size, 64, 14, 14]
        x = F.max_pool2d(x, 2)         # -> [batch_size, 64, 7, 7]

        # Flatten
        x = x.view(x.size(0), -1)      # -> [batch_size, 64 * 7 * 7] = [batch_size, 3136]

        # Two fully connected layers
        x = F.relu(self.fc1(x))        # -> [batch_size, 256]
        features = self.fc2(x)         # -> [batch_size, feature_dim]
        return features

class Meta_CNNETE(Meta_Base):
    def __init__(
        self,
        img_height: int,
        img_width: int,
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
        darcy: bool = False,
    ) -> None:
        super().__init__(conv_cfg=conv_cfg, linear_cfg=linear_cfg, device=device)
        self.darcy = darcy
        self.conv_cfg = conv_cfg
        self.linear_cfg = linear_cfg
        self.digital_norm_cfg = digital_norm_cfg
        self.digital_act_cfg = digital_act_cfg
        self.optical_norm_cfg = optical_norm_cfg
        self.optical_act_cfg = optical_act_cfg
        self.img_height = img_height
        self.img_width = img_width
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
            self.feature_extraction = FMNISTFeatureExtractor(in_channels=self.in_channels, feature_dim=self.feature_dim)
            
        
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
        out_size = self.img_height - self.window_size + 1
        if self.darcy:
            # self.pre_digit = ConvBlock(
            #     in_channels=1,
            #     out_channels=self.window_size ** 2,
            #     kernel_size=1,
            #     stride=1,
            #     padding=0,
            #     dilation=1,
            #     groups=1,
            #     bias=False,
            #     mid_channels=0,
            #     conv_cfg=dict(type="Conv2d"),
            #     norm_cfg=dict(type="BN", affine=True),
            #     act_cfg=None,
            #     device=self.device,
            # )
            self.regressor = OrderedDict()
            self.regressor["regression_head_1"] = ConvBlock(
                in_channels=self.feature_dim,
                out_channels=self.feature_dim * 2,
                kernel_size=7,
                stride=1,
                padding=3,
                dilation=1,
                groups=1,
                bias=False,
                mid_channels=0,
                conv_cfg=dict(type="Conv2d"),
                norm_cfg=dict(type="BN", affine=True),
                act_cfg=dict(type="ReLU"),
                device=self.device,
            )
            self.regressor["regression_head_2"] = ConvBlock(
                in_channels=self.feature_dim * 2,
                out_channels=self.feature_dim * 4,
                kernel_size=5,
                stride=1,
                padding=2,
                dilation=1,
                groups=1,
                bias=False,
                mid_channels=0,
                conv_cfg=dict(type="Conv2d"),
                norm_cfg=dict(type="BN", affine=True),
                act_cfg=dict(type="ReLU"),
                device=self.device,
            )
            self.regressor["regression_head_3"] = ConvBlock(
                in_channels=self.feature_dim * 4,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=3,
                dilation=1,
                groups=1,
                bias=False,
                mid_channels=0,
                conv_cfg=dict(type="Conv2d"),
                norm_cfg=None,
                act_cfg=None,
                device=self.device,
            )
            self.regressor = nn.Sequential(self.regressor)
        else:
            self.classifier = LTNForClassification(
                soc_out_channels=self.feature_dim,
                out_H=out_size,
                out_W=out_size,
                num_classes=self.num_classes
            )

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

    def forward(self, x: Tensor, sharpness: float = None) -> Tensor:
        if sharpness is not None:
            assert self.conv_cfg["TM_model_method"] == "end2end", "only end2end TM model can use sharpness"
        total_sys_lens = round(self.kernel_size_list[-1] * self.conv_cfg["resolution"] * self.conv_cfg["pixel_size_data"])
        # [B, 1, 28, 28]
        bs, C, H, W = x.shape
        if C > 1:
            x = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
            C = 1
        p = self.window_size
        if self.darcy:
        #     # we need to pad the image using window size // 2
        #     x = self.pre_digit(x) # [B, window_size^2, H, W]
            x = F.pad(x, (p // 2, p // 2, p // 2, p // 2), mode='constant')
        #     x = torch.fft.rfftn(x, dim=(-2, -1), norm="backward") # [B, window_size^2, H, W // 2 + 1]
        #     B, _, H2, W2 = x.shape
        #     patches = x.permute(0, 2, 3, 1).reshape(B * H2 * W2, self.window_size ** 2).unsqueeze(1) # [B * H * (W // 2 + 1), 1, window_size^2]
        # else:
        # Ensure image size is sufficient
        assert H >= p and W >= p, "Image smaller than patch size"
        # Unfold: turn image into sliding patches
        patches = x.unfold(2, p, 1).unfold(3, p, 1)  # (bs, C, H-p+1, W-p+1, p, p)
        out_H = patches.shape[2]
        out_W = patches.shape[3]
        patches = patches.contiguous().view(bs, C, out_H * out_W, p * p)  # (bs, C, N, p*p)
        patches = patches.reshape(bs * C * out_H * out_W, p * p).unsqueeze(1)  # (bs*C*out_H*out_W, 1, p*p)
        source_mask = torch.ones_like(patches, dtype=torch.bool, device=patches.device)
        x = insert_zeros_after_every_N_except_last(patches, self.input_wg_width_px, self.input_wg_interval_px)
        source_mask = insert_zeros_after_every_N_except_last(source_mask, self.input_wg_width_px, self.input_wg_interval_px)
        assert x.shape[-1] <= total_sys_lens, f"the length of the input signal is larger than the total system length, {x.shape[-1]} > {total_sys_lens}"
        total_pad_len = total_sys_lens - x.shape[-1]
        pad_len_1 = total_pad_len // 2
        pad_len_2 = total_pad_len - pad_len_1
        padding_1 = torch.zeros((*x.shape[:-1], pad_len_1), dtype=x.dtype, device=x.device)
        padding_2 = torch.zeros((*x.shape[:-1], pad_len_2), dtype=x.dtype, device=x.device)
        boolean_padding_1 = torch.zeros((*source_mask.shape[:-1], pad_len_1), dtype=source_mask.dtype, device=source_mask.device)
        boolean_padding_2 = torch.zeros((*source_mask.shape[:-1], pad_len_2), dtype=source_mask.dtype, device=source_mask.device)
        x = torch.cat([padding_1, x, padding_2], dim=-1) # bs*C*out_H*out_W, 1, 480
        source_mask = torch.cat([boolean_padding_1, source_mask, boolean_padding_2], dim=-1)
        # concate x and source_mask making a tensor with shape (bs*C*out_H*out_W, 1, 480, 2)
        x = torch.cat([x.unsqueeze(-1), source_mask.unsqueeze(-1)], dim=-1)


        # x = self.feature_extraction(x)
        # x = x[:, :self.fft_mode_1, :self.fft_mode_2].flatten(1)
        # x = x.unsqueeze(1) # [B, 1, L]
        # # linear interpolation on the last dimension to the self.kernel_size_list[0]
        # x_real = F.interpolate(
        #     x.real, 
        #     size=round(self.kernel_size_list[0] * 50 * self.conv_cfg["pixel_size_data"] / self.conv_cfg["in_downsample_rate"]), 
        #     mode='linear'
        # )
        # x_imag = F.interpolate(
        #     x.imag, 
        #     size=round(self.kernel_size_list[0] * 50 * self.conv_cfg["pixel_size_data"] / self.conv_cfg["in_downsample_rate"]), 
        #     mode='linear'
        # )
        # x = x_real + 1j*x_imag
        # print("this is the stat of x before feature extraction", flush=True)
        # print_stat(x)
        x, inner_fields = self.features((x, sharpness)) # the pd is already considered in the feature layer bs*C*out_H*out_W, 1, 480
        inner_fields = [inner_field.squeeze().unsqueeze(1) for inner_field in inner_fields]
        inner_fields = torch.cat(inner_fields, dim=1) # bs*C*out_H*out_W, 1, 480
        x = F.interpolate(
            x, 
            size=round(self.kernel_size_list[-1] * self.conv_cfg["resolution"] * self.conv_cfg["pixel_size_data"]), 
            mode='linear'
        ) # bs*C*out_H*out_W, feature_out_channel, 480
        # make it to be (bs * out_H * out_W, C, 480)
        # if not self.darcy:
        #     x = x.view(bs, C, out_H, out_W, -1).permute(0, 2, 3, 1, 4).reshape(bs * out_H * out_W, C, -1)
        #     assert C == 1, "the channel must be 1"
        x = x.view(bs, C, out_H, out_W, -1).permute(0, 2, 3, 1, 4).reshape(bs * out_H * out_W, C, -1)
        assert C == 1, "the channel must be 1"
        # print("this is the shape of features before receiver", x.shape, flush=True)
        x = self.receiver(x) # bs*out_H*out_W, out_length
        x = self.rms_norm(x)
        x = x.reshape(bs, out_H, out_W, -1).permute(0, 3, 1, 2) # bs, out_length, out_H, out_W
        # if not self.darcy:
        #     x = x.reshape(bs, out_H, out_W, -1).permute(0, 3, 1, 2) # bs, out_length, out_H, out_W
        # else:
        #     x = x.reshape(B, H2, W2, -1).permute(0, 3, 1, 2)
        #     zero_mask = torch.zeros(H2, W2, device=x.device)
        #     zero_mask[
        #         :10, :10
        #     ] = 1
        #     zero_mask[
        #         -10:, :10
        #     ] = 1
        #     x = x * zero_mask.unsqueeze(0).unsqueeze(0)
        #     x = torch.fft.irfftn(x, dim=(-2, -1), norm="backward")
        # print_stat(x)
        # quit()
        if self.darcy:
            x = self.regressor(x)
        else:    
            x = self.classifier(x)

        return x, inner_fields


def CNN(*args, **kwargs):
    kwargs.pop("conv_cfg")
    kwargs.pop("linear_cfg")
    kwargs.update(dict(conv_cfg=dict(type="Conv2d"), linear_cfg=dict(type="Linear")))
    return Meta_CNNETE(*args, **kwargs)
