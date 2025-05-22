from collections import OrderedDict
from typing import List

import torch
from torch import Tensor, nn
from torch.types import Device

from core.models.layers.utils import *  # noqa

from .meta_base import ConvBlock, ConvBlockPTC, Meta_Base

__all__ = ["Meta_CNN_DNO", "CNN_DNO"]


class Meta_CNN_DNO(Meta_Base):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
        norm_cfg=dict(type="BN", affine=True),
        act_cfg=dict(type="ReLU", inplace=True),
        prediction_kernel_list: List[int] = [32],
        prediction_kernel_size_list: List[int] = [32],
        prediction_stride_list=[1],
        prediction_padding_list=[0],
        prediction_dilation_list=[1],
        prediction_groups_list=[1],
        prediction_conv_cfg=dict(type="PTCBlockConv2d"),
        prediction_norm_cfg=dict(type="BN", affine=True),
        prediction_act_cfg=dict(type="GeLU", inplace=True),
        device: Device = torch.device("cuda"),
    ) -> None:
        super().__init__(conv_cfg=conv_cfg, linear_cfg=linear_cfg)
        self.conv_cfg = conv_cfg
        self.linear_cfg = linear_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_channels = in_channels
        self.kernel_list = kernel_list
        self.mid_channel_list = mid_channel_list
        self.kernel_size_list = kernel_size_list
        self.stride_list = stride_list
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.groups_list = groups_list
        self.out_channels = out_channels
        self.prediction_kernel_list = prediction_kernel_list
        self.prediction_kernel_size_list = prediction_kernel_size_list
        self.prediction_stride_list = prediction_stride_list
        self.prediction_padding_list = prediction_padding_list
        self.prediction_dilation_list = prediction_dilation_list
        self.prediction_groups_list = prediction_groups_list
        self.prediction_conv_cfg = prediction_conv_cfg
        self.prediction_norm_cfg = prediction_norm_cfg
        self.prediction_act_cfg = prediction_act_cfg

        self.pool_out_size = pool_out_size
        self.hidden_list = hidden_list

        self.device = device

        self.build_layers()
        self.reset_parameters()

    def build_layers(self):
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
                bias=True,
                mid_channels=mid_channels,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,  # enable batchnorm
                act_cfg=self.act_cfg,  # enable relu
                device=self.device,
            )

        self.features = nn.Sequential(self.features)

        self.predictor = OrderedDict()
        for idx, out_channels in enumerate(self.prediction_kernel_list, 0):
            layer_name = "predictor_conv" + str(idx + 1)
            in_channels = (
                self.kernel_list[-1]
                if (idx == 0)
                else self.prediction_kernel_list[idx - 1]
            )

            # If it's the last layer, use the provided out_channels instead of the one from the list
            if idx == len(self.prediction_kernel_list) - 1 and hasattr(
                self, "out_channels"
            ):
                out_channels = self.out_channels

            self.predictor[layer_name] = ConvBlockPTC(
                in_channels,
                out_channels,
                self.prediction_kernel_size_list[idx],
                self.prediction_stride_list[idx],
                self.prediction_padding_list[idx],
                self.prediction_dilation_list[0],
                1,
                bias=True,
                conv_cfg=self.prediction_conv_cfg,
                norm_cfg=self.prediction_norm_cfg
                if idx < len(self.prediction_kernel_list) - 1
                else None,  # enable batchnorm
                act_cfg=self.prediction_act_cfg
                if idx < len(self.prediction_kernel_list) - 1
                else None,  # enable GELU
                device=self.device,
            )
        self.predictor = nn.Sequential(self.predictor)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.predictor(x)

        return x


def CNN_DNO(*args, **kwargs):
    kwargs.pop("conv_cfg")
    kwargs.pop("linear_cfg")
    kwargs.update(dict(conv_cfg=dict(type="Conv2d"), linear_cfg=dict(type="Linear")))
    return Meta_CNN_DNO(*args, **kwargs)
