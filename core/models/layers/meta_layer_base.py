from typing import Optional

import torch
from pyutils.compute import gen_gaussian_noise
# from pyutils.quant.lsq import ActQuantizer_LSQ
from torch import Tensor, nn
from torch.types import Device

from .utils import DeviceQuantizer, pad_quantize_fn, ActQuantizer_LSQ
from pyutils.general import print_stat
__all__ = ["Meta_Layer_BASE"]


class Meta_Layer_BASE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        # zero control pads for metasurface by default. By default is passive metasuface
        n_pads: int = 0,
        w_bit: int = 16,
        in_bit: int = 16,
        phase_res: int = 2,  # Number digits after decimal point
        pixel_size_res: int = 1,
        delta_z_res: int = 10,
        mid_channels: int = 0,
        # constant scaling factor from intensity to detected voltages
        input_uncertainty: float = 0,
        pad_noise_std: float = 0,
        Meta: Optional[nn.Module] = None,
        pad_max: float = 1.0,
        mode: str = "phase",
        path_multiplier: int = 2,  # how many metalens in parallel
        path_depth: int = 2,  # how may metalens cascaded
        unfolding: bool = False,
        sigma_trainable: str = "row_col",
        enable_xy_pol: bool = True,  # whether to use x/y polarization
        # whether to use alpha factor for weighted input channel summation
        enable_alpha: bool = True,
        # whether to use beta factor as polarization angle for x direction
        enable_beta: bool = True,
        encode_mode: str = "mag",  # mag, phase, complex, intensity
        weight_train: bool = True,
        skip_path: bool = False,
        scale_mode: str = "bilinear",
        delta_z_mode: str = "fixed",  # fixed, train_share, train, this one is reprogrammable
        # fixed, train_share, train, this one is not reprogrammable after fabrication
        pixel_size_mode: str = "fixed",
        # fixed, train_share this one is reprogrammable after fabrication
        lambda_mode: str = "fixed",
        # fixed, train, this one is reprogrammable after fabrication
        rotate_mode: str = "fixed",
        gumbel_mode: str = "gumbel_soft",  # gumbel_hard, gumbel_soft, softmax, random
        # whether to use identity phase mask, i.e., delta_phi=0, can be learned together with rotation
        enable_identity: bool = False,
        # fixed, train_stage, train, this one is reprogrammable after fabrication
        swap_mode: str = "fixed",
        device: Device = torch.device("cuda"),
        verbose: bool = False,
        with_cp: bool = False,
        gumbel_temperature: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels

        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.n_pads = n_pads
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.phase_res = phase_res
        self.pixel_size_res = pixel_size_res
        self.delta_z_res = delta_z_res
        self.weight_train = weight_train
        self.encode_mode = encode_mode
        self.skip_path = skip_path
        self.scale_mode = scale_mode

        self.pad_max = pad_max
        # constant scaling factor from intensity to detected voltages
        self.input_uncertainty = input_uncertainty
        self.pad_noise_std = pad_noise_std
        self.mode = mode
        self.path_multiplier = path_multiplier
        self.path_depth = path_depth
        self.unfolding = unfolding
        self.enable_xy_pol = enable_xy_pol
        self.enable_alpha = enable_alpha
        self.enable_beta = enable_beta
        self.delta_z_mode = delta_z_mode
        self.pixel_size_mode = pixel_size_mode
        self.lambda_mode = lambda_mode
        self.rotate_mode = rotate_mode
        self.gumbel_mode = gumbel_mode
        self.enable_identity = enable_identity
        self.swap_mode = swap_mode

        self.sigma_trainable = sigma_trainable
        self.gumbel_temperature = gumbel_temperature

        self.verbose = verbose
        self.with_cp = with_cp
        self.device = device

        # allocate parameters
        self.weight = None
        self.path_weight = None
        self.sigma = None
        self.x_zero_pad = None
        self.mag = None
        # quantization tool
        self.pad_quantizer = pad_quantize_fn(
            max(2, self.w_bit), v_max=pad_max, quant_ratio=1
        )

        self.input_quantizer = ActQuantizer_LSQ(
            None,
            device=device,
            nbits=self.in_bit,
            offset=True,
            mode="tensor_wise",
        )

        self.phase_quantizer = DeviceQuantizer(
            device=device,
            resolution=self.phase_res,
            mode="degree",
        )

        self.mag_quantizer = DeviceQuantizer(
            device=device,
            resolution=None,
            mode="mag",
        )

        self._requires_grad_Meta = True
        self.input_er = 0
        self.input_max = 6
        self.input_snr = 0
        self.detection_snr = 0
        self.pad_noise_std = 0

    def build_parameters(self, bias: bool):
        raise NotImplementedError

    def reset_parameters(self, fan_in=None):
        # the fourier-domain convolution is equivalent to h x w kernel-size convolution
        if self.weight is not None:
            nn.init.uniform_(self.weight, -torch.pi, torch.pi)

        if self.beta is not None:
            self.beta.data.uniform_(-0.01, 0.01)

        if self.alpha_pre is not None:
            # very important, need to shrink the huge range after metaconv.
            self.alpha_pre.data.copy_(
                nn.Conv2d(self.in_channels, self.mid_channels, 1, bias=False)
                .weight.data.to(self.device)
                .view_as(self.alpha_pre)
            )

        if self.alpha_post is not None:
            # very important, need to shrink the huge range after metaconv.
            self.alpha_post.data.copy_(
                nn.Conv2d(self.mid_channels, self.out_channels, 1, bias=False)
                .weight.data.to(self.device)
                .view_as(self.alpha_post)
                * (1.2**self.path_depth)
            )

        if self.bias is not None:
            nn.init.uniform_(self.bias, 0, 0)

    def requires_grad_Meta(self, mode: bool = True):
        self._requires_grad_Meta = mode

    def set_input_er(self, er: float = 0, x_max: float = 6.0) -> None:
        # extinction ratio of input modulator
        self.input_er = er
        self.input_max = x_max

    def requires_grad_beta(self, mode: bool = True):
        self.beta.requires_grad = mode

    def set_meta_params(self, meta_params: nn.Module) -> None:
        self._conv_pos.meta_params = meta_params
        if self._conv_neg is not None:
            self._conv_neg.meta_params = meta_params

    def set_input_snr(self, snr: float = 0) -> None:
        self.input_snr = snr

    def set_detection_snr(self, snr: float = 0) -> None:
        self.detection_snr = snr

    def add_input_noise(self, x: Tensor) -> Tensor:
        if 1e-5 < self.input_er < 80:
            x_max = self.input_max
            x_min = x_max / 10 ** (self.input_er / 10)
            x = x.mul((x_max - x_min) / x_max).add(x_min)
        if 1e-5 < self.input_snr < 80:
            avg_noise_power = 1 / 10 ** (self.input_snr / 10)
            noise = gen_gaussian_noise(x, 1, noise_std=avg_noise_power**0.5)
            return x.mul(noise)
        return x

    def add_detection_noise(self, x: Tensor) -> Tensor:
        if 1e-5 < self.detection_snr < 80:
            avg_noise_power = 0.5 / 10 ** (self.detection_snr / 10)
            noise = gen_gaussian_noise(x, 0, noise_std=avg_noise_power**0.5)
            return x.add(noise)
        return x

    def set_gumbel_temperature(self, T: float = 5.0):
        self.gumbel_temperature = T

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        self.input_quantizer.set_bit(in_bit)

    def set_phase_res(self, res: int) -> None:
        self.phase_res = res
        self.phase_quantizer.set_resolution(res)

    @property
    def _weight(self):
        # mask: quantize the metasurface mask, if one channel used, it means phase, otherwise is complex value for
        # by default, the weight is complex number
        # the phase should be the angle of the complex number
        # and then convert it to degree instead of radians
        phase = torch.angle(self.weight)
        mag = torch.abs(self.weight)
        phase = self.phase_quantizer(phase)
        mag = self.mag_quantizer(mag)
        # here, we apply learnable rotation to the phase mask
        # rotate_mask = self.meta_params.build_rotate_mask()
        # weight = self.meta_params.apply_rotate_mask(weight, rotate_mask)  # [m,d,2,s,s]

        # next, we apply learnable permutation/swap matrix
        # weight = self.meta_params.apply_swap(weight)  # [m,d,2,s,s]

        return phase, mag

    def _forward_impl(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self._forward_impl(x)
