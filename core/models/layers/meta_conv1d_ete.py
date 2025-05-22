from functools import lru_cache
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from einops import einsum
from mmengine.registry import MODELS
from pyutils.general import print_stat
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair, _single
from torch.types import Device

from core.models.layers.meta_layer_base import Meta_Layer_BASE
from core.models.layers.utils import hard_diff_round
from core.utils import DeterministicCtx, get_mid_weight
import h5py
from .utils import DeviceQuantizer, PixelLambdaConstraint, WeightQuantizer_LSQ
import matplotlib.pyplot as plt
from ceviche.constants import *
from thirdparty.MAPS_old.core.fdfd.near2far import get_farfields_GreenFunction
from thirdparty.MAPS_old.core.fdfd.fdfd import fdfd_hz
from thirdparty.MAPS_old.core.invdes.models.base_optimization import DefaultSimulationConfig
from thirdparty.MAPS_old.core.invdes.models.layers import MetaLens
from thirdparty.MAPS_old.core.invdes.models import MetaLensOptimization
import csv
__all__ = ["MetaParams", "MetaConv1dETE"]

DEBUG = False

def load_lut_from_csv(csv_path):
    widths = []
    phases = []
    mags = []

    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            width = float(row["width"])
            phase = float(row["phase_mean"])
            mag = float(row["mag_mean"])
            widths.append(width)
            phases.append(phase)
            mags.append(mag)

    # Remove duplicates by converting to dict then back to list (if needed)
    zipped = list(dict.fromkeys(zip(widths, mags, phases)))  # deduplicate based on width
    widths, mags, phases = zip(*zipped)

    lut_widths = torch.tensor(widths, dtype=torch.float32)
    amp_lut = torch.tensor(mags, dtype=torch.float32)
    phase_lut = torch.tensor(phases, dtype=torch.float32)

    return lut_widths, amp_lut, phase_lut

def torch_interpolate_1d(x, xp, fp):
    """
    Mimics numpy.interp(x, xp, fp) using torch operations.
    Assumes xp is 1D sorted in ascending order.
    """
    indices = torch.searchsorted(xp, x, right=True).clamp(1, len(xp)-1)
    x0 = xp[indices - 1]
    x1 = xp[indices]
    y0 = fp[indices - 1]
    y1 = fp[indices]

    slope = (y1 - y0) / (x1 - x0 + 1e-12)
    return y0 + slope * (x - x0)

class CPositionAdaptiveConv1D(nn.Module):

    def __init__(
        self,
        length: int,
        kernel_size: int = 7,
        stride: int = 1,
        padding: int = None,
        device: str = "cuda",
        init_file_path: str = None,
        init_file_dataset: str = "transfer_matrix",
        fixed_amp: bool = False,
        in_downsample_rate: int = 1,
        out_downsample_rate: int = 1,
        pixel_size: float = 0.3,
        resolution: int = 50,
        max_tm_norm: bool = False,
        calculate_in_hr: bool = False,
        TM_model_method: str = "default", # can be conv or default
        LUT_path: str = None,
    ):
        super().__init__()

        self.length = length
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        self.init_file_path = init_file_path
        self.init_file_dataset = init_file_dataset
        self.test_mode = False
        self.fixed_amp = fixed_amp
        self.in_downsample_rate = in_downsample_rate
        self.out_downsample_rate = out_downsample_rate
        self.pixel_size = pixel_size
        self.resolution = resolution
        self.max_tm_norm = max_tm_norm
        self.calculate_in_hr = calculate_in_hr
        self.TM_model_method = TM_model_method
        assert TM_model_method in ["default", "conv", "fourier_basis", "end2end"], f"TM model method {TM_model_method} is not supported, please use default, conv, fourier_basis or end2end."
        self.LUT_path = LUT_path
        if self.fixed_amp:
            assert kernel_size == 1, "Fixed amplitude only works for kernel_size=1 where we use pure diagonal assumption."

        if padding is None:
            # 'same' for stride=1
            self.padding = (kernel_size - 1) // 2
        else:
            self.padding = padding

        self.zero_mask = torch.zeros(
            round(length * self.pixel_size * self.resolution / self.out_downsample_rate), 
            round(length * self.pixel_size * self.resolution / self.in_downsample_rate),
            device=device,
        ) # 480, 32
        for i in range(self.zero_mask.shape[1]):
            row_center_idx = round(i / self.zero_mask.shape[1] * self.zero_mask.shape[0]) + round(self.pixel_size * self.resolution / self.out_downsample_rate) // 2
            kernel_length = round(kernel_size * self.pixel_size * self.resolution / self.out_downsample_rate)
            for j in range(
                max(0, row_center_idx - kernel_length // 2),
                min(self.zero_mask.shape[0], row_center_idx - kernel_length // 2 + kernel_length),
            ):
                self.zero_mask[j, i] = 1
        # plt.figure()
        # plt.imshow(self.zero_mask.cpu().numpy())
        # plt.savefig("./figs/zero_mask.png")
        # plt.close()
        # quit()

        # Initialize phase as a 2D learnable matrix
        self.init_param()
        if self.TM_model_method == "end2end":
            # we need to init a metalensoptimization to so that we can build the compute graph from the width all the way to the transfer matrix
            self.build_metalens_optimization()
            self.total_normalizer_list = None
        # ------------------------------------------------
        # 3) (Optional) register a buffer to store the final W
        # ------------------------------------------------
        self.register_buffer(
            "W_buffer",
            torch.zeros(
                round(length * self.pixel_size * self.resolution / self.out_downsample_rate), 
                round(length * self.pixel_size * self.resolution / self.in_downsample_rate),
                dtype=torch.complex64, 
                device=device
            )
        )

    def build_metalens_optimization(self):
        total_sim_cfg = DefaultSimulationConfig()
        total_sim_cfg.update(
            dict(
                solver="ceviche_torch",
                numerical_solver="solve_direct",
                use_autodiff=False,
                neural_solver=None,
                border_width=[0, 0, 0.5, 0.5],
                PML=[0.5, 0.5],
                resolution=50,
                wl_cen=0.85,
                plot_root="./figs/dummy_plot",
            )
        )
        total_metalens = MetaLens(
            material_bg="Air",
            material_r = "Si",
            material_sub="SiO2",
            sim_cfg=total_sim_cfg,
            aperture=self.pixel_size * self.length,
            port_len=(1, 1),
            port_width=(self.pixel_size * self.length, self.pixel_size),
            substrate_depth=0,
            ridge_height_max=0.75,
            nearfield_dx=0.3,
            nearfield_size=self.pixel_size * self.length,
            farfield_dxs=((4, 4 + 2/self.resolution),), # hard code since we only see the farfield at 4 um
            farfield_sizes=(self.pixel_size * self.length,),
            device=self.device,
            design_var_type="width",
            atom_width=self.pixel_size / 2, # does not matter since we are not using the height as the design variable type
        )
    
        hr_total_metalens = total_metalens.copy(resolution=200)
        self.total_opt = MetaLensOptimization(
            device=total_metalens,
            hr_device=hr_total_metalens,
            sim_cfg=total_sim_cfg,
            operation_device=self.device,
            design_region_param_cfgs={},
        )

    def init_param(self):
        if self.TM_model_method == "default":
            self.phase = nn.Parameter(
                torch.empty(
                    round(self.length * self.pixel_size * self.resolution / self.out_downsample_rate), 
                    round(self.length * self.pixel_size * self.resolution / self.in_downsample_rate),
                    device=self.device, 
                    dtype=torch.float32
                ).uniform_(-torch.pi, torch.pi)
            )

            # Initialize amplitude as a 2D matrix of ones
            self.amp = nn.Parameter(
                torch.rand_like(self.phase, device=self.device, dtype=torch.float32)
            )

            if self.fixed_amp:
                self.amp.requires_grad = False  # Make amplitude non-learnable if fixed


            self.reset_parameters()
            if self.init_file_path is not None:
                # 2a) Load W_target from the HDF5 file
                with h5py.File(self.init_file_path, "r") as f:
                    W_data = f[self.init_file_dataset][...]  # e.g. (L, L) complex
                W_target = torch.as_tensor(W_data, device=self.device)
                if self.max_tm_norm:
                    W_target = W_target / torch.max(torch.abs(W_target))
                self.set_param_from_target_matrix(W_target)
        elif self.TM_model_method == "conv" or self.TM_model_method == "end2end":
            assert self.LUT_path is not None, "LUT path should be provided for conv TM model."
            if self.TM_model_method == "conv":
                assert self.in_downsample_rate == self.out_downsample_rate == round(self.pixel_size * self.resolution), "TM model method conv only supports in_downsample_rate == out_downsample_rate == pixel_size * resolution."
            lut_widths, amp_lut, phase_lut = load_lut_from_csv(self.LUT_path)
            self.set_lookup_table(lut_widths, amp_lut, phase_lut)

            self.widths = nn.Parameter(
                torch.empty(
                    self.length, 
                    device=self.device, 
                    dtype=torch.float32
                ).uniform_(0.01, 0.28)
            )

            if self.init_file_path is not None:
                # 2a) Load W_target from the HDF5 file
                with h5py.File(self.init_file_path, "r") as f:
                    W_data = f[self.init_file_dataset][...]  # e.g. (L, L) complex
                W_target = torch.as_tensor(W_data, device=self.device)
                if self.max_tm_norm:
                    W_target = W_target / torch.max(torch.abs(W_target))
                self.set_param_from_target_matrix(W_target)
        elif self.TM_model_method == "fourier_basis":
            assert round(self.length * self.pixel_size * self.resolution / self.out_downsample_rate) >= 15, "Fourier basis TM model requires at least 15 pixels in the input dimension."
            self.low_freq_component_left = nn.Parameter(
                torch.zeros(
                    8, # hard code since we only tolerant 8 low freq basis
                    round(self.length * self.pixel_size * self.resolution / self.in_downsample_rate), 
                    device=self.device, 
                    dtype=torch.complex64
                )
            )
            self.low_freq_component_right = nn.Parameter(
                torch.zeros(
                    7, # hard code since we only tolerant 7 low freq basis
                    round(self.length * self.pixel_size * self.resolution / self.in_downsample_rate), 
                    device=self.device, 
                    dtype=torch.complex64
                )
            )

            if round(self.length * self.pixel_size * self.resolution / self.out_downsample_rate) >= 38 + 34:
                self.high_freq_component = nn.Parameter(
                    torch.zeros(
                        round(self.length * self.pixel_size * self.resolution / self.out_downsample_rate) - 38 - 34,
                        round(self.length * self.pixel_size * self.resolution / self.in_downsample_rate), 
                        device=self.device, 
                        dtype=torch.complex64
                    )
                )  
            else:
                self.high_freq_component = None

            if self.init_file_path is not None:
                # 2a) Load W_target from the HDF5 file
                with h5py.File(self.init_file_path, "r") as f:
                    W_data = f[self.init_file_dataset][...]  # e.g. (L, L) complex
                W_target = torch.as_tensor(W_data, device=self.device)
                if self.max_tm_norm:
                    W_target = W_target / torch.max(torch.abs(W_target))
                self.set_param_from_target_matrix(W_target)  
            else:
                raise ValueError("better provide the initialization file for fourier basis TM model.")
        else:
            raise ValueError(f"TM model method {self.TM_model_method} is not supported, please use default or conv.")

    def set_lookup_table(self, lut_widths: torch.Tensor, amp: torch.Tensor, phase: torch.Tensor):
        self.register_buffer("lut_widths", lut_widths.to(self.device))
        self.register_buffer("amp_lut", amp.to(self.device))
        self.register_buffer("phase_lut", phase.to(self.device))

    def set_test_mode(self, test_mode: bool = True):
        self.test_mode = test_mode

    def reset_parameters(self):
        if self.fixed_amp:
            amp_init = torch.zeros_like(self.amp)
            amp_init[self.zero_mask == 1] = 1.0
        else:
            amp_init = torch.empty_like(self.amp, dtype=torch.float32).uniform_(0.8, 1.2)
        phase_init = torch.empty_like(self.phase, dtype=torch.float32).uniform_(-torch.pi, torch.pi)
        self.amp.data.copy_(amp_init)
        self.phase.data.copy_(phase_init)
        # weight = amp_init * torch.exp(1j * phase_init)
        # self.weight.data.copy_(weight)

    def set_param_from_target_matrix(self, W_target: torch.Tensor, if_quit=False):
        # have to ensure that the target matrix is noemalized and the shape is the same as the current weight matrix
        if self.max_tm_norm:
            assert torch.isclose(torch.max(torch.abs(W_target)).to(torch.float32), torch.tensor(1.0, dtype=torch.float32, device=W_target.device), atol=1e-3), f"The target matrix should be normalized {torch.max(torch.abs(W_target))}."
        # assert W_target.shape == self.phase.data.shape, f"The target matrix should have the same shape ({W_target.shape}) as the current weight matrix {self.phase.data.shape}."
        if isinstance(W_target, np.ndarray):
            W_target = torch.as_tensor(W_target, device=self.device)
        # if self.fixed_amp and not self.test_mode:
        #     W_target_diagonal_phase = torch.diag(torch.angle(W_target))
        #     W_target = torch.diag(torch.exp(1j * W_target_diagonal_phase))
        if W_target.shape != (
            round(self.length * self.pixel_size * self.resolution / self.out_downsample_rate),
            round(self.length * self.pixel_size * self.resolution / self.in_downsample_rate),
        ):
            # assert W_target.shape[-1] >= self.phase.data.shape[-1], f"only support writing in a bigger target matrix, the target matrix is now with shape ({W_target.shape}) while current weight matrix {self.phase.data.shape}."
            parameter_height = round(self.length * self.pixel_size * self.resolution / self.out_downsample_rate)
            parameter_width = round(self.length * self.pixel_size * self.resolution / self.in_downsample_rate)
            ds_rate = W_target.shape[-1] / parameter_width
            # raise ValueError(f"The target matrix should have the same shape ({W_target.shape}) as the current weight matrix {self.phase.data.shape}.")
            W_target_real = F.interpolate(W_target.real.unsqueeze(0).unsqueeze(0), size=(parameter_height, parameter_width), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            W_target_imag = F.interpolate(W_target.imag.unsqueeze(0).unsqueeze(0), size=(parameter_height, parameter_width), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            W_target = W_target_real + 1j * W_target_imag
            W_target = W_target * ds_rate
        if self.TM_model_method == "default":
            W_target_amp = torch.abs(W_target)
            W_target_phase = torch.angle(W_target)
            assert W_target_amp.shape == self.amp.data.shape, f"The target matrix should have the same shape ({W_target_amp.shape}) as the current amp matrix {self.amp.data.shape}."
            assert W_target_phase.shape == self.phase.data.shape, f"The target matrix should have the same shape ({W_target_phase.shape}) as the current phase matrix {self.phase.data.shape}."
            # if not self.fixed_amp or self.test_mode:
            msg = ""
            if not self.fixed_amp:
                # only two situation that we can chagne the value of amp
                # 1. we are not fixing the amp value
                # 2. we are about to test the model
                self.amp.data.copy_(W_target_amp)
                msg += "amp is set to the target matrix, "
            self.phase.data.copy_(W_target_phase)
            msg += "phase is set to the target matrix"
            print(msg, flush=True)
        elif self.TM_model_method == "conv" or self.TM_model_method == "end2end":
            diag_phase = torch.diag(torch.angle(W_target))
            assert diag_phase.shape == self.widths.data.shape, (
                f"The target phase must match the shape of current width matrix. "
                f"Got {diag_phase.shape} vs {self.widths.data.shape}"
            )
            assert self.phase_lut is not None and self.lut_widths is not None, "LUT must be initialized first."

            new_widths = torch.empty_like(diag_phase)

            # Expand dimensions for broadcasting
            phase_lut = self.phase_lut.view(1, -1)                    # (1, N)
            diag_phase_exp = diag_phase.view(-1, 1)                  # (M, 1)

            # Compute L2 distance between target phases and LUT phases
            diff = torch.abs(diag_phase_exp - phase_lut)             # (M, N)
            nearest_indices = torch.argmin(diff, dim=1)              # (M,)

            # Lookup the corresponding width
            new_widths = self.lut_widths[nearest_indices]            # (M,)

            # Set to the model
            self.widths.data.copy_(new_widths)
        elif self.TM_model_method == "fourier_basis":
            W_target_spectrum = torch.fft.fft(W_target, dim=0)
            self.low_freq_component_left.data.copy_(
                W_target_spectrum[
                    :self.low_freq_component_left.shape[0],
                    :
                ]
            )
            self.low_freq_component_right.data.copy_(
                W_target_spectrum[
                    -self.low_freq_component_right.shape[0]:,
                    :
                ]
            )
            if self.high_freq_component is not None:
                self.high_freq_component.data.copy_(
                    W_target_spectrum[
                        38:-34, # this is hardcoded since we have see the relative error
                        :
                    ]
                )
        else:
            raise ValueError(f"TM model method {self.TM_model_method} is not supported, please use default or conv or fourier_basis.")

    def get_convolution_kernel(self) -> torch.Tensor:
        widths = self.widths  # (K,)
        # print("this is the widths:", widths, flush=True)
        amp = torch_interpolate_1d(widths, self.lut_widths, self.amp_lut)
        phase = torch_interpolate_1d(widths, self.lut_widths, self.phase_lut)
        return amp * torch.exp(1j * phase)

    
    def get_transfer_matrix_from_conv(self) -> torch.Tensor:
        """
        Construct a Toeplitz-like transfer matrix that applies the same effect as conv1d
        with zero padding. Output shape: (L, L) complex.
        """
        kernel = self.get_convolution_kernel()  # shape: (K,)
        L = self.length

        tm = torch.zeros((L, L), dtype=kernel.dtype, device=self.device)

        # Fill in diagonals corresponding to the convolution kernel
        # kernel[0] aligns with offset = -pad
        for i in range (L):
            tm[i, i-1: i+2] = kernel[i]

        return tm

    def get_transfer_matrix(self, sharpness: float = None) -> torch.Tensor:
        """
        Build the (L x L) complex matrix W by combining amplitude and angle
        with the im2col'ed identity matrix.
        """
        if self.TM_model_method == "default":
            if not self.test_mode: # in train mode
                amp = self.amp * self.zero_mask
                # print_stat(amp, "stat of the amp before interpolation")
                phase = self.phase * self.zero_mask
                transfer_matrix = amp * torch.exp(1j * phase)
            else:
                amp = self.amp
                phase = self.phase
                transfer_matrix = amp * torch.exp(1j * phase)
            return transfer_matrix
        elif self.TM_model_method == "conv":
            transfer_matrix = self.get_transfer_matrix_from_conv()
            return transfer_matrix
        elif self.TM_model_method == "end2end":
            if self.test_mode:
                # read the tm from the buffer
                transfer_matrix = self.W_buffer
            else:
                transfer_matrix = self.get_transfer_matrix_from_end2end(sharpness)
            return transfer_matrix
        elif self.TM_model_method == "fourier_basis":
            if self.high_freq_component is not None:
                zero_padding_left = torch.zeros(
                    38 - 8,
                    round(self.length * self.pixel_size * self.resolution / self.in_downsample_rate),
                    device=self.device,
                    dtype=torch.complex64
                )
                zero_padding_right = torch.zeros(
                    34 - 7,
                    round(self.length * self.pixel_size * self.resolution / self.in_downsample_rate),
                    device=self.device,
                    dtype=torch.complex64
                )
                transfer_matrix_spectrum = torch.cat(
                    [
                        self.low_freq_component_left,
                        zero_padding_left,
                        self.high_freq_component,
                        zero_padding_right,
                        self.low_freq_component_right
                    ],
                    dim=0
                )
                transfer_matrix = torch.fft.ifft(transfer_matrix_spectrum, dim=0)
            else:
                zero_padding = torch.zeros(
                    round(self.length * self.pixel_size * self.resolution / self.out_downsample_rate) - 8 - 7,
                    round(self.length * self.pixel_size * self.resolution / self.in_downsample_rate),
                    device=self.device,
                    dtype=torch.complex64
                )
                transfer_matrix_spectrum = torch.cat(
                    [
                        self.low_freq_component_left,
                        zero_padding,
                        self.low_freq_component_right
                    ],
                    dim=0
                )
                transfer_matrix = torch.fft.ifft(transfer_matrix_spectrum, dim=0)
            if not self.test_mode:
                transfer_matrix = transfer_matrix * self.zero_mask
            return transfer_matrix
    def forward(self, x: torch.Tensor, sharpness: float = None) -> torch.Tensor:
        """
        x: (batch_size, L) complex
        Returns: (batch_size, L) complex
        """
        if sharpness is not None:
            assert self.TM_model_method == "end2end", "only end2end TM model can use sharpness"
        if self.calculate_in_hr:
            assert round(self.length * self.pixel_size * self.resolution) == x.shape[-1], f"Input shape mismatch, {round(self.length * self.pixel_size * self.resolution)} != {x.shape[-1]}"
        else:
            assert round(self.length * self.pixel_size * self.resolution // self.in_downsample_rate) == x.shape[-1], f"Input shape mismatch, {round(self.length * self.pixel_size * self.resolution // self.in_downsample_rate)} != {x.shape[-1]}"
        # 1) Get the transfer matrix
        W = self.get_transfer_matrix(sharpness)  # => (L, L) complex

        # plt.figure()
        # plt.imshow(W.abs().detach().cpu().numpy(), cmap='viridis')
        # plt.title("Transfer Matrix Magnitude")
        # plt.axis('off')
        # plt.savefig(
        #     f"./figs/end2end_TM_magnitude_sharpness_{sharpness}.png"
        # )
        # project it to unitary matrix using SVD
        # eps = 1e-6
        # U, _, Vh = torch.linalg.svd(W + eps * torch.eye(W.shape[0], device=W.device, dtype=W.dtype))
        # W = torch.matmul(U, Vh)
        self.W = W # record the W for loss calculation, two loss can be added 1. the unitary loss 2. the regulaization loss that limit the W not too far away
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].imshow(W.abs().detach().cpu().numpy(), cmap='viridis')
        # ax[0].set_title("Transfer Matrix Magnitude")
        # ax[0].axis('off')
        # ax[1].imshow(W.angle().detach().cpu().numpy(), cmap='viridis')
        # ax[1].set_title("Transfer Matrix Phase")
        # ax[1].axis('off')
        # plt.savefig(
        #     "./figs/TM_wo_zero_mask_test_mode.png" if self.test_mode else "./figs/TM_w_zero_mask_train_mode.png"
        # )
        # plt.close()

        # plt.figure()
        # plt.imshow(W.abs().detach().cpu().numpy(), cmap='viridis')
        # plt.title("Transfer Matrix Magnitude")
        # plt.axis('off')
        # plt.savefig(
        #     "./figs/transfer_matrix_magnitude.png"
        # )
        # plt.close()

        # plt.figure()
        # plt.imshow(W.angle().detach().cpu().numpy(), cmap='viridis')
        # plt.title("Transfer Matrix Phase")
        # plt.axis('off')
        # plt.savefig(
        #     "./figs/transfer_matrix_phase.png"
        # )
        # plt.close()

        # plt.figure()
        # plt.imshow(self.zero_mask.detach().cpu().numpy(), cmap='viridis')
        # plt.title("Zero Mask")
        # plt.axis('off')
        # plt.savefig(
        #     "./figs/zero_mask.png"
        # )
        # plt.close()


        with torch.no_grad():
            # Make sure shapes match.  Or do .copy_(W) if you don't want to re-allocate.
            self.W_buffer.copy_(W)

        # 2) Multiply: (bs, L) x (L, L)^T => (bs, L)
        if not self.calculate_in_hr:
            out = torch.matmul(
                x.view(-1, round(self.length * self.pixel_size * self.resolution // self.in_downsample_rate)), 
                W.T
            ).view_as(x)
        else:
            # interpolate W into high resolution
            W_real = F.interpolate(W.real.unsqueeze(0).unsqueeze(0), size=(round(self.length * self.pixel_size * self.resolution), round(self.length * self.pixel_size * self.resolution)), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            W_imag = F.interpolate(W.imag.unsqueeze(0).unsqueeze(0), size=(round(self.length * self.pixel_size * self.resolution), round(self.length * self.pixel_size * self.resolution)), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            W = W_real + 1j * W_imag
            W = W / self.in_downsample_rate
            W = W.to(x.dtype)   
            # print_stat(torch.abs(W), "stat of the amp after interpolation")
            out = torch.matmul(
                x.view(-1, round(self.length * self.pixel_size * self.resolution)), 
                W.T
            ).view_as(x)
        return out
    
    def extra_repr(self) -> str:
        return f"length={self.length}, kernel_size={self.kernel_size}, stride={self.stride}"

    def get_transfer_matrix_from_end2end(self, sharpness: float = None) -> torch.Tensor:
        full_wave_down_sample_rate = self.in_downsample_rate
        number_atoms = self.length
        sources = torch.eye(number_atoms * round(15 // full_wave_down_sample_rate), device=self.device)

        sim_key = list(self.total_opt.objective.sims.keys())
        assert len(sim_key) == 1, f"there should be only one sim key, but we got {sim_key}"
        if hasattr(self.total_opt.objective.sims[sim_key[0]].solver, "clear_solver_cache"):
            self.total_opt.objective.sims[sim_key[0]].solver.clear_solver_cache()
            self.total_opt.objective.sims[sim_key[0]].solver.set_cache_mode(True)
        # we first need to run the normalizer
        if self.total_normalizer_list is None or len(self.total_normalizer_list) < number_atoms * round(15 // full_wave_down_sample_rate):
            with torch.no_grad():
                total_normalizer_list = []
                for idx in range(number_atoms * round(15 // full_wave_down_sample_rate)):
                    source_i = sources[idx].repeat_interleave(full_wave_down_sample_rate)
                    source_zero_padding = torch.zeros(int(0.5 * 50), device=self.device)
                    source_i = torch.cat([source_zero_padding, source_i, source_zero_padding])
                    boolean_source_mask = torch.zeros_like(source_i, dtype=torch.bool)
                    boolean_source_mask[torch.where(source_i != 0)] = True
                    custom_source = dict(
                        source=source_i,
                        slice_name="in_slice_1",
                        mode="Hz1",
                        wl=0.85,
                        direction="x+",
                    )

                    weight = -0.05 * torch.ones(self.length * 2 + 1, device=self.device)
                    _ = self.total_opt(
                        sharpness=256, 
                        weight={"design_region_0": weight.unsqueeze(0)},
                        custom_source=custom_source
                    )

                    source_field = self.total_opt.objective.response[('in_slice_1', 'in_slice_1', 0.85, "Hz1", 300)]["fz"].squeeze()
                    total_normalizer_list.append(source_field[boolean_source_mask].mean())
                    if idx == number_atoms * round(15 // full_wave_down_sample_rate) - 1:
                        self.total_normalizer_list = total_normalizer_list
        # now we already have the normalizer, we can run the full wave response
        if hasattr(self.total_opt.objective.sims[sim_key[0]].solver, "clear_solver_cache"):
            self.total_opt.objective.sims[sim_key[0]].solver.clear_solver_cache()
            self.total_opt.objective.sims[sim_key[0]].solver.set_cache_mode(True)
            
        full_wave_response = torch.zeros(
            (
                number_atoms * round(15 // full_wave_down_sample_rate),
                number_atoms * round(15 // full_wave_down_sample_rate),
            ),
            device=self.device, 
            dtype=torch.complex128
        )
        for idx in range(number_atoms * round(15 // full_wave_down_sample_rate)):
            source_i = sources[idx].repeat_interleave(full_wave_down_sample_rate)
            source_zero_padding = torch.zeros(int(0.5 * 50), device=self.device)
            source_i = torch.cat([source_zero_padding, source_i, source_zero_padding])
            boolean_source_mask = torch.zeros_like(source_i, dtype=torch.bool)
            boolean_source_mask[torch.where(source_i != 0)] = True
            custom_source = dict(
                source=source_i,
                slice_name="in_slice_1",
                mode="Hz1",
                wl=0.85,
                direction="x+",
            )
            ls_knots = -0.05 * torch.ones(self.length * 2 + 1, device=self.device)
            ls_knots[1::2] = get_mid_weight(0.05, self.widths.clone())  # Clone to ensure gradient flow
            _ = self.total_opt(
                sharpness=sharpness, 
                weight={"design_region_0": ls_knots.unsqueeze(0)},
                custom_source=custom_source
            )

            response = self.total_opt.objective.response[('in_slice_1', 'nearfield_1', 0.85, "Hz1", 300)]["fz"].squeeze()
            response = response[self.out_downsample_rate // 2 :: self.out_downsample_rate]
            assert len(response) == number_atoms * round(15 // self.out_downsample_rate), f"{len(response)}!= {number_atoms * round(15 // self.out_downsample_rate)}"
            full_wave_response[idx] = response
        full_wave_response = full_wave_response.transpose(0, 1)

        normalizer = torch.stack(self.total_normalizer_list, dim=0).to(self.device)
        normalizer = normalizer.unsqueeze(1)
        full_wave_response = full_wave_response / normalizer

        # if hasattr(self.total_opt.objective.sims[sim_key[0]].solver, "clear_solver_cache"):
        #     self.total_opt.objective.sims[sim_key[0]].solver.clear_solver_cache()
        #     self.total_opt.objective.sims[sim_key[0]].solver.set_cache_mode(False)
        return full_wave_response

class MetaParams(nn.Module):
    path_depth: int
    path_multiplier: int
    unfolding: bool
    delta_z_mode: str
    pixel_size_mode: str
    lambda_mode: str
    rotate_mode: str
    swap_mode: str
    gumbel_mode: str
    enable_identity: bool
    pixel_size_res: int  # nm
    delta_z_res: int  # nm
    delta_z_mask: List[bool]
    delta_z_data: float
    lambda_data: float
    pixel_size_data: float

    def __init__(
        self,
        path_depth: int = 2,
        path_multiplier: int = 2,
        delta_z_mode: str = "fixed",  # fixed, train_share, train, this one is reprogrammable
        pixel_size_mode: str = "fixed",  # fixed, train_share, train, this one is not reprogrammable after fabrication
        lambda_mode: str = "fixed",  # fixed, train_share, train, this one is reprogrammable after fabrication
        rotate_mode: str = "fixed",  # fixed, train, this one is reprogrammable after fabrication
        gumbel_mode: str = "gumbel_soft",  # gumbel_hard, gumbel_soft, softmax, random
        enable_identity: bool = False,  # whether to use identity phase mask, i.e., delta_phi=0, can be learned together with rotation
        swap_mode: str = "fixed",  # fixed, train_stage, train, this one is reprogrammable after fabrication
        pixel_size_res: int = 1,  # nm
        delta_z_res: int = 10,  # nm
        lambda_data: float = 0.850,  # wavelength in um
        delta_z_data: float = 10,  # distance between metasurfaces in um
        ref_lambda: float = 0.850,  # reference wavelength in um
        ref_pixel_size: float = 0.3,  # reference pixel size in um
        gumbel_T: float = 5.0,
        pixel_size_data: float = 0.3,  # pixel size in um
        lambda_train: bool = False,
        delta_z_train: bool = False,
        pixel_size_train: bool = False,
        delta_z_mask: List[bool] = None,
        lambda_res: int = 1,  # um
        device: Device = torch.device("cuda:0"),
    ):
        super().__init__()
        self.path_depth = path_depth
        self.path_multiplier = path_multiplier
        self.gumbel_T = gumbel_T
        assert (
            delta_z_mode
            in {
                "fixed",  # manually defined and fixed to initial value
                "train_share",  # shared delta_z, delta_z is a scalar
                "train_stage",  # each metasurface stage can have different delta_z, we have path_depth values to learn
            }
        ), f"Invalid delta_z_mode: {delta_z_mode}"
        self.set_delta_z_mode(delta_z_mode)

        assert (
            pixel_size_mode
            in {
                "fixed",  # manually defined and fixed to initial value
                "train_share",  # shared pixel_size for all metasurfaces, pixel_size is a scalar
            }
        ), f"Invalid pixel_size_mode: {pixel_size_mode}"
        self.set_pixel_size_mode(pixel_size_mode)

        assert lambda_mode in {
            "fixed",  # manually defined and fixed to initial value
            "train_share",  # shared lambda for all metasurfaces, lambda is a scalar
        }, f"Invalid lambda_mode: {lambda_mode}"
        self.set_lambda_mode(lambda_mode)

        assert rotate_mode in {
            "fixed",  # fixed to initial orientation
            "train",  # each metasurface can rotate
        }, f"Invalid rotate_mode: {rotate_mode}"

        self.set_rotate_mode(rotate_mode)
        self.set_gumbel_mode(gumbel_mode)
        self.enable_identity = enable_identity

        self.pixel_size_res = pixel_size_res
        self.delta_z_res = delta_z_res
        self.lambda_data = lambda_data
        self.delta_z_data = delta_z_data
        self.pixel_size_data = pixel_size_data
        self.lambda_res = lambda_res
        self.ref_lambda = ref_lambda
        self.ref_pixel_size = ref_pixel_size
        self.lambda_train = lambda_train
        self.delta_z_train = delta_z_train
        self.pixel_size_train = pixel_size_train

        self.pixel_size_quantizer = DeviceQuantizer(
            device=device,
            resolution=1,
            mode="nm",
        )

        self.delta_z_quantizer = DeviceQuantizer(
            device=device,
            resolution=1,
            mode="nm",
        )

        self.lambda_quantizer = DeviceQuantizer(
            device=device,
            resolution=1,
            mode="nm",
        )

        self.pixel_lambda_constraint = PixelLambdaConstraint(
            ref_lambda_=self.ref_lambda,
            origin_lambda=self.lambda_data,
            ref_pixel_size=self.ref_pixel_size,
            pixel_size_mode=self.pixel_size_mode,
            device=device,
        )

        ## please note this is to learn how to swap metasurfaces to recontruct new functions for the hardware system
        ## this is DIFFERENT from the ordering to assign channel to path. the assignment ordering is layer-wise learnable weights
        ## this swap (permutation) of metasurface is shared params for the hardware system, not specific each layer.
        assert swap_mode in {
            "fixed",  # fixed to initial order
            "train_stage",  # allow metasurface to swap orders within one stage
            "train",  # allow all metasurfaces to swap locations
        }, f"Invalid swap_mode: {swap_mode}"
        self.set_swap_mode(swap_mode)

        self.device = device
        self.build_parameters()
        self.reset_parameters()

    def build_parameters(self) -> None:
        self.pixel_size = nn.Parameter(
            torch.empty(1, device=self.device),
            requires_grad=self.pixel_size_train,  # learned value 0.5, very similar to the 0.4 value in paper
        )
        self.pixel_size.not_decay = True

        self.delta_z = nn.Parameter(
            torch.empty(self.path_depth, device=self.device),
            requires_grad=self.delta_z_train,  # learned value, very similar to the 8.42 in paper
        )
        self.delta_z.not_decay = True

        self.lambda_ = nn.Parameter(
            torch.empty(1, device=self.device),
            requires_grad=self.lambda_train,  # learned value, very similar to the 0.532 in paper
        )
        self.lambda_.not_decay = True

        self.rotate_gumbel_coeff = nn.Parameter(
            torch.empty(
                self.path_multiplier,
                self.path_depth,
                9 if self.enable_identity else 8,
                device=self.device,
            ),
            requires_grad=self.rotate_mode
            != "fixed",  # use gumbel softmax to learn the rotation orientation, 0, 90, 180, 270 0'' 90'' 180'' 270''
        )
        self.rotate_gumbel_coeff.not_decay = True

        if self.swap_mode == "train_stage":
            self.swap_permutation = nn.Parameter(
                torch.empty(
                    self.path_depth,
                    self.path_multiplier,
                    self.path_multiplier,
                    device=self.device,
                ),
                requires_grad=True,
            )
            self.swap_alm_multiplier = nn.Parameter(
                torch.empty(
                    2, self.path_depth, self.path_multiplier, device=self.device
                ),
                requires_grad=False,
            )
            self.swap_permutation.not_decay = True
            self.swap_alm_multiplier.not_decay = True
        elif self.swap_mode == "train":
            self.swap_permutation = nn.Parameter(
                torch.empty(
                    self.path_depth * self.path_multiplier,
                    self.path_depth * self.path_multiplier,
                    device=self.device,
                ),
                requires_grad=True,
            )
            self.swap_alm_multiplier = nn.Parameter(
                torch.empty(
                    2, self.path_depth * self.path_multiplier, device=self.device
                ),
                requires_grad=False,
            )
            self.swap_permutation.not_decay = True
            self.swap_alm_multiplier.not_decay = True
        else:
            self.swap_permutation = None
            self.swap_alm_multiplier = None

    def reset_parameters(self):
        self.pixel_size.data.fill_(self.pixel_size_data)
        self.delta_z.data.fill_(self.delta_z_data)
        self.lambda_.data.fill_(self.lambda_data)
        self.rotate_gumbel_coeff.data.fill_(0)
        self.set_gumbel_temperature(self.gumbel_T)
        if self.swap_permutation is not None:
            ## noisy identity initialization for permutation matrix
            group_size = self.swap_permutation.shape[-1]
            self.swap_permutation.data.zero_()
            self.swap_permutation.data[
                ..., torch.arange(group_size), torch.arange(group_size)
            ] = 1
            margin = 0.5
            self.swap_permutation.data.mul_(
                margin - (1 - margin) / (group_size - 1)
            ).add_((1 - margin) / (group_size - 1))
            self.swap_permutation.data.add_(
                torch.randn_like(self.swap_permutation.data) * 0.05
            )

        if self.swap_alm_multiplier is not None:
            self.swap_alm_multiplier.data.zero_()

    def set_delta_z_mode(self, mode: bool = True):
        self.delta_z_mode = mode

    def set_pixel_size_mode(self, mode: bool = True):
        self.pixel_size_mode = mode

    def set_lambda_mode(self, mode: bool = True):
        self.lambda_mode = mode

    def set_rotate_mode(self, mode: bool = True):
        self.rotate_mode = mode

    def set_gumbel_temperature(self, T: float = 5.0):
        self.gumbel_T = T

    def set_gumbel_mode(self, mode: str = "gumbel_soft"):
        self.gumbel_mode = mode

    def set_swap_mode(self, mode: bool = True):
        self.swap_mode = mode

    def set_pixel_size_res(self, res: int) -> None:
        self.pixel_size_res = res
        self.pixel_size_quantizer.set_resolution(res)

    def set_delta_z_res(self, res: int) -> None:
        self.delta_z_res = res
        self.delta_z_quantizer.set_resolution(res)

    def set_delta_z_data(self, data: float) -> None:
        self.delta_z_data = data
        self.delta_z.data.fill_(self.delta_z_data)

    def set_lambda_data(self, data: float) -> None:
        self.lambda_data = data
        self.lambda_.data.fill_(self.lambda_data)

    def set_pixel_size_data(self, data: float) -> None:
        self.pixel_size_data = data
        self.pixel_size.data.fill_(self.pixel_size_data)

    def build_pixel_size(self) -> Tensor:
        return self.pixel_size.abs()  # [1] positive only

    def build_delta_z(self, stage: int = 0) -> Tensor:
        if self.delta_z_mode == "train_stage":
            delta_z = self.delta_z[stage]
        elif self.delta_z_mode in {"fixed", "train_share"}:
            delta_z = self.delta_z[0]
        else:
            raise NotImplementedError

        return self.delta_z_quantizer(delta_z.abs())  # [1] positive only

    def build_lambda(self) -> Tensor:
        return self.lambda_.abs()

    def build_constraint_pixel_size_and_lambda(self) -> Tuple[Tensor, Tensor]:
        ## pixel_size and lambda are both positive
        pixel_size = self.build_pixel_size()
        lambda_ = self.build_lambda()

        pixel_size, lambda_ = self.pixel_lambda_constraint(pixel_size, lambda_)

        return self.pixel_size_quantizer(pixel_size), self.lambda_quantizer(lambda_)

    def update_lambda_pixel_size(self) -> None:
        with torch.no_grad():
            pixel_size, lambda_ = self.build_constraint_pixel_size_and_lambda()
            self.pixel_size.data.copy_(pixel_size)
            self.lambda_.data.copy_(lambda_)

    def build_rotate_mask(self, mode=None, batch_size: int = 32) -> Tensor:
        mode = mode or self.gumbel_mode
        logits = self.rotate_gumbel_coeff  # [m, d, 4]
        if mode == "gumbel_hard":
            self.rotate_mask = torch.nn.functional.gumbel_softmax(
                logits,
                tau=self.gumbel_T,
                hard=True,
                dim=-1,
            )
        elif mode == "gumbel_soft":
            self.rotate_mask = torch.nn.functional.gumbel_softmax(
                logits,
                tau=self.gumbel_T,
                hard=False,
                dim=-1,
            )
        elif mode == "gumbel_soft_batch":
            self.rotate_mask = torch.nn.functional.gumbel_softmax(
                torch.log_softmax(logits, dim=-1).unsqueeze(0).repeat(batch_size, 1, 1),
                tau=self.gumbel_T,
                hard=False,
                dim=-1,
            )
        elif mode == "softmax":
            self.rotate_mask = torch.softmax(
                logits / self.gumbel_T,
                dim=-1,
            )
        elif mode == "random":
            logits = torch.ones_like(logits)
            self.rotate_mask = torch.nn.functional.gumbel_softmax(
                logits,
                tau=self.gumbel_T,
                hard=True,
                dim=-1,
            )
        elif mode == "fixed":
            self.rotate_mask = torch.zeros_like(logits)
            self.rotate_mask[..., 0] = 1
        else:
            raise NotImplementedError(f"Invalid gumbel_mode: {mode}")
        return self.rotate_mask  # [m, d, 8] or [m, d, 9]

    def apply_rotate_mask(self, weight: Tensor, rotate_mask: Tensor):
        ## weight is the metasurface phases [m, d, 2, s, s]
        ## rotate_mask is the rotation mask sampled using gubem softmax trick [m, d, 8] or [m, d, 9]
        weight_90 = torch.rot90(weight, 1, [-1, -2])
        weight_180 = torch.rot90(weight, 2, [-1, -2])
        weight_270 = torch.rot90(weight, -1, [-1, -2])
        flipped_weight = torch.transpose(weight, -1, -2)
        flipped_weight_90 = torch.rot90(flipped_weight, 1, [-1, -2])
        flipped_weight_180 = torch.rot90(flipped_weight, 2, [-1, -2])
        flipped_weight_270 = torch.rot90(flipped_weight, -1, [-1, -2])
        if self.enable_identity:
            weight = torch.stack(
                [
                    weight,
                    weight_90,
                    weight_180,
                    weight_270,
                    flipped_weight,
                    flipped_weight_90,
                    flipped_weight_180,
                    flipped_weight_270,
                    torch.zeros_like(weight),
                ],
                dim=-1,
            )
        else:
            weight = torch.stack(
                [
                    weight,
                    weight_90,
                    weight_180,
                    weight_270,
                    flipped_weight,
                    flipped_weight_90,
                    flipped_weight_180,
                    flipped_weight_270,
                ],
                dim=-1,
            )  # [m, d, 1, s, 4]
        # weight = torch.einsum("mdpskr,mdr->mdpsk", weight, rotate_mask)
        weight = einsum(weight, rotate_mask, "m d p s k r, m d r -> m d p s k")

        return weight  # [m,d,1,s]

    def build_swap_permutation(self) -> Tensor:
        ## reparametrization of permutation matrix to relax the constraint
        # abs -> row/col L1-normalization -> projection to legal permutation and stop gradient
        weight = self.swap_permutation.abs()  # W >= 0
        weight = weight / weight.data.sum(dim=-2, keepdim=True)  # Wx1=1 row norm
        weight = weight / weight.data.sum(dim=-1, keepdim=True)  # W^Tx1=1 col norm

        with torch.no_grad():
            perm_loss = (
                weight.data.norm(p=1, dim=-2)
                .sub(weight.data.norm(p=2, dim=-2).square())
                .mean()
                + (1 - weight.data.norm(p=2, dim=-1).square()).mean()
            )
        if perm_loss < 0.05:
            weight = hard_diff_round(
                weight
            )  # W -> P # once it is very close to permutation, it will be trapped and legalized without any gradients.
        return weight

    def apply_swap(self, weight: Tensor):
        ## weight is the metasurface phases [m, d, 2, s, s]
        ## swap_permutation is the permutation matrix [d, m, m] or [dm, dm]
        if self.swap_mode == "fixed":
            return weight
        swap_permutation = self.build_swap_permutation()

        if self.swap_mode == "train_stage":
            # weight = torch.einsum("mdpsk,dnm->ndpsk", weight, swap_permutation)
            weight = einsum(weight, swap_permutation, "m d p s k, d n m -> n d p s k")
        elif self.swap_mode == "train":
            swap_permutation = swap_permutation.view(
                self.path_depth,
                self.path_multiplier,
                self.path_depth,
                self.path_multiplier,
            )
            # weight = torch.einsum("mdpsk,fndm->nfpsk", weight, swap_permutation)
            weight = einsum(weight, swap_permutation, "m d p s k, f n d m -> n f p s k")
        else:
            raise NotImplementedError

        return weight

    def get_swap_loss(self):
        """https://www.math.uci.edu/~jxin/AutoShuffleNet_KDD2020F.pdf"""
        weight = self.build_swap_permutation()
        loss = (
            weight.norm(p=1, dim=-2).sub(weight.norm(p=2, dim=-2).square()).mean()
            + (1 - weight.norm(p=2, dim=-1).square()).mean()
        )
        return loss

    def get_swap_alm_loss(self, rho: float = 0.1):
        if self.swap_mode == "fixed":
            return 0
        ## quadratic tern is also controlled multiplier
        weight = self.build_swap_permutation()  # [d, m, m] or [dm, dm]
        d_weight_r = weight.norm(p=1, dim=-2).sub(
            weight.norm(p=2, dim=-2).square()
        )  # [d, m] or [dm]
        d_weight_c = (
            1 - weight.norm(p=2, dim=-1).square()
        )  # after reparametrization, i.e., row norm -> col norm, col L1-norm is all 1
        # multiplier [2, d, m] or [2, dm]
        loss = self.swap_alm_multiplier[0].flatten().dot(
            (d_weight_r + rho / 2 * d_weight_r.square()).flatten()
        ) + self.swap_alm_multiplier[1].flatten().dot(
            (d_weight_c + rho / 2 * d_weight_c.square()).flatten()
        )
        return loss

    def check_perm(self, indices):
        return tuple(range(len(indices))) == tuple(
            sorted(indices.cpu().numpy().tolist())
        )

    def _get_num_crossings(self, in_indices):
        res = 0
        for idx, i in enumerate(in_indices):
            for j in range(idx + 1, len(in_indices)):
                if i > in_indices[j]:
                    res += 1
        return res

    def unitary_projection(self, w: Tensor, n_step=10, t=0.005, noise_std=0.01):
        w = w.div(t).softmax(dim=-1).round()
        legal_solution = []
        for i in range(n_step):
            u, s, v = w.svd()
            w = u.matmul(v.permute(-1, -2))
            w.add_(torch.randn_like(w) * noise_std)
            w = w.div(t).softmax(dim=-1)
            indices = w.argmax(dim=-1)
            if self.check_perm(indices):
                n_cr = self._get_num_crossings(indices.cpu().numpy().tolist())
                legal_solution.append((n_cr, w.clone().round()))
        legal_solution = sorted(legal_solution, key=lambda x: x[0])
        w = legal_solution[0][1]
        return w

    def update_swap_alm_multiplier(
        self, rho: float = 0.1, max_lambda: Optional[float] = None
    ):
        if self.swap_mode == "fixed":
            return
        with torch.no_grad():
            weight = (
                self.build_swap_permutation().data.detach()
            )  # [d, m, m] or [dm, dm]
            d_weight_r = weight.norm(p=1, dim=-2).sub(
                weight.norm(p=2, dim=-2).square()
            )  # [d, m] or [dm]
            d_weight_c = weight.norm(p=1, dim=-1).sub(
                weight.norm(p=2, dim=-1).square()
            )  # [d, m] or [dm]
            self.swap_alm_multiplier[0].add_(
                rho * (d_weight_r + rho / 2 * d_weight_r.square())
            )
            self.swap_alm_multiplier[1].add_(
                rho * (d_weight_c + rho / 2 * d_weight_c.square())
            )
            if max_lambda is not None:
                self.swap_alm_multiplier.data.clamp_max_(max_lambda)

    def extra_repr(self):
        if self.path_depth is not None:
            s = "path_depth={path_depth}"
        if self.delta_z_mode is not None:
            s += ", delta_z_mode={delta_z_mode}"
        if self.pixel_size_mode is not None:
            s += ", pixel_size_mode={pixel_size_mode}"
        if self.lambda_mode is not None:
            s += ", lambda_mode={lambda_mode}"
        if self.rotate_mode is not None:
            s += ", rotate_mode={rotate_mode}"
        if self.enable_identity is not None:
            s += ", enable_identity={enable_identity}"
        if self.swap_mode is not None:
            s += ", swap_mode={swap_mode}"
        if self.pixel_size_res is not None:
            s += ", pixel_size_res={pixel_size_res}"
        if self.lambda_res is not None:
            s += ", lambda_res={lambda_res}"
        if self.delta_z_res is not None:
            s += ", delta_z_res={delta_z_res}"
        if self.delta_z_data is not None:
            s += ", delta_z_data={delta_z_data}"
        if self.pixel_size_data is not None:
            s += ", pixel_size_data={pixel_size_data}"
        if self.lambda_data is not None:
            s += ", lambda_data={lambda_data}"
        if self.ref_lambda is not None:
            s += ", ref_lambda={ref_lambda}"
        if self.ref_pixel_size is not None:
            s += ", ref_pixel_size={ref_pixel_size}"
        if self.lambda_train is not None:
            s += ", lambda_train={lambda_train}"
        if self.delta_z_train is not None:
            s += ", delta_z_train={delta_z_train}"
        if self.pixel_size_train is not None:
            s += ", pixel_size_train={pixel_size_train}"
        return s.format(**self.__dict__)


class _MetaConv1dMultiPath(Meta_Layer_BASE):
    def __init__(
        self,
        length: int,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        kernel_size_list: list,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        n_pads: int = 5,
        bias: bool = False,
        w_bit: int = 16,
        in_bit: int = 16,
        phase_res: int = 2,  # Number Digits after decimal point
        input_uncertainty: float = 0,
        pad_noise_std: float = 0,
        dpe=None,
        pad_max: float = 1.0,
        sigma_trainable: str = "row_col",
        alpha_train: List[bool] = [True, True],
        mode: str = "phase",
        scale_mode: str = "bilinear",
        weight_train: bool = True,
        skip_meta: bool = False,
        beta_train: bool = False,
        path_multiplier: int = 2,
        path_depth: int = 2,
        unfolding: bool = True,
        enable_xy_pol: bool = True,  # whether to use x/y polarization
        enable_alpha: (
            bool | Tuple[bool, bool]
        ) = True,  # whether to use alpha factor for weighted input channel summation
        enable_beta: bool = True,  # whether to use beta factor as polarization angle for x direction
        encode_mode: str = "mag",  # mag, phase, complex, intensity
        skip_path: bool = False,
        device: Device = torch.device("cuda"),
        verbose: bool = False,
        with_cp: bool = False,
        pac: bool = False,
        metalens_init_file_path: dict = {},
        lambda_data=0.85,
        delta_z_data=4,
        near2far_method="RS",
        in_downsample_rate=1,
        out_downsample_rate=1,
        pixel_size_data=0.3,
        resolution=50,
        max_tm_norm=False,
        calculate_in_hr=False,
        TM_model_method="default",  # default, conv
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            mid_channels=mid_channels,
            out_channels=out_channels,
            n_pads=n_pads,
            kernel_size=kernel_size,
            w_bit=w_bit,
            in_bit=in_bit,
            phase_res=phase_res,
            input_uncertainty=input_uncertainty,
            pad_noise_std=pad_noise_std,
            dpe=dpe,
            pad_max=pad_max,
            mode=mode,
            sigma_trainable=sigma_trainable,
            device=device,
            verbose=verbose,
            with_cp=with_cp,
        )
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.groups = groups
        self.pac = pac
        self.length = length
        self.metalens_init_file_path = metalens_init_file_path
        self.lambda_data = lambda_data
        self.delta_z_data = delta_z_data
        self.near2far_method = near2far_method
        self.near2far_matrix = None
        self.in_downsample_rate = in_downsample_rate
        self.out_downsample_rate = out_downsample_rate
        self.pixel_size_data = pixel_size_data # self.pixel size data is the period of the metalens
        self.resolution = resolution
        assert mode in {"phase", "phase_mag", "mag"}, f"Invalid weight mode: {mode}"
        self.TM_model_method = TM_model_method
        assert TM_model_method in {"default", "conv", "fourier_basis", "end2end"}, f"Invalid TM model method: {TM_model_method}"
        # allocate parameters
        self.weight = None
        self.path_weight = None
        self.x_zero_pad = None
        self.polarization = None

        self.w_bit = w_bit
        self.weight_train = weight_train
        self.sigma_trainable = sigma_trainable
        self.path_multiplier = path_multiplier
        self.alpha_train = [eval(value) if isinstance(value, str) else value for value in alpha_train]
        self.path_depth = path_depth
        self.unfolding = unfolding
        self.beta_train = beta_train
        self.enable_xy_pol = enable_xy_pol
        self.enable_alpha = _pair(enable_alpha)  # [alpha before, alpha after]
        self.enable_beta = enable_beta
        self.encode_mode = encode_mode
        self.skip_path = skip_path
        self.scale_mode = scale_mode
        self.skip_meta = skip_meta
        self.kernel_size_list = kernel_size_list
        self.W = {}
        self.max_tm_norm = max_tm_norm
        self.calculate_in_hr = calculate_in_hr

        if self.enable_alpha[0] and self.groups != self.mid_channels:
            raise ValueError(
                f"When alpha_pre is enabled, depthwise convolution requires groups{groups}=mid_channels{mid_channels}"
            )

        self.alpha_pre_quantizer = WeightQuantizer_LSQ(
            None,
            device=device,
            nbits=self.w_bit,
            offset=False,
            signed=True,
            mode="tensor_wise",
        )

        self.alpha_post_quantizer = WeightQuantizer_LSQ(
            None,
            device=device,
            nbits=self.w_bit,
            offset=False,
            signed=True,
            mode="tensor_wise",
        )
        if self.pac:
            self.build_paconv()
            self.register_buffer(
                "W_buffer",
                torch.zeros(
                    self.path_depth, 
                    round(length * self.pixel_size_data * self.resolution / self.out_downsample_rate), 
                    round(length * self.pixel_size_data * self.resolution / self.in_downsample_rate), 
                    dtype=torch.complex64, 
                    device=device)
            )
        if self.near2far_method == "RS" or self.near2far_method == "green_fn":
            # raise NotImplementedError("RS method is deprecated now")
            if self.calculate_in_hr:
                self.register_buffer(
                    "near2far_buffer",
                    torch.zeros(
                        self.path_depth, 
                        round(length * self.pixel_size_data * self.resolution), 
                        round(length * self.pixel_size_data * self.resolution), 
                        dtype=torch.complex64, 
                        device=device,
                    )
                )
            else:
                self.register_buffer(
                    "near2far_buffer",
                    torch.zeros(
                        self.path_depth, 
                        round(length * self.pixel_size_data * self.resolution / self.out_downsample_rate), 
                        round(length * self.pixel_size_data * self.resolution / self.in_downsample_rate), 
                        dtype=torch.complex64, 
                        device=device,
                    )
                )
        else:
            raise NotImplementedError(f"Invalid near2far_method: {self.near2far_method}")
        self.build_parameters(bias=bias)
        self.reset_parameters()

    def set_test_mode(self, test_mode: bool = True):
        for name, module in self.metalens.items():
            module.set_test_mode(test_mode)

    def set_near2far_matrix(self, near2far_matrix: Tensor):
        self.near2far_matrix = near2far_matrix

    def build_parameters(self, bias: bool) -> None:
        self.meta_params = None
        self.in_channels_flat = self.in_channels // self.groups

        # [m, d, 1, s]
        self.weight = nn.Parameter(
            torch.randn(
                self.path_multiplier,
                self.path_depth,
                2 if self.enable_xy_pol else 1,  # x/y polarization
                *self.kernel_size,
                device=self.device,
                dtype=torch.cfloat if self.mode in {"phase_mag"} else torch.float,
            ),
            requires_grad=self.weight_train,
        )

        """
        For path weight, if use Gumbel Softmax approximation method, the weight for the path should be 
        outc * inc * d * (path_multiplier + 1)
        """
        # Initialize path weights
        self.path_weight = nn.Parameter(
            torch.randn(
                self.path_depth,
                self.path_multiplier,
                self.path_multiplier,
                device=self.device,
            ),
            requires_grad=False,
        )

        self.path_weight.not_decay = True
        if self.enable_alpha[0]:
            self.alpha_pre = nn.Parameter(
                torch.randn(
                    self.mid_channels, self.in_channels, 1, 1, device=self.device
                ),
                requires_grad=self.alpha_train[0],
            )
        else:
            self.alpha_pre = None

        if self.enable_alpha[
            1
        ]:  # alpha is the pointwise convolution weight after metaconv
            self.alpha_post = nn.Parameter(
                torch.randn(
                    self.out_channels, self.mid_channels, 1, 1, device=self.device
                ),
                requires_grad=self.alpha_train[1],
            )
        else:
            self.alpha_post = None

        if (
            self.enable_beta and self.enable_xy_pol
        ):  # beta is the power for x pol, (1-beta) is the power for y pol
            if self.groups == self.mid_channels:  # dwconv
                self.beta = nn.Parameter(
                    torch.randn(self.mid_channels, 1, device=self.device),
                    requires_grad=self.beta_train,
                )
            else:
                self.beta = nn.Parameter(
                    torch.randn(
                        self.out_channels, self.in_channels, device=self.device
                    ),
                    requires_grad=self.beta_train,
                )
            self.beta.not_decay = True
        else:
            self.beta = None

        self.alm_multiplier = nn.Parameter(
            torch.empty(self.path_depth, 2, self.path_multiplier, device=self.device),
            requires_grad=False,
        )

        self.alm_multiplier.not_decay = True

        if bias:
            self.bias = Parameter(torch.zeros(self.out_channels, device=self.device))
        else:
            self.register_parameter("bias", None)

    def build_paconv(self):
        self.metalens = nn.ModuleDict()
        print("this is self.mode that will be used", self.mode, flush=True)
        for n_layer in range(self.path_depth):
            for kernel_size in self.kernel_size_list:
                self.metalens[f"{n_layer}_{kernel_size}"] = CPositionAdaptiveConv1D(
                    length=self.length,
                    kernel_size=kernel_size, 
                    stride=1, 
                    padding=kernel_size // 2, 
                    device=self.device,
                    init_file_path=self.metalens_init_file_path.get(
                        (n_layer, kernel_size), 
                        f"/home/pingchua/projects/MAPS/figs/metalens_TF_uniform_numA-{self.length}_wl-0.85_p-0.3_mat-Si/transfer_matrix.h5"
                    ),
                    fixed_amp = self.mode == "phase",
                    in_downsample_rate=self.in_downsample_rate,
                    out_downsample_rate=self.out_downsample_rate,
                    pixel_size=self.pixel_size_data,
                    resolution=self.resolution,
                    max_tm_norm=self.max_tm_norm,
                    calculate_in_hr=self.calculate_in_hr,
                    TM_model_method=self.TM_model_method,
                    LUT_path="core/metaatom_response_fsdx-0.3.csv",
                )

    def build_path_weight(self):
        # Ensure normalization across each n x n tensor
        path_weight = self.path_weight
        path_weight = path_weight.abs()
        path_weight = path_weight / path_weight.data.sum(
            dim=1, keepdim=True
        )  # Sum over rows
        path_weight = path_weight / path_weight.data.sum(
            dim=2, keepdim=True
        )  # Sum over columns

        with torch.no_grad():
            perm_loss = path_weight.data.norm(p=1, dim=1).sub(
                path_weight.data.norm(p=2, dim=1)
            ).mean(dim=-1) + (1 - path_weight.data.norm(p=2, dim=2)).mean(dim=(-1))

        for i in range(perm_loss.shape[0]):
            if perm_loss[i] < 0.05:
                path_weight[i] = hard_diff_round(path_weight[i])

        return path_weight

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.constant_(self.alm_multiplier.data, 0)
        # nn.init.constant_(self.alpha.data, 0.002)
        # nn.init.constant_(self.beta.data, 1)

    # From 0 to 2pi
    def phase_rounding(self):
        with torch.no_grad():
            # self.weight.copy_((self.weight + torch.pi) % (2 * torch.pi) - torch.pi)
            self.weight.copy_(self.weight % (2 * torch.pi))

    def get_perm_loss(self):
        path_weight = self.build_path_weight()
        loss = path_weight.data.norm(p=1, dim=1).sub(
            path_weight.data.norm(p=2, dim=1)
        ).mean(dim=-1) + (1 - path_weight.data.norm(p=2, dim=2)).mean(dim=(-1))
        return loss

    def get_alm_perm_loss(self, rho: float = 0.1):
        ## quadratic tern is also controlled multiplier
        path_weight = self.build_path_weight()
        d_path_weight_r = path_weight.norm(p=1, dim=1).sub(path_weight.norm(p=2, dim=1))
        # d_weight_c = weight.norm(p=1, dim=1).sub(weight.norm(p=2, dim=1))
        d_path_weight_c = 1 - path_weight.norm(p=2, dim=2)
        loss = torch.zeros(path_weight.shape[0])
        d_path_weight_r_square = d_path_weight_r.square()
        d_path_weight_c_square = d_path_weight_c.square()

        for i in range(path_weight.shape[0]):
            loss_r = self.alm_multiplier[i, 0].dot(
                d_path_weight_r[i] + rho / 2 * d_path_weight_r_square[i]
            )
            loss_c = self.alm_multiplier[i, 1].dot(
                d_path_weight_c[i] + rho / 2 * d_path_weight_c_square[i]
            )
            loss[i] = loss_r + loss_c

        return loss

    def update_alm_multiplier(
        self, rho: float = 0.1, max_lambda: Optional[float] = None
    ):
        with torch.no_grad():
            path_weight = self.build_path_weight().detach()
            d_path_weight_r = path_weight.norm(p=1, dim=1).sub(
                path_weight.norm(p=2, dim=1)
            )
            d_path_weight_c = path_weight.norm(p=1, dim=2).sub(
                path_weight.norm(p=2, dim=2)
            )
            d_path_weight_r_square = d_path_weight_r.square()
            d_path_weight_c_square = d_path_weight_c.square()
            for i in range(path_weight.shape[0]):
                self.alm_multiplier[i, 0].add_(
                    d_path_weight_r[i] + rho / 2 * d_path_weight_r_square[i]
                )
                self.alm_multiplier[i, 1].add_(
                    d_path_weight_c[i] + rho / 2 * d_path_weight_c_square[i]
                )
            if max_lambda is not None:
                self.alm_multiplier.data.clamp_max_(max_lambda)

    def path_generation(self, path_weight):
        path_before_transpose = torch.argmax(path_weight, dim=-1)
        path_after_transpose = torch.transpose(path_before_transpose, 0, 1)
        full_repetitions, remainder = divmod(
            self.in_channels, path_after_transpose.size(0)
        )
        repeated_a = path_after_transpose.repeat(full_repetitions, 1)
        if remainder:
            repeated_a = torch.cat(
                (repeated_a, path_after_transpose[:remainder]), dim=0
            )

        return repeated_a.unsqueeze(0).expand(self.out_channels, -1, -1)

    def get_smoothing_loss(self, lambda_smooth=1e-3) -> torch.Tensor:
        # print("Here2")
        """
        Compute a smoothness penalty on the layer's weight.
        Returns a scalar tensor that is the sum (or mean) of the penalties
        across all depths.
        """
        # Suppose self.weight has shape [m, d, pol, s]
        # or [outC, inC, d, s], etc. We'll just be generic below.

        # You can loop over each depth or flatten it out. Here's a loop version:
        total_penalty = 0.0
        for depth_idx in range(self.path_depth):
            # Extract the slice for this depth. For example:
            # weight_d = self.weight[..., depth_idx, :] if depth is in the second-to-last dim
            # or weight_d = self.weight[:, :, depth_idx] if depth is in the 2nd dimension, etc.
            weight_d = self.weight[:, depth_idx]  # or adapt indexing to your actual shape
            
            # Now do a 1D "TV" or finite-diff:
            # E.g., sum of |w[i+1] - w[i]| across the spatial dimension
            diff = (weight_d[..., 1:] - weight_d[..., :-1])
            penalty_this_depth = diff.mean()  # or sum, etc.

            total_penalty += penalty_this_depth

        # Scale by lambda_smooth
        total_penalty = lambda_smooth * total_penalty
        # print("Here3")
        # print(total_penalty)
        return total_penalty

    def build_initial_path(self):
        # [[0000],[1111], [2222]]
        # Generate a tensor of size (m, 1) ranging from 0 to m-1
        row_values = torch.arange(self.path_multiplier).view(self.path_multiplier, 1)

        # Expand the tensor to size (m, n) by repeating the columns
        init_path = row_values.expand(-1, self.path_depth)

        full_repetitions, remainder = divmod(self.mid_channels, init_path.size(0))
        repeated_a = init_path.repeat(full_repetitions, 1)
        if remainder:
            repeated_a = torch.cat((repeated_a, init_path[:remainder]), dim=0)

        # repeated_a [inc, d]
        if self.groups == self.mid_channels:  # dwconv
            repeated_a = repeated_a.unsqueeze(1)  # [inc, 1, d]
        else:
            repeated_a = repeated_a.unsqueeze(0).expand(
                self.out_channels, -1, -1
            )  # [outc,inc,d]
        return repeated_a

    def build_alpha(self) -> Tensor:
        alpha_pre = self.alpha_pre
        alpha_post = self.alpha_post
        return self.alpha_pre_quantizer(alpha_pre), self.alpha_post_quantizer(
            alpha_post
        )

    def build_beta(self) -> Tensor:
        beta = self.beta
        if beta is not None:
            beta = torch.sigmoid(beta)[
                ..., None, None
            ]  # beta must be positive from 0 to 1
        # beta = torch.clamp(beta, min=0, max=1) # Add sigmoid or tanh
        return beta  # [outc, inc//group, 1, 1]

    def build_weight(self) -> Tensor:
        paths = self.build_initial_path()

        phase, mag = super()._weight  # [m, d, 1, s]
        if self.mode == "phase":
            phase = torch.exp(1j * phase)  # phase: becomes exp(j*phase)
            weight = phase
            # [m, d, 1, s]
        elif self.mode == "phase_mag":
            weight = mag * torch.exp(1j * phase)
        elif self.mode == "mag":
            weight = mag
        else:
            raise NotImplementedError(f"Invalid weight mode: {self.mode}")

        weight = weight[
            paths, torch.arange(weight.shape[1])
        ]  # [outc, inc, d, 1, s] complex

        return weight

    @lru_cache(maxsize=32)
    def construct_coordinate_diff(self, k):
        # For a 1D line of length k, the 'positions' array is [k].
        x = torch.arange(k, device=self.device, dtype=torch.float32)
        # coord_diff will then be shape [k, k].
        coord_diff = x[:, None] - x[None, :]  # Broadcasting differences
        return coord_diff

    def encode_x_information(self, input: Tensor) -> Tensor:
        # assert self.encode_mode == "phase", f"only support phase encoding, but got {self.encode_mode}"
        if self.encode_mode == "intensity":
            input = input.sqrt()
        elif self.encode_mode == "phase":
            input = torch.exp(1j * input)
        elif self.encode_mode == "complex":
            amplitude = torch.abs(input)
            phase = torch.angle(input)
            input = amplitude * torch.exp(1j * phase)
        elif self.encode_mode == "mag":
            return input
        else:
            raise NotImplementedError(f"Invalid encode_mode: {self.encode_mode}")
        return input

    def build_diffraction_matrix(self, stage: int = 0) -> Tensor:
        assert self.out_downsample_rate == self.in_downsample_rate, "downsample rate should be the same when using RS method"
        # k = self.kernel_size[0]
        k = round(self.kernel_size[0] * self.pixel_size_data * self.resolution / self.out_downsample_rate)
        # 1D coordinate differences
        coord_diff = self.construct_coordinate_diff(k)
        
        # pixel_size, lambda_ = self.meta_params.build_constraint_pixel_size_and_lambda()
        pixel_size = 1 / self.resolution * self.out_downsample_rate
        lambda_ = self.lambda_data
        
        # Convert 1D differences to physical distances in the x direction
        delta_x = pixel_size * coord_diff  # shape [k, k]
        
        # The z-spacing (distance to next diffractive layer)
        delta_z = self.meta_params.build_delta_z(stage)  # scalar
        wave_vec = 2 * np.pi / lambda_

        # In 1D, each distance is sqrt( delta_x^2 + delta_z^2 )
        squared_distances = delta_x**2 + delta_z**2
        distances_efficient = torch.sqrt(squared_distances)

        # Replace pixel_size^2 with pixel_size because we have a line, not an area
        # You may also wish to change the rest of the formula depending on your model.
        # For example, one common 1D approximation is:
        #
        #   transfer ~ (pixel_size / (2π)) * ( delta_z / r^2 ) * (...) * e^{i*k*r}
        #
        # but it depends on how you derived your 2D formula. For simplicity, we just
        # directly replace pixel_size^2 -> pixel_size below:

        self.transfer_matrix = (
            (pixel_size / (2 * np.pi))
            * (delta_z / squared_distances)
            * (1 / distances_efficient - wave_vec * 1j)
            * torch.exp(1j * wave_vec * distances_efficient)
        )
        
        return self.transfer_matrix

    # def build_near2far_tools(self):
    #     num_meta_atom = self.length
    #     res = self.resolution
    #     atom_period = self.pixel_size_data 
    #     pillar_material = "Si"
    #     near_field_dx = 0.3

    #     sim_cfg = DefaultSimulationConfig()

    #     sim_cfg.update(
    #                 dict(
    #                     solver="ceviche_torch",
    #                     numerical_solver="solve_direct",
    #                     use_autodiff=False,
    #                     neural_solver=None,
    #                     border_width=[0, 0, 0.5, 0.5],
    #                     PML=[0.5, 0.5],
    #                     resolution=res,
    #                     wl_cen=self.lambda_data,
    #                     plot_root="figs/dummy_plot",
    #                 )
    #             )
    
    #     metalens = MetaLens(
    #         material_bg="Air",
    #         material_r = pillar_material,
    #         material_sub="SiO2",
    #         sim_cfg=sim_cfg,
    #         aperture=atom_period * num_meta_atom,
    #         port_len=(1, 2),
    #         port_width=(atom_period * num_meta_atom, atom_period),
    #         substrate_depth=0,
    #         ridge_height_max=0.75,
    #         nearfield_dx=near_field_dx,
    #         nearfield_size=atom_period * num_meta_atom,
    #         farfield_dxs=((near_field_dx + self.delta_z_data, near_field_dx + self.delta_z_data + 2 / res),),
    #         farfield_sizes=(atom_period * num_meta_atom,),
    #         device=self.device,
    #     )
    #     hr_metalens = metalens.copy(resolution=200)
    #     design_region_param_cfgs = dict(
    #         rho_resolution=[0, 2/atom_period],
    #     )
    #     obj_cfgs = dict(
    #         near_field_response_record=dict(
    #             wl=[self.lambda_data],
    #         ),
    #     )
    #     self.opt = MetaLensOptimization(
    #             device=metalens,
    #             design_region_param_cfgs=design_region_param_cfgs,
    #             hr_device=hr_metalens,
    #             sim_cfg=sim_cfg,
    #             obj_cfgs=obj_cfgs,
    #             operation_device=self.device,
    #         ).to(self.device)

    #     eps = torch.ones((1, 480), device=self.device)
    #     self.fx_fy_calculator = fdfd_hz(
    #         omega=2 * np.pi * C_0 / (self.lambda_data * 1e-6),
    #         dL=1 / 50 * 1e-6,
    #         eps_r=eps,
    #         npml=[0, 0],
    #         power=1e-8,
    #         bloch_phases=None,
    #         neural_solver=None,
    #         numerical_solver="solve_direct",
    #         use_autodiff=False,
    #         sym_precond=True,
    #     )

    #     eps_vec = eps.flatten()
    #     entries_a, indices_a, eps_matrix, eps_vec_xx, eps_vec_yy = self.fx_fy_calculator._make_A(eps_vec)
    #     self.eps_vec_xx = eps_vec_xx
    #     self.eps_vec_yy = eps_vec_yy

    def scale(self, x, size, mode="bilinear"):
        current_size = x.shape[-2:]

        if mode == "padding":
            pad_h = size[0] - current_size[0]
            pad_w = size[1] - current_size[1]
            if pad_h > 0 or pad_w > 0:
                pad_h1 = pad_h // 2
                pad_h2 = pad_h - pad_h1
                pad_w1 = pad_w // 2
                pad_w2 = pad_w - pad_w1
                padding = (pad_w1, pad_w2, pad_h1, pad_h2)
                if torch.is_complex(x):
                    x = torch.complex(
                        F.pad(x.real, padding, mode="constant", value=0),
                        F.pad(x.imag, padding, mode="constant", value=0),
                    )
                else:
                    x = F.pad(x, padding, mode="constant", value=0)

        else:
            if current_size != size:
                if torch.is_complex(x):
                    x = torch.complex(
                        F.interpolate(x.real, size=size, mode=mode),
                        F.interpolate(x.imag, size=size, mode=mode),
                    )
                else:
                    x = F.interpolate(x, size=size, mode=mode)

        return x

    def _forward_impl(self, x: tuple) -> Tensor:
        # modulation
        # x: [bs, inc, h, w] real
        # x :[bs, inc, 2, h, w]
        # weight = self.build_weight()  # [outc, inc/g, d, 2, s, s]
        # beta = self.build_beta()  # [outc, inc/g]
        # alpha_pre, alpha_post = self.build_alpha()  # [outc, inc]
        # assert self.calculate_in_hr, f"we only support calculate_in_hr=True now"
        x, sharpness = x
        inner_field = []
        source_mask = x[..., 1].to(torch.bool)
        x = x[..., 0]
        bs, inC, L = x.shape
        if inC != self.out_channels:
            raise ValueError(
                f"1D metaline for simplicity assumes inC == outC, but got inC={inC}, outC={self.out_channels}."
            )
        
        eps = 1e-8
        # meanval = x.square().mean(dim=[-1], keepdim=True).add(eps).sqrt()  # shape [bs, inC, 1]
        # x = x / meanval  # [bs, inC, L]
        
        # print(x.shape)
        
        x = self.encode_x_information(x)
        x[~source_mask] = 0
        if not self.calculate_in_hr:
            x_real = F.interpolate(x.real, size=round(self.length * self.pixel_size_data * self.resolution / self.out_downsample_rate), mode="linear")
            x_imag = F.interpolate(x.imag, size=round(self.length * self.pixel_size_data * self.resolution / self.out_downsample_rate), mode="linear")
            x = x_real + 1j * x_imag

        # weight = self.build_weight()
        
        # [bs, inc, s] -> [bs, 1, inc, 1, s]
        x = x.unsqueeze(1).unsqueeze(3)
        # check if there is nan in x
        W_buffer = torch.zeros(
            self.path_depth, 
            round(self.length * self.pixel_size_data * self.resolution / self.out_downsample_rate), 
            round(self.length * self.pixel_size_data * self.resolution / self.in_downsample_rate), 
            dtype=torch.complex64, 
            device=self.device
        )
        equivalent_W = torch.zeros_like(W_buffer, dtype=torch.complex64, device=self.device)
        if self.near2far_method == "RS" or self.near2far_method == "green_fn":
            if self.calculate_in_hr:
                near2far_buffer = torch.zeros(
                    self.path_depth, 
                    round(self.length * self.pixel_size_data * self.resolution), 
                    round(self.length * self.pixel_size_data * self.resolution), 
                    dtype=torch.complex64, 
                    device=self.device,
                )
            else:
                near2far_buffer = torch.zeros(
                    self.path_depth, 
                    round(self.length * self.pixel_size_data * self.resolution / self.in_downsample_rate), 
                    round(self.length * self.pixel_size_data * self.resolution / self.out_downsample_rate), 
                    dtype=torch.complex64, 
                    device=self.device,
                )
        for i in range(self.path_depth):
            # weight: [1, 1, 2, 1, 32]
            # kernel = weight[None, :, :, i]
            # final_kernel_size = kernel.shape[-1] 
            # final_kernel_size = round(kernel.shape[-1] * self.pixel_size_data * self.resolution // self.out_downsample_rate)
            if self.calculate_in_hr:
                final_kernel_size = round(self.length * self.pixel_size_data * self.resolution)
            else:
                final_kernel_size = round(self.length * self.pixel_size_data * self.resolution / self.out_downsample_rate)

            if i == 0:
                # we first do the diffraction
                if self.near2far_method == "RS":
                    raise NotImplementedError("RS method is deprecated now")
                    transfer_matrix = self.build_diffraction_matrix(stage=i) # [s, s]
                    near2far_buffer[i] = transfer_matrix
                    x = torch.matmul(x.view(-1, final_kernel_size), transfer_matrix).view_as(x) # [bs, outc, inc, 1, s]
                elif self.near2far_method == "green_fn":
                    assert self.near2far_matrix is not None, "near2far_matrix should be set before using green_fn method, please probe and set it"
                    near2far_buffer[i] = self.near2far_matrix
                    x = torch.matmul(x.view(-1, final_kernel_size), self.near2far_matrix.T).view_as(x)
                else:
                    raise NotImplementedError(f"Invalid near2far_method: {self.near2far_method}")
                inner_field.append(x)
            # if not self.pac:
            #     x = x * kernel # [bs, outc, inc, 1, s]
            # else:
            lens_response = []
            W_buffer_i = torch.zeros(
                round(self.length * self.pixel_size_data * self.resolution / self.out_downsample_rate), 
                round(self.length * self.pixel_size_data * self.resolution / self.in_downsample_rate), 
                dtype=torch.complex64, 
                device=self.device
            )
            equivalent_W_i = torch.zeros_like(W_buffer_i, dtype=torch.complex64, device=self.device)
            for key, metalens in self.metalens.items():
                n_layer, kernel_size = key.split("_")
                n_layer = int(n_layer)
                if n_layer != i:
                    continue
                kernel_size = int(kernel_size)
                lens_response.append(metalens(x, sharpness))
                W_buffer_i = W_buffer_i + metalens.W_buffer
                equivalent_W_i = equivalent_W_i + metalens.W
            lens_response = torch.cat(lens_response, dim=2)
            x = torch.sum(lens_response, dim=2, keepdim=True)
            W_buffer[i] = W_buffer_i
            equivalent_W[i] = equivalent_W_i
            if self.near2far_method == "RS":
                raise NotImplementedError("RS method is deprecated now")
                transfer_matrix = self.build_diffraction_matrix(stage=i) # [s, s]
                near2far_buffer[i] = transfer_matrix
                x = torch.matmul(x.view(-1, final_kernel_size), transfer_matrix).view_as(x) # [bs, outc, inc, 1, s]
            elif self.near2far_method == "green_fn":
                assert self.near2far_matrix is not None, "near2far_matrix should be set before using green_fn method, please probe and set it"
                near2far_buffer[i] = self.near2far_matrix
                x = torch.matmul(x.view(-1, final_kernel_size), self.near2far_matrix.T).view_as(x)
                # need to first interpolate the x to the size of the near2far tool
                # and then use the near2far tool to calculate Ex and Ey
                # and then use the near2far tool to calculate the farfield Hz component
                # x_shape = x.shape
                # target_length = self.out_downsample_rate * x.shape[-1]
                # assert target_length == 480, f"expected target_length=480, but got {target_length}"
                # print("this is x shape before green function near 2 far", x.shape, flush=True)
                # x = x.flatten(start_dim=0, end_dim=2)
                # print("this is the shape of x after flatten", x.shape, flush=True)
                # x_real = F.interpolate(x.real, size=target_length, mode="linear")
                # x_imag = F.interpolate(x.imag, size=target_length, mode="linear")
                # x = x_real + 1j * x_imag
                # x = x.flatten(end_dim=-1)
                # Hz_vec = x.flatten(-1)
                # Ex_vec, Ey_vec = self.fx_fy_calculator._Hz_to_Ex_Ey(Hz_vec, self.eps_vec_xx, self.eps_vec_yy)
                # x = get_farfields_GreenFunction(
                #     nearfield_slices=[
                #         self.opt.objective.port_slices["nearfield_1"]
                #     ],
                #     nearfield_slices_info=[
                #         self.opt.objective.port_slices_info["nearfield_1"]
                #     ],
                #     Fz=Hz_vec.unsqueeze(-1),
                #     Fx=Ex_vec.unsqueeze(-1),
                #     Fy=Ey_vec.unsqueeze(-1),
                #     farfield_x=None,
                #     farfield_slice_info=self.opt.objective.port_slices_info["farfield_1"],
                #     freqs=torch.tensor([1 / self.lambda_data], device=Hz_vec.device),
                #     eps=1,
                #     mu=MU_0,
                #     dL=self.opt.objective.grid_step,
                #     component="Hz",
                #     decimation_factor=1,
                #     passing_slice=True,
                # )
                # x = x["Hz"][..., 0]
                # x = x[..., self.in_downsample_rate // 2::self.in_downsample_rate]
                # print("this is the shape of x after green function near 2 far", x.shape, flush=True)
                # quit()
            # x = x / torch.max(x.abs())
            inner_field.append(x)
            print_stat(x, f"meta{i}: ", DEBUG)
        self.equivalent_W = equivalent_W
        if self.pac:
            with torch.no_grad():
                self.W_buffer.copy_(W_buffer)
        if self.near2far_method == "RS" or self.near2far_method == "green_fn":
            with torch.no_grad():
                self.near2far_buffer.copy_(near2far_buffer)
        x = x.transpose(
                2, 3
            )  # [bs, outc, 1, inc, s] 
        
        # Photodetector
        x = x.real.square() + x.imag.square()
        
        # [bs, outc, inc, s] 
        x = x.squeeze(2)
        # print(x.shape)
        # exit(0)
        # [bs, outc, s] 
        x = x.sum(dim=2)
        # print(x.shape)
        if self.bias is not None:
            x = x + self.bias[None, :, None]
        # print(x.shape)

        return (x, inner_field[:-1])  # [bs, outc, h, w]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


@MODELS.register_module()
class MetaConv1dETE(Meta_Layer_BASE):
    _conv_types = _MetaConv1dMultiPath

    __annotations__ = {"bias": Optional[torch.Tensor]}

    _in_channels: int
    mid_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]
    mode: str
    path_depth: int
    path_multiplier: int
    unfolding: bool
    delta_z_mode: str
    pixel_size_mode: str
    lambda_mode: str
    weight_train: bool
    encode_mode: str
    skip_path: bool
    enable_xy_pol: bool
    enable_alpha: bool
    enable_beta: bool
    rotate_mode: str
    swap_mode: str
    gumbel_mode: str
    enable_identity: bool

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        kernel_size_list: list,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        n_pads: int = 5,
        bias: bool = True,
        mid_channels: int = 0,
        w_bit: int = 16,
        in_bit: int = 16,
        pixel_size_res: int = 1,  # nm
        delta_z_res: int = 10,  # nm
        phase_res: int = 1,  # degree
        # constant scaling factor from intensity to detected voltages
        input_uncertainty: float = 0,
        pad_noise_std: float = 0,
        dpe=None,
        pad_max: float = 1.0,
        sigma_trainable: str = "row_col",
        mode: str = "phase",
        path_multiplier: int = 1,
        path_depth: int = 2,
        unfolding: bool = False,
        enable_xy_pol: bool = True,  # whether to use x/y polarization
        enable_alpha: bool = True,  # whether to use alpha factor for weighted input channel summation
        enable_beta: bool = True,  # whether to use beta factor as polarization angle for x direction
        encode_mode: str = "mag",  # mag, phase, complex, intensity
        alpha_train: List[bool] = [True, True],
        skip_path: bool = False,
        delta_z_data: float = 10,  # um
        lambda_data: float = 0.850,  # um
        pixel_size_data: float = 0.3,  # um
        gumbel_T: float = 5.0,
        lambda_res: int = 1,
        ref_lambda: float = 0.850,
        ref_pixel_size: float = 0.3,
        lambda_train: bool = False,
        pixel_size_train: bool = False,
        delta_z_train: bool = False,
        beta_train: bool = False,
        gumbel_decay_rate: float = 0.956,
        skip_meta: bool = False,
        delta_z_mode: str = "fixed",  # fixed, train_share, train, this one is reprogrammable
        pixel_size_mode: str = "fixed",  # fixed, train_share, train, this one is not reprogrammable after fabrication
        lambda_mode: str = "fixed",  # fixed, train_share, train, this one is reprogrammable after fabrication
        rotate_mode: str = "fixed",  # fixed, train, this one is reprogrammable after fabrication
        gumbel_mode: str = "fixed",  # gumbel_hard, gumbel_soft, softmax, random, fixed
        scale_mode: str = "bilinear",  # bilinear, nearest, area, bicubic, lanczos, padding
        weight_train: bool = True,  #
        enable_identity: bool = False,  # whether to use identity phase mask, i.e., delta_phi=0, can be learned together with rotation
        swap_mode: str = "fixed",  # fixed, train_stage, train, this one is reprogrammable after fabrication
        device: Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose: bool = False,
        with_cp: bool = False,
        pac: bool = False,
        length: int = 32,
        metalens_init_file_path: dict = {},
        near2far_method: str = "RS",
        in_downsample_rate: int = 1,
        out_downsample_rate: int = 1,
        resolution: int = 50,
        max_tm_norm: bool = False,
        calculate_in_hr: bool = False,
        TM_model_method: str = "default",
    ) -> None:
        if mid_channels == 0:
            mid_channels = in_channels
        super().__init__(
            in_channels=in_channels,
            mid_channels=mid_channels,
            out_channels=out_channels,
            n_pads=n_pads,
            kernel_size=kernel_size,
            w_bit=w_bit,
            in_bit=in_bit,
            pixel_size_res=pixel_size_res,
            delta_z_res=delta_z_res,
            phase_res=phase_res,
            input_uncertainty=input_uncertainty,
            pad_noise_std=pad_noise_std,
            dpe=dpe,
            pad_max=pad_max,
            mode=mode,
            scale_mode=scale_mode,
            path_multiplier=path_multiplier,
            path_deth=path_depth,
            unfolding=unfolding,
            weight_train=weight_train,
            enable_alpha=enable_alpha,
            enable_beta=enable_beta,
            enable_xy_pol=enable_xy_pol,
            encode_mode=encode_mode,
            skip_path=skip_path,
            delta_z_mode=delta_z_mode,
            pixel_size_mode=pixel_size_mode,
            lambda_mode=lambda_mode,
            rotate_mode=rotate_mode,
            swap_mode=swap_mode,
            gumbel_mode=gumbel_mode,
            enable_identity=enable_identity,
            device=device,
            verbose=verbose,
            with_cp=with_cp,
        )
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.groups = groups
        self.path_depth = path_depth
        self.path_multiplier = path_multiplier
        self.delta_z_data = delta_z_data
        self.gumbel_T = gumbel_T
        self.gumbel_decay_rate = gumbel_decay_rate
        self.lambda_data = lambda_data
        self.pixel_size_data = pixel_size_data
        self.skip_meta = skip_meta
        self.lambda_res = lambda_res
        self.ref_lambda = ref_lambda
        self.ref_pixel_size = ref_pixel_size
        self.lambda_train = lambda_train
        self.pixel_size_train = pixel_size_train
        self.delta_z_train = delta_z_train
        self.alpha_train = alpha_train
        self.scale_mode = scale_mode
        self.mid_channels = mid_channels
        self.in_channels_pos = self.in_channels
        self.in_channels_neg = 0 if unfolding else self.in_channels
        self.length = length
        self._conv_pos = _MetaConv1dMultiPath(
            length=self.length,
            in_channels=self.in_channels_pos,
            mid_channels=self.mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            kernel_size_list=kernel_size_list,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            n_pads=n_pads,
            bias=False,
            w_bit=w_bit,
            in_bit=in_bit,
            phase_res=phase_res,
            weight_train=weight_train,
            input_uncertainty=input_uncertainty,
            pad_noise_std=pad_noise_std,
            dpe=dpe,
            pad_max=pad_max,
            sigma_trainable=sigma_trainable,
            mode=mode,
            beta_train=beta_train,
            path_multiplier=path_multiplier,
            path_depth=path_depth,
            alpha_train=alpha_train,
            unfolding=unfolding,
            enable_xy_pol=enable_xy_pol,  # whether to use x/y polarization
            enable_alpha=enable_alpha,  # whether to use alpha factor for weighted input channel summation
            enable_beta=enable_beta,  # whether to use beta factor as polarization angle for x direction
            encode_mode=encode_mode,  # whether to encode phase information
            skip_path=skip_path,
            skip_meta=skip_meta,
            scale_mode=scale_mode,
            device=device,
            verbose=verbose,
            with_cp=with_cp,
            pac=pac,
            metalens_init_file_path=metalens_init_file_path,
            lambda_data=lambda_data,
            delta_z_data=delta_z_data,
            near2far_method=near2far_method,
            in_downsample_rate=in_downsample_rate,
            out_downsample_rate=out_downsample_rate,
            pixel_size_data=pixel_size_data,
            resolution=resolution,
            max_tm_norm=max_tm_norm,
            calculate_in_hr=calculate_in_hr,
            TM_model_method=TM_model_method,
        )
        self._conv_neg = None
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels).to(self.device))
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()

    def requires_grad_Meta(self, mode: bool = True):
        self._requires_grad_Meta = mode
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.requires_grad_Meta(mode)

    def get_perm_loss(self):
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.get_perm_loss()

    def get_smoothing_loss(self, smoothing_lambda=1e-3):
        # print("here1")
        for m in self.modules():
            if isinstance(m, self._conv_types):
                return(m.get_smoothing_loss(smoothing_lambda))
    
    def get_alm_perm_loss(self):
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.get_alm_perm_loss()

    def set_input_er(self, er: float = 0, x_max: float = 6.0) -> None:
        ## extinction ratio of input modulator
        self.input_er = er
        self.input_max = x_max
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.set_input_er(er, x_max)

    def set_input_snr(self, snr: float = 0) -> None:
        self.input_snr = snr
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.set_input_snr(snr)

    def set_detection_snr(self, snr: float = 0) -> None:
        self.detection_snr = snr
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.set_detection_snr(snr)

    def set_weight_train(self, flag: bool = True) -> None:
        self.weight_train = flag
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.weight_train = flag

    def phase_rounding(self) -> None:
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.phase_rounding()

    def set_alpha_train(self, flag: List = [True, True]) -> None:
        self.alpha_train = flag
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.alpha_train = flag

    def set_skip_path(self, flag: bool = True) -> None:
        self.skip_path = flag
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.skip_path = flag

    def set_skip_meta(self, flag: bool = False) -> None:
        self.skip_meta = flag
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.skip_meta = flag

    def set_scale_mode(self, mode: str = "bilinear") -> None:
        self.scale_mode = mode
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.scale_mode = mode

    def set_encode_mode(self, mode: str = "mag") -> None:
        self.encode_mode = mode
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.encode_mode = mode

    @property
    def _weight(self):
        # control pads to complex transfer matrix
        # [p, q, n_pads] real -> [p, q, k, k] complex
        weights = []
        for m in self.modules():
            if isinstance(m, self._conv_types):
                weights.append(m._build_weight())
        return weights

    @property
    def _weight_unroll(self):
        weights = []
        for m in self.modules():
            if isinstance(m, self._conv_types):
                weights.append(m.build_weight_unroll(m._build_weight()))
        return weights

    @property
    def _weight_complex(self):
        weights = []
        for m in self.modules():
            if isinstance(m, self._conv_types):
                weights.append(m.build_weight(m._build_weight()))
        return weights

    def _forward_impl(self, x):
        y = self._conv_pos(x) # y is now a tuple, containing the light collected in the receiver as well as the inner field injected into the metasurfaces
        if self._conv_neg is not None:
            y_neg = self._conv_neg(x)
            y = y - y_neg

        if self.bias is not None:
            y = y + self.bias[None, :, None]
        return y # y is a tuple.

    def get_output_dim(self, img_height: int, img_width: int) -> Tuple[int, int]:
        h_out = (img_height - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[
            0
        ] + 1
        w_out = (img_width - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[
            1
        ] + 1
        return (int(h_out), int(w_out))

    def forward(self, x):
        assert isinstance(x, tuple), "input x should be a tuple"
        if self.in_bit <= 8:
            x = self.input_quantizer(x)
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self._forward_impl, x)
        else:
            out = self._forward_impl(x)

        return out # now this out is a tuple, containing the light collected in the receiver as well as the inner field injected into the metasurfaces
    
    def set_test_mode(self, test_mode: bool = True):
        self._conv_pos.set_test_mode(test_mode)

    def set_near2far_matrix(self, near2far_matrix: Tensor):
        self._conv_pos.set_near2far_matrix(near2far_matrix)

    def extra_repr(self):
        s = (
            "{in_channels}, {mid_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups is not None:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.mode is not None:
            s += ", mode={mode}"
        if self.path_depth is not None:
            s += ", path_depth={path_depth}"
        if self.path_multiplier is not None:
            s += ", path_multiplier={path_multiplier}"
        if self.enable_xy_pol is not None:
            s += ", xy_pol={enable_xy_pol}"
        if self.enable_alpha is not None:
            s += ", alpha={enable_alpha}"
        if self.enable_beta is not None:
            s += ", beta={enable_beta}"
        if self.skip_path is not None:
            s += ", skip_path={skip_path}"
        if self.weight_train is not None:
            s += ", weight_train={weight_train}"
        if self.encode_mode is not None:
            s += ", encode_mode={encode_mode}"
        if self.scale_mode is not None:
            s += ", scale_mode={scale_mode}"
        if self.skip_meta is not None:
            s += ", skip_meta={skip_meta}"
        if self.alpha_train is not None:
            s += ", alpha_train={alpha_train}"
        return s.format(**self.__dict__)
        