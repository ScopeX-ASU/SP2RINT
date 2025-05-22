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
from torch.nn.modules.utils import _pair
from torch.types import Device

from core.models.layers.meta_layer_base import Meta_Layer_BASE
from core.models.layers.utils import hard_diff_round

from .utils import DeviceQuantizer, PixelLambdaConstraint, WeightQuantizer_LSQ

__all__ = ["MetaParams", "MetaConv2d"]

DEBUG = False


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
        lambda_data: float = 0.532,  # wavelength in um
        delta_z_data: float = 8.42,  # distance between metasurfaces in um
        ref_lambda: float = 0.532,  # reference wavelength in um
        ref_pixel_size: float = 0.3,  # reference pixel size in um
        gumbel_T: float = 5.0,
        pixel_size_data: float = 0.5,  # pixel size in um
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
            resolution=1000,
            mode="nm",
        )

        self.delta_z_quantizer = DeviceQuantizer(
            device=device,
            resolution=1000,
            mode="nm",
        )

        self.lambda_quantizer = DeviceQuantizer(
            device=device,
            resolution=1000,
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
        return self.rotate_mask  # [m, d, 4]

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
            )  # [m, d, 2, s, s, 4]
        # weight = torch.einsum("mdpskr,mdr->mdpsk", weight, rotate_mask)
        weight = einsum(weight, rotate_mask, "m d p s k r, m d r -> m d p s k")

        return weight  # [m,d,2,s,s]

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


class _MetaConv2dMultiPath(Meta_Layer_BASE):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
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
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        assert mode in {"phase", "phase_mag", "mag"}, f"Invalid weight mode: {mode}"

        # allocate parameters
        self.weight = None
        self.path_weight = None
        self.x_zero_pad = None
        self.polarization = None

        self.w_bit = w_bit
        self.weight_train = weight_train
        self.sigma_trainable = sigma_trainable
        self.path_multiplier = path_multiplier
        self.alpha_train = alpha_train
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

        self.build_parameters(bias=bias)
        self.reset_parameters()

    def build_parameters(self, bias: bool) -> None:
        self.meta_params = None
        self.in_channels_flat = self.in_channels // self.groups

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

    def phase_rounding(self):
        with torch.no_grad():
            self.weight.copy_((self.weight + torch.pi) % (2 * torch.pi) - torch.pi)

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

        weight = super()._weight  # [m, d, s, s, 2] real, [outc, inc, d] long
        if self.mode == "phase":
            weight = torch.exp(1j * weight)  # phase: becomes exp(j*phase)
            # [m, d, s, s]
        elif self.mode == "phase_mag":
            pass
        elif self.mode == "mag":
            weight = weight.abs()  # positive only for magnitude modulation
        else:
            raise NotImplementedError(f"Invalid weight mode: {self.mode}")

        weight = weight[
            paths, torch.arange(weight.shape[1])
        ]  # [outc, inc, d, s, s] complex

        return weight

    @lru_cache(maxsize=32)
    def construct_coordinate_diff(self, k):
        # Generate a grid of x and y coordinates
        x = torch.arange(k, device=self.device)
        y = torch.arange(k, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

        # Flatten the grid to create a list of positions
        positions = torch.stack(
            [grid_x.flatten(), grid_y.flatten()], dim=1
        )  # Convert to float for distance calculations

        coord_diff = positions[:, None, :] - positions[None, :, :]  # [k,k,2]
        return coord_diff

    def encode_x_information(self, input: Tensor) -> Tensor:
        if self.encode_mode == "intensity":
            input = input.sqrt()
        elif self.encode_mode == "phase":
            input = torch.exp(1j * input)
        elif self.encode_mode == "complex":
            amplitude = input.sqrt()
            input = amplitude * torch.exp(1j * input)
        elif self.encode_mode == "mag":
            return input
        else:
            raise NotImplementedError(f"Invalid encode_mode: {self.encode_mode}")
        return input

    def build_diffraction_matrix(self, stage: int = 0) -> Tensor:
        k = self.kernel_size[0]
        # Coordinates for the first matrix
        coord_diff = self.construct_coordinate_diff(k)

        # Compute the differences in positions using broadcasting
        # Notice this is meta-atoms pixel positions, not actual distance.
        # we define meta-atom pixel dimension is 0.1 um, keep th same unit as wavelength 1.55 um

        pixel_size, lambda_ = self.meta_params.build_constraint_pixel_size_and_lambda()

        delta_xy = pixel_size * coord_diff  # Resulting shape will be [k, k, 2]
        delta_z = self.meta_params.build_delta_z(stage)  # um
        wave_vec = 2 * np.pi / lambda_
        # Calculate the squared differences in the x and y coordinates, summing over the last dimension, and add 1 for the z-distance squared
        squared_distances = delta_xy.square().sum(2) + delta_z.square()

        # Take the square root to get the Euclidean distances
        distances_efficient = torch.sqrt(squared_distances)

        self.transfer_matrix = (
            (pixel_size**2 / (2 * np.pi))
            * (delta_z / squared_distances)
            * (1 / distances_efficient - wave_vec * 1j)
            * (torch.exp(1j * wave_vec * distances_efficient))
        )
        return self.transfer_matrix

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

    def _forward_impl(self, x: Tensor) -> Tensor:
        # modulation
        # x: [bs, inc, h, w] real
        # x :[bs, inc, 2, h, w]
        weight = self.build_weight()  # [outc, inc/g, d, 2, s, s]
        beta = self.build_beta()  # [outc, inc/g]
        alpha_pre, alpha_post = self.build_alpha()  # [outc, inc]

        ## match input channels to mid_channels
        if self.enable_alpha[0] and alpha_pre is not None:
            x = torch.nn.functional.conv2d(x, alpha_pre, stride=1, padding=0, bias=None)

        else:
            if self.mid_channels > self.in_channels:
                ## need to duplicate input channels to match mid_channels
                full_repetitions, remainder = divmod(
                    self.mid_channels, self.in_channels
                )

                repeated_x = x.repeat(1, full_repetitions, 1, 1)
                if remainder:
                    repeated_x = torch.cat((repeated_x, x[:, :remainder]), dim=1)
                x = repeated_x

            elif self.mid_channels < self.in_channels:
                raise NotImplementedError(
                    f"Invalid mid_channels: {self.mid_channels} < in_channels({self.in_channels}), but alpha_pre is None"
                )

        # each metasurface is (M, F)
        # (F), M, F, M, F, M, F, total d masks for each image
        #      ----  ----  ----

        # first scale input size to match metasurface size via zoom lens
        if not self.skip_meta:
            ## RMS Norm here to always make sure input images to metasurfaces are normalized
            x = x / x.square().mean(dim=[-2, -1], keepdim=True).add(1e-8).sqrt()

            print_stat(x, "rmsnorm: ", DEBUG)
            input_size = x.shape[-2:]

            x = self.scale(x, self.kernel_size, mode=self.scale_mode)
            x = self.encode_x_information(x)

            if self.groups == self.mid_channels:  # dwconv
                # [bs, inc, s, s] -> [bs, inc, 1, 1, s, s]
                x = x.unsqueeze(2).unsqueeze(3)
            else:  # normal conv
                # [bs, inc, s, s] -> [bs, 1, inc, 1, s, s]
                x = x.unsqueeze(1).unsqueeze(3)

            # Expand only if xy_pol and beta are enabled,
            # thus [bs, 1, inc, 1, s, s] -> [bs, 1, inc, 2, s, s] or
            # [bs, 1, 1, 1, s, s] -> [bs, 1, 1, 2, s, s] for dwconv
            # Otherwise remain previous shape
            if self.enable_xy_pol and self.enable_beta:
                x = x.expand(-1, -1, -1, 2, -1, -1)

            if self.skip_path:
                # [bs, 1, inc, 2, s, s] or [bs, 1, 1, 2, s, s] for dwconv
                x_skip = x
            else:
                x_skip = None

            for i in range(self.path_depth):
                kernel = weight[None, :, :, i]
                x = x * kernel
                transfer_matrix = self.build_diffraction_matrix(stage=i)
                # [bs, outc, inc, 2, s, s] or [bs, inc, 1, 2, s, s] for dwconv
                x = torch.matmul(x.flatten(-2, -1), transfer_matrix).view_as(x)
                print_stat(x, f"meta{i}: ", DEBUG)

            # rescale it back via zoom lens
            x = x.transpose(
                2, 3
            )  # [bs, outc, 2, inc, s, s] or [bs, inc, 2, 1, s, s] for dwconv

            # # [bs, 1, 2, inc, s, s] or [bs, 1, 2, 1, s, s] for dwconv
            x = (
                (x + x_skip.transpose(2, 3).expand(-1, x.shape[1], -1, -1, -1, -1))
                if x_skip is not None
                else x
            )

            # [bs, outc, 2, inc, h, w] or [bs, inc, 2, 1, h, w] for dwconv
            x = self.scale(x.flatten(0, 2), input_size).reshape(
                *(x.shape[:-2]), *input_size
            )
            x = x.real.square() + x.imag.square()

            print_stat(x, "PD: ", DEBUG)

            if self.enable_xy_pol and beta is not None:
                # x [bs, outc, 2, inc, h, w] or [bs, inc, 2, 1, h, w] for dwconv
                # beta [outc, inc, 1, 1] or [inc, 1, 1, 1] for dwconv
                # result [bs, outc, inc, h, w] or [bs, inc, 1, h, w] for dwconv
                x = x[:, :, 0] * beta - x[:, :, 1] * (1 - beta)
            else:
                # [bs, outc, inc, h, w] or [bs, inc, 1, h, w] for dwconv
                x = x.squeeze(2)
            print_stat(x, "beta: ", DEBUG)

        else:
            x = x.unsqueeze(2)

        ### match mid_channels to out_channels
        if self.enable_alpha[1] and alpha_post is not None:
            # alpha [outc, inc, 1, 1]
            x = torch.nn.functional.conv2d(
                x.squeeze(2), alpha_post, stride=1, padding=0, bias=self.bias
            )
        else:
            x = x.sum(dim=1)  # [bs, outc, h, w]
            if self.bias is not None:
                x = x + self.bias[None, :, None, None]
        print_stat(x, "alpha_post: ", DEBUG)

        return x  # [bs, outc, h, w]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


@MODELS.register_module()
class MetaConv2d(Meta_Layer_BASE):
    _conv_types = _MetaConv2dMultiPath

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
        path_multiplier: int = 2,
        path_depth: int = 2,
        unfolding: bool = False,
        enable_xy_pol: bool = True,  # whether to use x/y polarization
        enable_alpha: bool = True,  # whether to use alpha factor for weighted input channel summation
        enable_beta: bool = True,  # whether to use beta factor as polarization angle for x direction
        encode_mode: str = "mag",  # mag, phase, complex, intensity
        alpha_train: List[bool] = [True, True],
        skip_path: bool = False,
        delta_z_data: float = 8.42,  # um
        lambda_data: float = 0.532,  # um
        pixel_size_data: float = 0.5,  # um
        gumbel_T: float = 5.0,
        lambda_res: int = 1,
        ref_lambda: float = 0.532,
        ref_pixel_size: float = 0.5,
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

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
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
        self._conv_pos = _MetaConv2dMultiPath(
            in_channels=self.in_channels_pos,
            mid_channels=self.mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
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
        y = self._conv_pos(x)
        if self._conv_neg is not None:
            y_neg = self._conv_neg(x)
            y = y - y_neg

        if self.bias is not None:
            y = y + self.bias[None, :, None, None]
        return y

    def get_output_dim(self, img_height: int, img_width: int) -> Tuple[int, int]:
        h_out = (img_height - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[
            0
        ] + 1
        w_out = (img_width - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[
            1
        ] + 1
        return (int(h_out), int(w_out))

    def forward(self, x):
        if self.in_bit <= 8:
            x = self.input_quantizer(x)
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self._forward_impl, x)
        else:
            out = self._forward_impl(x)

        return out

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
