"""
Date: 2024-10-05 02:02:33
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-02-23 10:34:04
FilePath: /MAPS/core/invdes/models/layers/parametrization/levelset.py
"""

from functools import lru_cache
from typing import Tuple

import h5py
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.types import Device

from .base_parametrization import BaseParametrization
from .utils import HeavisideProjection

__all__ = ["GratingWidthParameterization"]

class GratingWidthParameterization(BaseParametrization):
    def __init__(
        self,
        *args,
        cfgs: dict = dict(
            method="grating_width",
            # rho_resolution=[50, 0],  #  50 knots per um, 0 means reduced dimention
            binary_projection=dict(
                fw_threshold=100,
                bw_threshold=100,
                mode="regular",
            ),
            transform=[
                dict(
                    type="mirror_symmetry",  # Mirror symmetry
                    dims=[],  # Symmetry dimensions
                ),
                dict(type="transpose_symmetry", flag=False),  # Transpose symmetry
            ],
            init_method="random",
            denorm_mode="linear_eps",
        ),
        **kwargs,
    ):
        super().__init__(*args, cfgs=cfgs, **kwargs)

        method = cfgs["method"]

        self.register_parameter_build_per_region_fn(
            method, self._build_parameters_grating_width
        )
        self.register_parameter_reset_per_region_fn(
            method, self._reset_parameters_grating_width
        )

        self.build_parameters(cfgs, self.design_region_cfg)
        self.reset_parameters(cfgs, self.design_region_cfg)
        self.binary_projection = HeavisideProjection(**self.cfgs["binary_projection"])
        self.eta = torch.tensor(
            [
                0.5,
            ],
            device=self.operation_device,
        )

    @lru_cache(maxsize=3)
    def _prepare_parameters_grating_width(
        self,
        period: float,
        region_size: Tuple[float, float],
        grating_dir: str = "y",
    ):
        if grating_dir == "x":
            num_atom = round(region_size[0] / period)
        elif grating_dir == "y":
            num_atom = round(region_size[1] / period)
        else:
            raise ValueError(f"Unsupported grating direction: {grating_dir}")

        n_lr = [(m.stop - m.start) for m in self.design_region_mask]

        n_hr = [(m.stop - m.start) for m in self.hr_design_region_mask]

        param_dict = dict(
            n_atom=num_atom,
            period=period,
            region_size=region_size,
            grating_dir=grating_dir,
            n_lr=n_lr,
            n_hr=n_hr,
        )
        return param_dict
        n_rho = [
            int(region_s * res) + 1
            for region_s, res in zip(region_size, rho_resolution)
        ]
        ### this makes sure n_phi is the same as design_region_mask
        ## add 1 here due to leveset needs to have one more point than the design region
        n_phi = [(m.stop - m.start) for m in self.design_region_mask]

        n_hr_phi = [(m.stop - m.start) for m in self.hr_design_region_mask]

        rho = [
            torch.linspace(-region_s / 2, region_s / 2, n, device=self.operation_device)
            for region_s, n in zip(region_size, n_rho)
        ]
        ## if one dimension has rho_resolution=0, then this dimension need to be duplicated, e.g., ridge
        ## then all n_phi points need to be the same number, which is -region_s/2
        phi = [
            torch.linspace(
                -region_s / 2,
                region_s / 2 if rho_res > 0 else -region_s / 2,
                n,
                device=self.operation_device,
            )
            for region_s, n, rho_res in zip(region_size, n_phi, rho_resolution)
        ]

        ## if one dimension has rho_resolution=0, then this dimension need to be duplicated, e.g., ridge
        ## then all n_phi points need to be the same number, which is -region_s/2
        hr_phi = [
            torch.linspace(
                -region_s / 2,
                region_s / 2 if rho_res > 0 else -region_s / 2,
                n,
                device=self.operation_device,
            )
            for region_s, n, rho_res in zip(region_size, n_hr_phi, rho_resolution)
        ]

        param_dict = dict(
            n_rho=n_rho, n_phi=n_phi, rho=rho, phi=phi, n_hr_phi=n_hr_phi, hr_phi=hr_phi
        )

        return param_dict

    def _build_parameters_grating_width(self, param_cfg, region_cfg):
        param_dict = self._prepare_parameters_grating_width(
            # tuple(param_cfg["rho_resolution"]),
            param_cfg["period"],
            tuple(region_cfg["size"]),
            param_cfg["grating_dir"],
        )
        n_atom = param_dict["n_atom"]
        widths = nn.Parameter(
            0.05 * torch.ones(n_atom, device=self.operation_device)
        )
        weight_dict = dict(widths=widths)
        return weight_dict, param_dict

    def _reset_parameters_grating_width(
        self, weight_dict, param_cfg, region_cfg, init_method: str = "random"
    ):
        init_file_path = param_cfg.get("initialization_file", None)
        if init_file_path is not None:
            pass
        if init_method == "random":
            nn.init.normal_(weight_dict["widths"], mean=0.05, std=0.01)
        elif init_method == "uniform":
            nn.init.uniform_(weight_dict["widths"], a=0.0, b=0.1)
        elif init_method == "constant":
            nn.init.constant_(weight_dict["widths"], val=0.05)
        else:
            pass
            # raise ValueError(f"Unsupported initialization method: {init_method}")
        
    def _build_permittivity(
        self, weights, n_atom, period, region_size, grating_dir, n_lr, n_hr, customize_widths=None
    ):
        if customize_widths is not None:
            assert customize_widths.shape == weights["widths"].shape, (
                f"the shape of customize_widths {customize_widths.shape} should be the same as the shape of ls_knots in weights {weights['widths'].shape}"
            )
            assert len(customize_widths) == n_atom, (
                f"the length of customize_widths {len(customize_widths)} should be the same as the length of ls_knots in weights {n_atom}"
            )
        # sigma = getattr(self.cfgs, "sigma", 1 / max(self.cfgs["rho_resolution"]))
        # interpolation = getattr(self.cfgs, "interpolation", "gaussian")
        design_param = weights["widths"] if customize_widths is None else customize_widths
        ### to avoid all knots becoming unreasonable large to make it stable
        ### also to avoid most knots concentrating near threshold, otherwise, binarization will not work

        pillar_placer = PillarPlacer(
            width = design_param,
            n_atom = n_atom,
            period = period,
            region_size = region_size,
            grating_dir = grating_dir,
            n_lr = n_lr,
            n_hr = n_hr,
            res = self.hr_device.resolution,
        )

        return pillar_placer.forward() # {0, 1}

        phi_model = LevelSetInterp(
            x0=rho[0],
            y0=rho[1],
            z0=design_param,
            sigma=sigma,
            interpolation=interpolation,
            device=design_param.device,
        )
        phi = phi_model.get_ls(x1=phi[0], y1=phi[1], shape=n_phi)  # [76, 2001]

        ## This is used to constrain the value to be [0, 1] for heaviside input
        phi = torch.tanh(phi) * 0.5
        phi = phi.to(self.operation_device)
        phi = phi + self.eta
        eps_phi = self.binary_projection(phi, sharpness, self.eta)

        self.phi = torch.reshape(phi, n_phi)

        return eps_phi

    def build_permittivity(self, weights, sharpness, customize_widths=None):
        ## this is the high resolution, e.g., res=200, 310 permittivity
        ## return:
        #   1. we need the first one for gds dump out
        #   2. we need the second one for evaluation, do not need to downsample it here. transform will handle it.
        
        hr_permittivity = self._build_permittivity(
            weights,
            self.params["n_atom"],
            self.params["period"],
            self.params["region_size"],
            self.params["grating_dir"],
            self.params["n_lr"],
            self.params["n_hr"],
            customize_widths=customize_widths,
        )

        return hr_permittivity
    
class Width2Eps(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        width,
        n_atom,
        period,
        region_size,
        grating_dir,
        n_lr,
        n_hr,
        res,
    ):
        ctx.width = width
        ctx.n_atom = n_atom
        ctx.period = period
        ctx.region_size = region_size
        ctx.grating_dir = grating_dir
        ctx.n_lr = n_lr
        ctx.n_hr = n_hr

        device = width.device

        size = n_hr[0] if grating_dir == "x" else n_hr[1]
        eps = torch.zeros(size, device=device)

        atom_width_px = (width * res).round().long()
        atom_centers = (((torch.arange(n_atom, device=device) + 0.5) * period) * res).round().long()

        # Generate positions tensor
        positions = torch.arange(size, device=device).unsqueeze(0)  # shape: [1, size]

        start = (atom_centers - atom_width_px // 2).unsqueeze(1)  # shape: [n_atom, 1]
        end = (atom_centers + atom_width_px // 2).unsqueeze(1)    # shape: [n_atom, 1]

        mask = (positions >= start) & (positions < end)  # shape: [n_atom, size]
        eps = mask.any(dim=0).float()

        ctx.save_for_backward(mask)

        return eps

    @staticmethod
    def backward(ctx, grad):
        (mask,) = ctx.saved_tensors
        # mask shape: [n_atom, size], grad shape: [size]
        grad_width = (mask * grad.unsqueeze(0)).sum(dim=1)

        return grad_width, None, None, None, None, None, None, None
    
# class Width2Eps(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         width,
#         n_atom,
#         period,
#         region_size,
#         grating_dir,
#         n_lr,
#         n_hr,
#     ):
#         ctx.width = width
#         ctx.n_atom = n_atom
#         ctx.period = period
#         ctx.region_size = region_size
#         ctx.grating_dir = grating_dir
#         ctx.n_lr = n_lr
#         ctx.n_hr = n_hr
#         assert round(n_hr[0] / region_size[0]) == round(n_hr[1] / region_size[1]), (
#             f"n_hr {n_hr} should be proportional to region_size {region_size}"
#         )
#         res = round(n_hr[0] / region_size[0])
#         device = width.device
#         if grating_dir == "x":
#             eps = torch.zeros(n_hr[0], device=device)
#         elif grating_dir == "y":
#             eps = torch.zeros(n_hr[1], device=device)
#         for idx in range(n_atom):
#             atom_width = width[idx]
#             atom_width_px = atom_width * res
#             atom_center = ((idx + 0.5) * period) * res
#             eps[
#                 atom_center - atom_width_px // 2 : atom_center + atom_width_px // 2
#             ] = 1
#         return eps

#     @staticmethod
#     def backward(ctx, grad):
#         width = ctx.width
#         n_atom = ctx.n_atom
#         period = ctx.period
#         region_size = ctx.region_size
#         grating_dir = ctx.grating_dir
#         n_lr = ctx.n_lr
#         n_hr = ctx.n_hr
#         res = round(n_hr[0] / region_size[0])
#         grad_width = torch.zeros_like(width)

#         for idx in range(n_atom):
#             atom_width_px = (width[idx] * res).round().long()
#             atom_center = ((idx + 0.5) * period * res).round().long()
#             start = atom_center - atom_width_px // 2
#             end = atom_center + atom_width_px // 2
#             grad_width[idx] = grad[start:end].sum()

#         return grad_width, None, None, None, None, None, None

class PillarPlacer(torch.nn.Module):
    def __init__(
            self,
            width,
            n_atom,
            period,
            region_size,
            grating_dir,
            n_lr,
            n_hr,
            res,
        ):
        super(PillarPlacer, self).__init__()
        self.width = width
        self.n_atom = n_atom
        self.period = period
        self.region_size = region_size
        self.grating_dir = grating_dir
        self.n_lr = n_lr
        self.n_hr = n_hr
        self.res = res

    def forward(
        self,
    ):
        line_eps = Width2Eps.apply(
            self.width,
            self.n_atom,
            self.period,
            self.region_size,
            self.grating_dir,
            self.n_lr,
            self.n_hr,
            self.res,
        )
        # print("this is the n_hr", self.n_hr, flush=True)
        # quit()
        if self.grating_dir == "y":
            eps = line_eps.unsqueeze(0).expand(self.n_hr[1], -1)
        elif self.grating_dir == "x":
            eps = line_eps.unsqueeze(1).expand(-1, self.n_hr[0])
        return eps