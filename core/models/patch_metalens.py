import torch
from torch import nn
import os
import sys
# sys.path.insert(
#     0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../core"))
# )
# print("\n".join(sys.path))
import torch

from thirdparty.MAPS_old.core.invdes.models import (
    MetaLensOptimization,
)
from thirdparty.MAPS_old.core.invdes.models.base_optimization import DefaultSimulationConfig
from thirdparty.MAPS_old.core.invdes.models.layers import MetaLens
from thirdparty.MAPS_old.core.utils import set_torch_deterministic, SharpnessScheduler
from pyutils.config import Config
import h5py
from thirdparty.MAPS_old.core.invdes import builder
from thirdparty.MAPS_old.core.utils import print_stat
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import random
from pyutils.general import ensure_dir
from core.utils import DeterministicCtx, get_mid_weight, get_terminal_weight
import time
import torch
import torch.nn.functional as F
import h5py
import copy
from thirdparty.MAPS_old.core.invdes.optimizer.nesterov import NesterovAcceleratedGradientOptimizer

def interpolate_1d(input_tensor, x0, x1, method="linear"):
    """
    Perform 1D interpolation on a tensor.
    
    Args:
        input_tensor (torch.Tensor): 1D tensor of shape (N,)
        x0 (torch.Tensor): Original positions of shape (N,)
        x1 (torch.Tensor): Target positions of shape (M,)
        method (str): Interpolation method ("linear" or "gaussian").
    
    Returns:
        torch.Tensor: Interpolated tensor of shape (M,)
    """
    if method == "linear":
        # linear interpolation
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        return F.interpolate(input_tensor, size=x1.shape[0], mode="linear", align_corners=False).squeeze()
    elif method == "gaussian":
        sigma = 0.1
        dist_sq = (x1.reshape(-1, 1) - x0.reshape(1, -1)).square().to(input_tensor.device)
        weights = (-dist_sq / (2 * sigma ** 2)).exp()
        weights = weights / weights.sum(dim=1, keepdim=True)
        return weights @ input_tensor
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")

def find_closest_width(lut, target_phase):
    """
    Given a LUT (dictionary) where keys are widths and values are phase shifts,
    find the width whose phase shift is closest to the target phase.

    Parameters:
    lut (dict): Dictionary with widths as keys and phase shifts as values.
    target_phase (float): The desired phase shift.

    Returns:
    float: The width corresponding to the closest phase shift.
    """
    closest_width = min(lut, key=lambda w: abs(lut[w] - target_phase))
    return closest_width

class PatchMetalens(nn.Module):
    def __init__(
        self,
        atom_period: float,
        patch_size: int,
        num_atom: int,
        probing_region_size: int,
        target_phase_response: torch.Tensor,
        LUT: dict = None,
        device: torch.device = torch.device("cuda:0"),
        target_dx: float = 0.3,
        plot_root: str = "./figs/patched_metalens",
        downsample_mode: str = "both",
        downsample_method: str = "point",
        dz: float = 4.0,
        param_method: str = "level_set",
        tm_norm: str = "max",
        field_norm_condition: str = "w_lens",
        opts: dict = None,
        total_opt: MetaLensOptimization = None,
        design_var_type: str = "width", # width or height
        atom_width: float = 0.12,
    ):
        super(PatchMetalens, self).__init__()
        self.atom_period = atom_period
        self.patch_size = patch_size
        self.num_atom = num_atom
        self.probing_region_size = probing_region_size
        self.target_phase_response = target_phase_response # this is used to initialize the metalens
        self.LUT = LUT
        self.device = device
        self.target_dx = target_dx
        self.plot_root = plot_root
        self.downsample_mode = downsample_mode
        assert downsample_method in ["point", "avg"], f"Unsupported downsample_method: {downsample_method}"
        self.downsample_method = downsample_method
        self.dz = dz
        self.param_method = param_method
        self.tm_norm = tm_norm
        self.field_norm_condition = field_norm_condition
        assert self.field_norm_condition in ["w_lens", "wo_lens"], f"Unsupported field_norm_condition: {self.field_norm_condition}"
        self.normalizer_list = []
        self.total_normalizer_list = []
        self.design_var_type = design_var_type
        self.atom_width = atom_width
        if param_method == "level_set":
            if self.design_var_type == "width":
                self.design_region_param_cfgs = {}
            elif self.design_var_type == "height":
                self.design_region_param_cfgs = {
                    "rho_resolution": [1 / 0.75, 0]
                }
            else:
                raise ValueError(f"Unsupported design_var_type: {self.design_var_type}")
        elif param_method == "grating_width":
            self.design_region_param_cfgs = {
                "method": "grating_width",
                "transform": [],
                "init_method": "constant",
                "denorm_mode": "linear_eps",
                "period": atom_period,
                "grating_dir": "y",
            }
        else:
            raise ValueError(f"Unsupported param_method: {param_method}")
        self.build_param()
        if opts is None:
            assert total_opt is None, "total_opt must be set if opts is None"
            self.build_patch()
        else:
            self.opt = opts
            self.total_opt = total_opt

    def set_total_normalizer_list(self, normalizer_list):
        self.total_normalizer_list = normalizer_list

    def set_LUT(self, LUT):
        self.LUT = LUT

    def set_target_phase_response(self, target_phase_response):
        if target_phase_response.shape[-1] != self.num_atom or target_phase_response.shape[-2] != self.num_atom:
            assert target_phase_response.shape[-1] % self.num_atom == 0, "The last dimension of target_phase_response must be a multiple of num_atom"
            assert target_phase_response.shape[-2] % self.num_atom == 0, "The second last dimension of target_phase_response must be a multiple of num_atom"
            x_downsample_rate = target_phase_response.shape[-1] // self.num_atom
            y_downsample_rate = target_phase_response.shape[-2] // self.num_atom
            target_phase_response = target_phase_response[
                ...,
                y_downsample_rate // 2::y_downsample_rate,
                x_downsample_rate // 2::x_downsample_rate,
            ]

        self.target_phase_response = target_phase_response

    def rebuild_param(self):
        assert self.LUT is not None and self.target_phase_response is not None, "LUT and target_phase_response must be set before rebuilding parameters"
        if self.param_method == "level_set":
            for i in range(self.num_atom):
                if self.design_var_type == "width":
                    self.weights.data[i] = get_mid_weight(0.05, find_closest_width(self.LUT, self.target_phase_response[i, i].item()))
                elif self.design_var_type == "height":
                    self.weights.data[i] = get_terminal_weight(0.05, find_closest_width(self.LUT, self.target_phase_response[i, i].item()))
                else:
                    raise ValueError(f"Unsupported design_var_type: {self.design_var_type}")
        elif self.param_method == "grating_width":
            for i in range(self.num_atom):
                self.weights.data[i] = find_closest_width(self.LUT, self.target_phase_response[i, i].item())
        else:
            raise ValueError(f"Unsupported param_method: {self.param_method}")
    
    def direct_set_pillar_width(self, weights):
        assert len(weights) == self.num_atom, f"The length of weights must be equal to the number of atoms {len(weights)}!={self.num_atom}"
        self.weights.data = weights

    def get_pillar_width(self):
        return self.weights.data

    def build_param(self):
        if self.param_method == "level_set":
            self.weights = nn.Parameter(
                0.05 * torch.ones((self.num_atom), device=self.device)
            )
            if (self.LUT is None) or (self.target_phase_response is None):
                with DeterministicCtx(seed=41):
                    self.weights.data = self.weights.data + 0.01 * torch.randn_like(self.weights.data)
            else:
                for i in range(self.num_atom):
                    self.weights.data[i] = get_mid_weight(0.05, find_closest_width(self.LUT, self.target_phase_response[i, i].item()))
        elif self.param_method == "grating_width":
            raise NotImplementedError("Grating width is deprecated.")
            self.weights = nn.Parameter(
                0.05 * torch.ones((self.num_atom), device=self.device)
            )
            if self.LUT is not None and self.target_phase_response is not None:
                for i in range(self.num_atom):
                    self.weights.data[i] = find_closest_width(self.LUT, self.target_phase_response[i, i].item())
        else:
            raise ValueError(f"Unsupported param_method: {self.param_method}")

    def build_patch(self):
        sim_cfg = DefaultSimulationConfig()
        total_sim_cfg = DefaultSimulationConfig()

        wl = 0.850
        sim_cfg.update(
            dict(
                solver="ceviche_torch",
                numerical_solver="solve_direct",
                use_autodiff=False,
                neural_solver=None,
                border_width=[0, 0, 0.5, 0.5],
                PML=[0.5, 0.5],
                resolution=50,
                wl_cen=wl,
                plot_root=self.plot_root,
            )
        )
        total_sim_cfg.update(
            dict(
                solver="ceviche_torch",
                numerical_solver="solve_direct",
                use_autodiff=False,
                neural_solver=None,
                border_width=[0, 0, 0.5, 0.5],
                PML=[0.5, 0.5],
                resolution=50,
                wl_cen=wl,
                plot_root=self.plot_root,
            )
        )
        self.opt = nn.ModuleDict()
        for patch_size in range(self.patch_size // 2 + 1, self.patch_size + 1):
            if patch_size > self.num_atom:
                break
            patch_metalens = MetaLens(
                            material_bg="Air",
                            material_r = "Si",
                            material_sub="SiO2",
                            sim_cfg=sim_cfg,
                            aperture=self.atom_period * (patch_size + 1),
                            port_len=(1, 1),
                            port_width=(
                                self.atom_period * (patch_size + 1),
                                self.atom_period
                            ),
                            substrate_depth=0,
                            ridge_height_max=0.75,
                            nearfield_dx=0.3,
                            nearfield_offset=0, # only support y direction offset for now, so this is only a scalar, not a tuple
                            nearfield_size=self.atom_period * (min(self.probing_region_size, patch_size) + 1),
                            farfield_dxs=((self.target_dx, self.target_dx + 1 / 50),),
                            farfield_sizes=(self.atom_period * self.num_atom,),
                            farfield_offset=0,
                            device=self.device,
                            design_var_type=self.design_var_type,
                            atom_period=self.atom_period,
                            atom_width=self.atom_width,
                        )
            hr_patch_metalens = patch_metalens.copy(resolution=200)
            self.opt[str(patch_size + 1)] = MetaLensOptimization(
                device=patch_metalens,
                hr_device=hr_patch_metalens,
                sim_cfg=sim_cfg,
                operation_device=self.device,
                design_region_param_cfgs=self.design_region_param_cfgs,
            ).to(self.device)
            if patch_size == self.patch_size:
                patch_metalens = MetaLens(
                                material_bg="Air",
                                material_r = "Si",
                                material_sub="SiO2",
                                sim_cfg=sim_cfg,
                                aperture=self.atom_period * (patch_size + 2),
                                port_len=(1, 1),
                                port_width=(
                                    self.atom_period * (patch_size + 2),
                                    self.atom_period
                                ),
                                substrate_depth=0,
                                ridge_height_max=0.75,
                                nearfield_dx=0.3,
                                nearfield_offset=0, # only support y direction offset for now, so this is only a scalar, not a tuple
                                nearfield_size=self.atom_period * (min(self.probing_region_size, patch_size) + 2),
                                farfield_dxs=((self.target_dx, self.target_dx + 1 / 50),),
                                farfield_sizes=(self.atom_period * self.num_atom,),
                                farfield_offset=0,
                                device=self.device,
                                design_var_type=self.design_var_type,
                                atom_period=self.atom_period,
                                atom_width=self.atom_width,
                            )
                hr_patch_metalens = patch_metalens.copy(resolution=200)
                self.opt[str(patch_size + 2)] = MetaLensOptimization(
                    device=patch_metalens,
                    hr_device=hr_patch_metalens,
                    sim_cfg=sim_cfg,
                    operation_device=self.device,
                    design_region_param_cfgs=self.design_region_param_cfgs,
                ).to(self.device)
            # the key is from 9 (0+1+8) to 17 (8+1+8)
        total_metalens = MetaLens(
            material_bg="Air",
            material_r = "Si",
            material_sub="SiO2",
            sim_cfg=total_sim_cfg,
            aperture=self.atom_period * self.num_atom,
            port_len=(1, 1),
            port_width=(self.atom_period * self.num_atom, self.atom_period),
            substrate_depth=0,
            ridge_height_max=0.75,
            nearfield_dx=0.3,
            nearfield_size=self.atom_period * self.num_atom,
            farfield_dxs=((self.dz, self.dz + 2/50),),
            farfield_sizes=(self.atom_period * self.num_atom,),
            device=self.device,
            design_var_type=self.design_var_type,
            atom_period=self.atom_period,
            atom_width=self.atom_width,
        )
        hr_total_metalens = total_metalens.copy(resolution=200)
        self.total_opt = MetaLensOptimization(
            device=total_metalens,
            hr_device=hr_total_metalens,
            sim_cfg=total_sim_cfg,
            operation_device=self.device,
            design_region_param_cfgs=self.design_region_param_cfgs,
        )

    def enable_solver_cache(self):
        sim_key = list(self.total_opt.objective.sims.keys())
        assert len(sim_key) == 1, f"there should be only one sim key, but we got {sim_key}"
        for key, opt in self.opt.items():
            if hasattr(opt.objective.sims[sim_key[0]].solver, "clear_solver_cache"):
                opt.objective.sims[sim_key[0]].solver.clear_solver_cache()
                opt.objective.sims[sim_key[0]].solver.set_cache_mode(True)

    def disable_solver_cache(self):
        sim_key = list(self.total_opt.objective.sims.keys())
        assert len(sim_key) == 1, f"there should be only one sim key, but we got {sim_key}"
        for key, opt in self.opt.items():
            if hasattr(opt.objective.sims[sim_key[0]].solver, "clear_solver_cache"):
                opt.objective.sims[sim_key[0]].solver.clear_solver_cache()
                opt.objective.sims[sim_key[0]].solver.set_cache_mode(False)

    def set_ref_response(
            self, 
            ref_mag: torch.Tensor, 
            ref_phase: torch.Tensor,
        ):
        self.ref_phase = ref_phase
        self.ref_mag = ref_mag

    def get_design_variables(self, weight):
        if self.design_var_type == "width":
            num_atom = len(weight)

            zero_ls_knots = -0.05 * torch.ones((2 * num_atom + 1,), device=weight.device, dtype=weight.dtype)
            zero_ls_knots[1::2] = weight  # insert weight into odd indices

            custom_source = {
                "design_region_0": zero_ls_knots.unsqueeze(0),
            }

        elif self.design_var_type == "height":
            custom_source = {}
            fixed_knots = 0.05 * torch.ones_like(weight, device=weight.device, dtype=weight.dtype)
            total_knots = torch.stack([fixed_knots, weight])  # shape (2, num_atom)

            for i in range(len(weight)):
                custom_source[f"design_region_{i}"] = total_knots[:, i].unsqueeze(1)  # shape (2,1)
        else:
            raise ValueError(f"Unsupported design_var_type: {self.design_var_type}")
        
        return custom_source
    
    def get_empty_design_variables(self, weight: dict = {}):
        if self.design_var_type == "width":
            custom_source = {
                "design_region_0": -0.05 * torch.ones_like(weight["design_region_0"])
            }

        elif self.design_var_type == "height":
            custom_source = {}
            for key, ls_knots in weight.items():
                custom_source[key] = -0.05 * torch.ones_like(ls_knots)

        else:
            raise ValueError(f"Unsupported design_var_type: {self.design_var_type}")
        
        return custom_source


    def forward(self, sharpness, in_down_sample_rate=15, out_down_sample_rate=None):
        if out_down_sample_rate is None:
            out_down_sample_rate = in_down_sample_rate
        # in each time of forward, we need simulate the transfer matrix using the stiched patch metaatom
        # in each of the for loop, we need to run 13 times of simulation for differnet input port
        self.disable_solver_cache()
        self.enable_solver_cache()
        source_per_atom = round(50 * self.atom_period // in_down_sample_rate) # down sample rate = 5, source_per_atom = 3
        total_response = torch.zeros(
            (
                round(self.num_atom * self.atom_period * 50), # 480
                self.num_atom * source_per_atom, # 96
            ), 
            dtype=torch.complex128,
            device=self.device,
        )
        if self.field_norm_condition == "w_lens":
            self.normalizer_list = [] # need to reset the normalizer list since each time the reflection is different and the source field is different
        elif self.field_norm_condition == "wo_lens":
            pass
        # if self.param_method == "level_set":
        #     total_weight = -0.05 * torch.ones(2 * (self.num_atom) + 1, device=self.device)
        #     self.level_set_knots = total_weight.clone()
        #     self.level_set_knots[1::2] = self.weights
        # else:
        #     raise ValueError(f"Unsupported param_method: {self.param_method}")
        for i in range(self.num_atom):
            # total_ls_knot[1::2] = self.pillar_ls_knots
            center_knot_idx = 2 * i + 1
            for j in range(source_per_atom): # range(3)
                if i >= self.patch_size // 2 + 1 and i < self.num_atom - self.patch_size // 2 - 1:
                    if self.param_method == "level_set":
                        # weights = {"design_region_0": self.level_set_knots[
                        #     center_knot_idx - 2 * (self.patch_size // 2 + 1) - 1 : center_knot_idx + 2 * (self.patch_size // 2 + 1 + 1)
                        # ].unsqueeze(0)}
                        weights = self.get_design_variables(
                            self.weights[
                                i - self.patch_size // 2 - 1 : i - self.patch_size // 2 - 1 + self.patch_size + 2
                            ]
                        )
                    elif self.param_method == "grating_width":
                        weights = {"design_region_0": self.weights[
                            i - self.patch_size // 2 - 1 : i - self.patch_size // 2 - 1 + self.patch_size + 2
                        ]}
                    else:
                        raise ValueError(f"Unsupported param_method: {self.param_method}")
                    required_patch = self.patch_size
                    opt_idx = required_patch + 2
                elif i < self.patch_size // 2 + 1:
                    if self.param_method == "level_set":
                        # weights = {"design_region_0": self.level_set_knots[
                        #     : center_knot_idx + 2 * (self.patch_size // 2 + 1 + 1)
                        # ].unsqueeze(0)}
                        weights = self.get_design_variables(
                            self.weights[
                                : i - self.patch_size // 2 + self.patch_size + 1
                            ]
                        )
                    elif self.param_method == "grating_width":
                        weights = {"design_region_0": self.weights[
                            : i - self.patch_size // 2 + self.patch_size + 1
                        ]}
                    else:
                        raise ValueError(f"Unsupported param_method: {self.param_method}")
                    required_patch = min(i + self.patch_size // 2 + 1, self.num_atom)
                    opt_idx = required_patch + 1
                else:
                    if self.param_method == "level_set":
                        # weights = {"design_region_0": self.level_set_knots[
                        #     center_knot_idx - 2 * (self.patch_size // 2 + 1) - 1 : 
                        # ].unsqueeze(0)}
                        weights = self.get_design_variables(
                            self.weights[
                                i - self.patch_size // 2 - 1 :
                            ]
                        )
                    elif self.param_method == "grating_width":
                        weights = {"design_region_0": self.weights[
                            i - self.patch_size // 2 - 1 :
                        ]}
                    else:
                        raise ValueError(f"Unsupported param_method: {self.param_method}")
                    required_patch = self.num_atom - i + self.patch_size // 2
                    opt_idx = required_patch + 1
                opt = self.opt[str(opt_idx)]
                if self.downsample_method == "avg":
                    source = torch.zeros(required_patch * source_per_atom, device=self.device)
                    if i >= self.patch_size // 2 + 1 and i < self.num_atom - self.patch_size // 2 - 1:
                        source[self.patch_size//2 * source_per_atom + j] = 1
                    elif i < self.patch_size // 2 + 1:
                        source[i * source_per_atom + j] = 1
                    else:
                        source[-(self.num_atom - i) * source_per_atom + j] = 1
                    source = source.repeat_interleave(in_down_sample_rate)
                elif self.downsample_method == "point":
                    source = torch.zeros(round(50 * self.atom_period * required_patch), device=self.device)
                    if i >= self.patch_size // 2 + 1 and i < self.num_atom - self.patch_size // 2 - 1:
                        source[self.patch_size // 2 * round(50 * self.atom_period) + j * in_down_sample_rate + in_down_sample_rate // 2] = 1
                    elif i < self.patch_size // 2 + 1:
                        source[i * round(50 * self.atom_period) + j * in_down_sample_rate + in_down_sample_rate // 2] = 1
                    else:
                        source[-(self.num_atom - i) * round(50 * self.atom_period) + j * in_down_sample_rate + in_down_sample_rate // 2] = 1
                else:
                    raise ValueError(f"Unsupported downsample_method: {self.downsample_method}")
                source_zero_padding = torch.zeros(int(0.5 * 50), device=self.device)
                additional_atom_padding = torch.zeros(int(self.atom_period * 50), device=self.device)
                additional_boolean_padding = torch.zeros(int(self.atom_period * 50), dtype=torch.bool, device=self.device)
                source = torch.cat([source_zero_padding, source, source_zero_padding])
                boolean_source_mask = torch.zeros_like(source, dtype=torch.bool)
                boolean_source_mask[torch.where(source != 0)] = True
                if i >= self.patch_size // 2 + 1 and i < self.num_atom - self.patch_size // 2 - 1:
                    source = torch.cat([additional_atom_padding, source, additional_atom_padding])
                    boolean_source_mask = torch.cat([additional_boolean_padding, boolean_source_mask, additional_boolean_padding])
                elif i < self.patch_size // 2 + 1:
                    source = torch.cat([source, additional_atom_padding])
                    boolean_source_mask = torch.cat([boolean_source_mask, additional_boolean_padding])
                else:
                    source = torch.cat([additional_atom_padding, source])
                    boolean_source_mask = torch.cat([additional_boolean_padding, boolean_source_mask])
                custom_source = dict(
                    source=source,
                    slice_name="in_slice_1",
                    mode="Hz1",
                    wl=0.85,
                    direction="x+",
                )
                # print(f"this is the weights: \n {weights}", flush=True)
                _ = opt(sharpness=sharpness, weight=weights, custom_source=custom_source, simulation_id = i * source_per_atom + j)

                # opt.plot(
                #     plot_filename=f"{self.design_var_type}_center_atom_{i}_source_{j}.png",
                #     eps_map=opt._eps_map,
                #     obj=None,
                #     field_key=("in_slice_1", 0.85, "Hz1", 300),
                #     field_component="Hz",
                #     in_slice_name="in_slice_1",
                #     exclude_slice_names=[],
                # )

                response = opt.objective.response[('in_slice_1', 'nearfield_1', 0.85, "Hz1", 300)]["fz"].squeeze()
                if self.field_norm_condition == "w_lens" and self.tm_norm == "field":
                    source_field = opt.objective.response[('in_slice_1', 'in_slice_1', 0.85, "Hz1", 300)]["fz"].squeeze()
                    self.normalizer_list.append(source_field[boolean_source_mask].mean())
                elif self.field_norm_condition == "wo_lens" and len(self.normalizer_list) < self.num_atom*source_per_atom and self.tm_norm == "field":
                    with torch.no_grad():
                        ls_knots_wo_lens = self.get_empty_design_variables(
                            weights,
                        )
                        _ = opt(sharpness=sharpness, weight=ls_knots_wo_lens, custom_source=custom_source, simulation_id = 100000 + i * source_per_atom + j)
                        source_field = opt.objective.response[('in_slice_1', 'in_slice_1', 0.85, "Hz1", 300)]["fz"].squeeze()
                        self.normalizer_list.append(source_field[boolean_source_mask].mean())

                # near 2 far projection
                px_per_atom = round(self.atom_period * 50)
                if i >= self.patch_size // 2 + 1 and i < self.num_atom - self.patch_size // 2 - 1:
                    response = response[
                        px_per_atom + (j - source_per_atom // 2) * in_down_sample_rate: -px_per_atom + (j - source_per_atom // 2) * in_down_sample_rate
                    ]
                elif i < self.patch_size // 2 + 1:
                    if i == self.patch_size // 2 and j > source_per_atom // 2:
                        response = response[
                            (j - source_per_atom // 2) * in_down_sample_rate: -px_per_atom + (j - source_per_atom//2) * in_down_sample_rate
                        ]
                    else:
                        response = response[
                            : -px_per_atom + (j - source_per_atom//2) * in_down_sample_rate
                        ]
                else:
                    if i == self.num_atom - self.patch_size // 2 - 1 and j < source_per_atom // 2:
                        response = response[
                            px_per_atom + (j - source_per_atom//2) * in_down_sample_rate : - (source_per_atom // 2 - j) * in_down_sample_rate
                        ]
                    else:
                        response = response[
                            px_per_atom + (j - source_per_atom//2) * in_down_sample_rate :
                        ]

                col_idx = i * source_per_atom + j
                if i >= self.patch_size // 2 + 1 and i < self.num_atom - self.patch_size // 2 - 1:
                    row_start = (i - self.patch_size // 2) * px_per_atom + (j - source_per_atom // 2) * in_down_sample_rate
                    row_end   = (i - self.patch_size // 2) * px_per_atom + (j - source_per_atom // 2) * in_down_sample_rate + self.patch_size * px_per_atom
                    # print(f"1 we are dealing with the {i}th atom, the {j}th source, the col_idx is {col_idx}, the row start is {row_start}, the row end is {row_end}", flush=True)
                elif i < self.patch_size // 2 + 1:
                    if i == self.patch_size // 2 and j > source_per_atom // 2:
                        row_start = (j - source_per_atom // 2) * in_down_sample_rate
                    else:
                        row_start = 0
                    row_end   = opt_idx * px_per_atom - px_per_atom + (j - source_per_atom//2) * in_down_sample_rate
                    # print(f"2 we are dealing with the {i}th atom, the {j}th source, the col_idx is {col_idx}, the row start is {row_start}, the row end is {row_end}", flush=True)
                else:
                    if i == self.num_atom - self.patch_size // 2 - 1 and j < source_per_atom // 2:
                        row_end = self.num_atom * px_per_atom - (source_per_atom // 2 - j) * in_down_sample_rate
                        row_start = row_end - self.patch_size * px_per_atom
                    else:
                        row_end = self.num_atom * px_per_atom
                        row_start = row_end - opt_idx * px_per_atom + px_per_atom + (j - source_per_atom//2) * in_down_sample_rate
                    # print(f"3 we are dealing with the {i}th atom, the {j}th source, the col_idx is {col_idx}, the row start is {row_start}, the row end is {row_end}", flush=True)
                # Construct a partial matrix that depends on `response`
                partial_matrix = torch.zeros_like(total_response)
                # print(f"we are dealing with the {i}th atom, the {j}th source, the col_idx is {col_idx}", flush=True)
                # print("this is the col_idx: ", col_idx, "this is the length of the response", row_end - row_start, "this is the row_start: ", row_start, "this is the row_end: ", row_end, flush=True)
                if self.tm_norm == "field" and self.field_norm_condition == "w_lens":
                    partial_matrix[row_start:row_end, col_idx] = response / self.normalizer_list[-1]
                elif self.tm_norm == "field" and self.field_norm_condition == "wo_lens":
                    partial_matrix[row_start:row_end, col_idx] = response / self.normalizer_list[col_idx]
                else:
                    partial_matrix[row_start:row_end, col_idx] = response
                
                # Now accumulate
                total_response = total_response + partial_matrix
                # self.opt.plot(
                #     plot_filename=f"patched_metalens_sharp-{sharpness}_{i}.png",
                #     eps_map=None,
                #     obj=None,
                #     field_key=("in_slice_1", 0.85, "Hz1", 300),
                #     field_component="Hz",
                #     in_slice_name="in_slice_1",
                #     exclude_slice_names=[],
                # )
                # print_stat(trust_worthy_phase)
                # trust_worthy_phase_list.append(trust_worthy_phase)
                # print("this is the shape of the trust_worthy_phase", trust_worthy_phase.shape, flush=True)
        if self.downsample_mode == "input_only":
            raise NotImplementedError("input_only downsample mode is deprecated now.")
            total_response = torch.repeat_interleave(total_response, repeats=down_sample_rate, dim=-1)
            rolled_total_response = torch.zeros_like(total_response)
            for i in range(total_response.shape[1]):
                if i % down_sample_rate == down_sample_rate // 2:
                    rolled_total_response[:, i] = total_response[:, i]
                else:
                    # up is nagative shift
                    shift = (i % down_sample_rate - down_sample_rate // 2)
                    shifted_col = torch.roll(total_response[:, i], shift)
                    if shift > 0:
                        rolled_total_response[shift:, i] = shifted_col[shift:]
                    else:
                        rolled_total_response[:shift, i] = shifted_col[:shift]
            total_response = rolled_total_response
        elif self.downsample_mode == "both":
            total_response = total_response[
                out_down_sample_rate // 2::out_down_sample_rate,
                :,
            ]
        if self.tm_norm == "max":
            return total_response / torch.max(total_response.abs())
        elif self.tm_norm == "field" or "none":
            return total_response
        elif self.tm_norm == "energy":
            raise NotImplementedError("Energy normalization is not supported yet.")
            return total_response
        else:
            raise ValueError(f"Unsupported tm_norm: {self.tm_norm}")