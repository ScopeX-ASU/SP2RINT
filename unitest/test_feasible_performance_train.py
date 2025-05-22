#!/usr/bin/env python
# coding=UTF-8
import argparse
import os
import sys
import psutil
from typing import Callable, Dict, Iterable, Optional, List

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from pyutils.config import configs
from pyutils.general import AverageMeter
from pyutils.general import logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core import builder
from core.datasets.mixup import MixupAll
from core.utils import (
    get_parameter_group, 
    register_hidden_hooks, 
    probe_near2far_matrix, 
    CosSimLoss,
    reset_optimizer_and_scheduler,
)
from core.models.patch_metalens import PatchMetalens
import csv
from thirdparty.MAPS_old.core.utils import SharpnessScheduler
from thirdparty.MAPS_old.core.fdfd.pardiso_solver import pardisoSolver
sys.path.pop()
from pyutils.general import ensure_dir
from matplotlib import pyplot as plt
import h5py
import copy
import numpy as np
import wandb
import datetime
import psutil
import torch

def update_admm_dual_variable(
    model: torch.nn.Module,
    stitched_response: List[torch.Tensor],
    admm_vars: Dict,
) -> None:
    """
    Updates the ADMM dual variable u:  u <- u + x - z

    Args:
        model: the DONN model with trained metasurface layers
        stitched_response: List of projected implementable transfer matrices (z)
        admm_vars: ADMM variables (z, u, rho)
        path_depth: number of metasurface layers
    """
    for i in range(configs.model.conv_cfg.path_depth):
        # x is the current (non-implementable) transfer matrix from DONN
        x = model.state_dict()[f"features.conv1.conv._conv_pos.metalens.{i}_17.W_buffer"]
        admm_vars['z_admm'][i] = stitched_response[i].detach()
        admm_vars['u_admm'][i] = admm_vars['u_admm'][i] + x - admm_vars['z_admm'][i]

def transfer_weights(src_model, tgt_model):
    """
    Transfers weights from src_model to tgt_model, skipping layers with mismatched sizes.
    
    Parameters:
    - src_model (torch.nn.Module): The source model providing the weights.
    - tgt_model (torch.nn.Module): The target model receiving the weights.

    Returns:
    - loaded_keys (list): Successfully loaded keys.
    - skipped_keys (list): Keys skipped due to shape mismatch.
    """

    src_state_dict = src_model.state_dict()
    tgt_state_dict = tgt_model.state_dict()

    loaded_keys = []
    skipped_keys = []

    # Create a new state_dict with only matching keys
    filtered_state_dict = {}

    for key, param in src_state_dict.items():
        if key in tgt_state_dict and param.shape == tgt_state_dict[key].shape:
            filtered_state_dict[key] = param  # Only load matching keys
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)

    # Update target model's state_dict
    tgt_state_dict.update(filtered_state_dict)
    tgt_model.load_state_dict(tgt_state_dict)

    # print(f"✅ Successfully loaded {len(loaded_keys)} layers.")
    # print(f"⚠️ Skipped {len(skipped_keys)} layers due to size mismatch:")
    # for key in skipped_keys:
    #     print(f"   - {key}: Expected {tgt_state_dict[key].shape}, but found {src_state_dict[key].shape}")
    return None
    # return loaded_keys, skipped_keys

def probe_full_tm(
    device,
    patched_metalens,
    full_wave_down_sample_rate = 1,
):
    # time_start = time.time()
    number_atoms = configs.invdes.num_atom
    sources = torch.eye(number_atoms * round(15 // full_wave_down_sample_rate), device=device)

    sim_key = list(patched_metalens.total_opt.objective.sims.keys())
    assert len(sim_key) == 1, f"there should be only one sim key, but we got {sim_key}"
    if hasattr(patched_metalens.total_opt.objective.sims[sim_key[0]].solver, "clear_solver_cache"):
        patched_metalens.total_opt.objective.sims[sim_key[0]].solver.clear_solver_cache()
        patched_metalens.total_opt.objective.sims[sim_key[0]].solver.set_cache_mode(True)
    # we first need to run the normalizer
    if patched_metalens.total_normalizer_list is None or len(patched_metalens.total_normalizer_list) < number_atoms * round(15 // full_wave_down_sample_rate):
        total_normalizer_list = []
        for idx in range(number_atoms * round(15 // full_wave_down_sample_rate)):
            source_i = sources[idx].repeat_interleave(full_wave_down_sample_rate)
            source_zero_padding = torch.zeros(int(0.5 * 50), device=device)
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
            if configs.invdes.param_method == "level_set":
                _ = patched_metalens.total_opt(
                    sharpness=256, 
                    weight={"design_region_0": -0.05 * torch.ones_like(patched_metalens.level_set_knots.unsqueeze(0))},
                    custom_source=custom_source
                )
            elif configs.invdes.param_method == "grating_width":
                raise NotImplementedError("grating_width is deprecated")
                _ = patched_metalens.total_opt(
                    sharpness=256, 
                    weight={"design_region_0": patched_metalens.weights},
                    custom_source=custom_source
                )
            else:
                raise NotImplementedError(f"Unknown param_method: {configs.invdes.param_method}")

            source_field = patched_metalens.total_opt.objective.response[('in_slice_1', 'in_slice_1', 0.85, "Hz1", 300)]["fz"].squeeze()
            total_normalizer_list.append(source_field[boolean_source_mask].mean())
            if idx == number_atoms * round(15 // full_wave_down_sample_rate) - 1:
                patched_metalens.set_total_normalizer_list(total_normalizer_list)
    # now we already have the normalizer, we can run the full wave response
    if hasattr(patched_metalens.total_opt.objective.sims[sim_key[0]].solver, "clear_solver_cache"):
        patched_metalens.total_opt.objective.sims[sim_key[0]].solver.clear_solver_cache()
        
    with torch.no_grad():
        full_wave_response = torch.zeros(
            (
                number_atoms * round(15 // full_wave_down_sample_rate),
                number_atoms * round(15 // full_wave_down_sample_rate),
            ),
            device=device, 
            dtype=torch.complex128
        )
        for idx in range(number_atoms * round(15 // full_wave_down_sample_rate)):
            source_i = sources[idx].repeat_interleave(full_wave_down_sample_rate)
            source_zero_padding = torch.zeros(int(0.5 * 50), device=device)
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
            if configs.invdes.param_method == "level_set":
                ls_knot = -0.05 * torch.ones_like(patched_metalens.level_set_knots)
                ls_knot[1::2] = patched_metalens.weights.data
                _ = patched_metalens.total_opt(
                    sharpness=256, 
                    weight={"design_region_0": ls_knot.unsqueeze(0)},
                    custom_source=custom_source
                )
            elif configs.invdes.param_method == "grating_width":
                _ = patched_metalens.total_opt(
                    sharpness=256, 
                    weight={"design_region_0": patched_metalens.weights},
                    custom_source=custom_source
                )
            else:
                raise NotImplementedError(f"Unknown param_method: {configs.invdes.param_method}")

            response = patched_metalens.total_opt.objective.response[('in_slice_1', 'nearfield_1', 0.85, "Hz1", 300)]["fz"].squeeze()
            response = response[full_wave_down_sample_rate // 2 :: full_wave_down_sample_rate]
            assert len(response) == number_atoms * round(15 // full_wave_down_sample_rate), f"{len(response)}!= {number_atoms * round(15 // full_wave_down_sample_rate)}"
            full_wave_response[idx] = response
        full_wave_response = full_wave_response.transpose(0, 1)
        if configs.model.conv_cfg.max_tm_norm:
            full_wave_response = full_wave_response / torch.max(full_wave_response.abs())
        else:
            normalizer = torch.stack(patched_metalens.total_normalizer_list, dim=0).to(device)
            normalizer = normalizer.unsqueeze(1)
            full_wave_response = full_wave_response / normalizer

    if hasattr(patched_metalens.total_opt.objective.sims[sim_key[0]].solver, "clear_solver_cache"):
        patched_metalens.total_opt.objective.sims[sim_key[0]].solver.clear_solver_cache()
        patched_metalens.total_opt.objective.sims[sim_key[0]].solver.set_cache_mode(False)

    # time_end = time.time()
    # print(f"this is the time for probing the full wave response: {time_end - time_start}", flush=True)
    return full_wave_response

def project_to_implementable_subspace(
        model, 
        model_test,
        patched_metalens,
        patched_metalens_list, 
        invdes_criterion,
        prev_pillar_width,
        out_epoch,
        proj_idx, 
        in_downsample_rate_scheduler,
        out_downsample_rate_scheduler,
        near2far_matrix,
        ds_near2far_matrix,
        device,
        probe_full_wave = True,
        invdes_lr=None,
        invdes_sharp=None,
        layer_wise_matching=True,
        epoch_per_proj=None,
        finetune_entire = False,
        match_entire_epoch=None,
        admm_vars=None,
    ):
    '''
    there is not reparameterization of the transfer matrix of the metasurface during the DONN training,
    so we cannot ensure that the transfer matrix from DONN is implementable in the real world,
    for example, transfer matrix is not unitary, which means that it is not energy-conserved.

    in this function, we use the inverse design to project the transfer matrix of the metasurface trained in DONN to the implementable subspace
    we can actually get the real-world width of the metaatom array that corresponds to the required DONN transfer matrix"
    '''
    # lr = 5e-3
    # lr = configs.invdes.lr
    print(f"learning rate to be used: {invdes_lr}", flush=True)
    print(f"this is the sharpness to be used: {invdes_sharp}", flush=True)
    stitched_response_list = []
    if model_test is not None:
        transfer_weights(model, model_test)
    current_pillar_width = []
    in_downsample_rate = in_downsample_rate_scheduler.step()
    out_downsample_rate = out_downsample_rate_scheduler.step()
    if finetune_entire:
        target_entire_transfer_matrix = None
        for i in range(configs.model.conv_cfg.path_depth):
            current_lens_tm = model.state_dict()[f"features.conv1.conv._conv_pos.metalens.{i}_17.W_buffer"]
            ds_near2far_matrix = ds_near2far_matrix.to(current_lens_tm.dtype)
            if target_entire_transfer_matrix is None:
                target_entire_transfer_matrix = ds_near2far_matrix @ current_lens_tm
            else:
                target_entire_transfer_matrix = ds_near2far_matrix @ current_lens_tm @ target_entire_transfer_matrix
        target_entire_transfer_matrix_phase = torch.angle(target_entire_transfer_matrix)
        target_entire_transfer_matrix_phase_variants = torch.stack([
            target_entire_transfer_matrix_phase,         # Original
            target_entire_transfer_matrix_phase + 2 * torch.pi,
            target_entire_transfer_matrix_phase - 2 * torch.pi,
        ], dim=0)

    target_transfer_matrix = []
    for i in range(configs.model.conv_cfg.path_depth):
        target_transfer_matrix.append(model.state_dict()[f"features.conv1.conv._conv_pos.metalens.{i}_17.W_buffer"])

    if layer_wise_matching:
        for i in range(configs.model.conv_cfg.path_depth):
            # first, we need to read the transfer matrix of the metasurface trained in DONN
            A = target_transfer_matrix[i]

            target_response = A
            if configs.model.conv_cfg.max_tm_norm:
                target_response = target_response / torch.max(target_response.abs())

            target_phase_response = torch.angle(target_response)
            target_phase_variants = torch.stack([
                target_phase_response,         # Original
                target_phase_response + 2 * torch.pi,
                target_phase_response - 2 * torch.pi,
            ], dim=0)  # Shape (3, N)

            if configs.invdes.project_init == "LPA" or prev_pillar_width is None:
                patched_metalens.set_target_phase_response(target_phase_response)
                patched_metalens.rebuild_param()
            elif configs.invdes.project_init == "last_time":
                patched_metalens.direct_set_pillar_width(prev_pillar_width[i])
            else:
                raise NotImplementedError(f"Unknown project_init: {configs.invdes.project_init}")
            # optimizer = torch.optim.Adam(patched_metalens.parameters(), lr=lr)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer, T_max=num_epoch, eta_min=lr * 1e-2
            # )
            invdes_optimizer, invdes_scheduler = reset_optimizer_and_scheduler(
                model=patched_metalens, 
                lr_init=invdes_lr[0],
                lr_final=invdes_lr[1], 
                num_epoch=epoch_per_proj,
            )
            sharp_scheduler = SharpnessScheduler(
                initial_sharp=invdes_sharp[0], 
                final_sharp=invdes_sharp[1], 
                total_steps=epoch_per_proj,
            )

            for epoch in range(epoch_per_proj):
                sharpness = sharp_scheduler.get_sharpness()
                invdes_optimizer.zero_grad()
                # time_start = time.time()
                stitched_response = patched_metalens.forward(
                    sharpness=sharpness, 
                    in_down_sample_rate=in_downsample_rate,
                    out_down_sample_rate=out_downsample_rate,
                )
                # time_end = time.time()
                # print(f"this is the time for the forward: {time_end - time_start}", flush=True)
                if configs.model.conv_cfg.max_tm_norm:
                    stitched_response = stitched_response / torch.max(stitched_response.abs()) # normalize the response
                if isinstance(invdes_criterion, CosSimLoss):
                    loss_fn_output = invdes_criterion(
                        stitched_response,
                        target_response.to(stitched_response.dtype),
                    )
                    loss_fn_output = -loss_fn_output
                elif configs.invdes.criterion.name == "TMMatching":
                    loss_fn_output = invdes_criterion(
                        stitched_response,
                        target_response,
                        target_phase_variants,
                        seperate_loss=configs.invdes.seperate_loss,
                    )
                elif configs.invdes.criterion.name == "ResponseMatching":
                    loss_fn_output = invdes_criterion(
                        target_response,
                        stitched_response,
                    )
                if isinstance(loss_fn_output, tuple):
                    loss = loss_fn_output[0]
                else:
                    loss = loss_fn_output
                
                if configs.invdes.admm:
                    x_admm = stitched_response
                    diff = x_admm - admm_vars['z_admm'][i] + admm_vars['u_admm'][i]
                    loss += (admm_vars['rho_admm'] / 2.0) * torch.norm(diff) ** 2
                # time_start = time.time()
                loss.backward()
                # time_end = time.time()
                # print(f"this is the time for the backward: {time_end - time_start}", flush=True)
                patched_metalens.disable_solver_cache()
                # process = psutil.Process(os.getpid())
                # mem = process.memory_info().rss / 1024**2  # in MB
                # print(f"CPU memory usage: {mem:.2f} MB", flush=True)
                invdes_optimizer.step()

                invdes_scheduler.step()
                sharp_scheduler.step()
                # for p in patched_metalens.parameters():
                #     if p.grad is not None:
                #         print(f"this is the norm of the gradient: {p.grad.norm()}", flush=True)
                print(f"epoch: {epoch}, loss: {loss.item()}", flush=True)
                # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                # im0 = ax[0].imshow(
                #     target_response.abs().cpu().numpy()
                # )
                # ax[0].set_title("Target Magnitude")
                # fig.colorbar(im0, ax=ax[0])
                # im1 = ax[1].imshow(
                #     stitched_response.detach().abs().cpu().numpy()
                # )
                # ax[1].set_title("Stitched Magnitude")
                # fig.colorbar(im1, ax=ax[1])
                # im2 = ax[2].imshow(
                #     (stitched_response - target_response).detach().abs().cpu().numpy()
                # )
                # ax[2].set_title("Difference Magnitude")
                # fig.colorbar(im2, ax=ax[2])

                # fig.suptitle(f"this is the sharpness we are using: {sharpness}", fontsize=16)

                # plt.savefig(configs.plot.plot_root + f"convid-{i}_epoch-{out_epoch}_projID-{proj_idx}_invdesEpoch-{epoch}.png", dpi = 300)
                # plt.close()

            
            current_pillar_width.append(patched_metalens.get_pillar_width())
            with torch.no_grad():
                sharpness = sharp_scheduler.get_sharpness()
                print(f"this is the sharpness we are using to set the model: {sharpness}", flush=True)
                stitched_response = patched_metalens.forward(
                    sharpness=sharpness, 
                    in_down_sample_rate=in_downsample_rate,
                    out_down_sample_rate=out_downsample_rate,
                )
                if isinstance(invdes_criterion, CosSimLoss):
                    loss_fn_output = invdes_criterion(
                        stitched_response,
                        target_response.to(stitched_response.dtype),
                    )
                    loss_fn_output = -loss_fn_output
                elif configs.invdes.criterion.name == "TMMatching":
                    loss_fn_output = invdes_criterion(
                        stitched_response,
                        target_response,
                        target_phase_variants,
                        seperate_loss=configs.invdes.seperate_loss,
                    )
                elif configs.invdes.criterion.name == "ResponseMatching":
                    loss_fn_output = invdes_criterion(
                        target_response,
                        stitched_response,
                    )
                if isinstance(loss_fn_output, tuple):
                    loss = loss_fn_output[0]
                else:
                    loss = loss_fn_output
                print(f"final loss: {loss.item()}", flush=True)
                # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                # im0 = ax[0].imshow(
                #     target_response.abs().cpu().numpy()
                # )
                # ax[0].set_title("Target Magnitude")
                # fig.colorbar(im0, ax=ax[0])
                # im1 = ax[1].imshow(
                #     stitched_response.detach().abs().cpu().numpy()
                # )
                # ax[1].set_title("Stitched Magnitude")
                # fig.colorbar(im1, ax=ax[1])
                # im2 = ax[2].imshow(
                #     (stitched_response - target_response).detach().abs().cpu().numpy()
                # )
                # ax[2].set_title("Difference Magnitude")
                # fig.colorbar(im2, ax=ax[2])

                # fig.suptitle(f"this is the sharpness we are using: {sharpness}", fontsize=16)

                # plt.savefig(configs.plot.plot_root + f"convid-{i}_epoch-{out_epoch}_projID-{proj_idx}_invdesEpoch-{epoch_per_proj}.png", dpi = 300)
                # plt.close()
            model.features.conv1.conv._conv_pos.metalens[f"{i}_17"].set_param_from_target_matrix(stitched_response)
            stitched_response_list.append(stitched_response)
            # sweep real transfer matrix and set it to the model_test
            if probe_full_wave and not finetune_entire:
                high_res_response = probe_full_tm(device=device, patched_metalens=patched_metalens)
                # print("this is the keys in the model: ", list(model_test.features.conv1.conv._conv_pos.metalens.keys()))
                model_test.features.conv1.conv._conv_pos.metalens[f"{i}_17"].set_param_from_target_matrix(high_res_response)
                full_wave_response = high_res_response
                if target_response.shape != full_wave_response.shape:
                    assert full_wave_response.shape[-1] % target_response.shape[-1] == 0, f"{full_wave_response.shape[-1]} % {target_response.shape[-1]} != 0"
                    assert full_wave_response.shape[-2] % target_response.shape[-2] == 0, f"{full_wave_response.shape[-2]} % {target_response.shape[-2]} != 0"
                    assert full_wave_response.shape[-1] // target_response.shape[-1] == full_wave_response.shape[-2] // target_response.shape[-2]

                    ds_rate = full_wave_response.shape[-1] // target_response.shape[-1]

                    full_wave_response = full_wave_response[
                        ds_rate//2::ds_rate, 
                        :
                    ]
                    full_wave_response = full_wave_response.reshape(full_wave_response.shape[0], -1, ds_rate).sum(dim=-1)
                
                fig, ax = plt.subplots(2, 4, figsize=(20, 10))
                im0 = ax[0, 0].imshow(
                    target_response.abs().cpu().numpy()
                )
                ax[0, 0].set_title("Target Magnitude")
                fig.colorbar(im0, ax=ax[0, 0])
                im1 = ax[0, 1].imshow(
                    stitched_response.detach().abs().cpu().numpy()
                )
                ax[0, 1].set_title("Stitched Magnitude")
                fig.colorbar(im1, ax=ax[0, 1])
                im2 = ax[0, 2].imshow(
                    full_wave_response.abs().cpu().numpy()
                )
                ax[0, 2].set_title("Full Magnitude")
                fig.colorbar(im2, ax=ax[0, 2])
                im3 = ax[0, 3].imshow(
                    (
                        full_wave_response / torch.norm(full_wave_response, p=2) - target_response / torch.norm(target_response, p=2)
                    ).abs().cpu().numpy()
                )
                ax[0, 3].set_title("Difference Magnitude")
                fig.colorbar(im3, ax=ax[0, 3])

                im4 = ax[1, 0].imshow(
                    torch.angle(target_response).cpu().numpy()
                )
                ax[1, 0].set_title("Target Phase")
                fig.colorbar(im4, ax=ax[1, 0])
                im5 = ax[1, 1].imshow(
                    torch.angle(stitched_response.detach()).cpu().numpy()
                )
                ax[1, 1].set_title("Stitched Phase")
                fig.colorbar(im5, ax=ax[1, 1])
                im6 = ax[1, 2].imshow(
                    torch.angle(full_wave_response).cpu().numpy()
                )
                ax[1, 2].set_title("Full Phase")
                fig.colorbar(im6, ax=ax[1, 2])
                im7 = ax[1, 3].imshow(
                    torch.angle(
                        full_wave_response / torch.norm(full_wave_response, p=2) - target_response / torch.norm(target_response, p=2)
                    ).cpu().numpy()
                )
                ax[1, 3].set_title("Difference Phase")
                fig.colorbar(im7, ax=ax[1, 3])
                plt.savefig(configs.plot.plot_root + f"epoch-{out_epoch}_convid-{i}.png")
                plt.close() 
            else:
                if stitched_response.shape != target_response.shape:
                    # interpolate the target to the same size as the transfer matrix
                    W_stitched = stitched_response.shape[-1]
                    W_target = target_response.shape[-1]
                    stitched_response_real = F.interpolate(stitched_response.real.unsqueeze(0).unsqueeze(0), size=target_response.shape[-2:], mode="bilinear", align_corners=False).squeeze()
                    stitched_response_imag = F.interpolate(stitched_response.imag.unsqueeze(0).unsqueeze(0), size=target_response.shape[-2:], mode="bilinear", align_corners=False).squeeze()
                    stitched_response = stitched_response_real + 1j * stitched_response_imag
                    ds_rate = W_stitched / W_target
                    stitched_response = stitched_response * ds_rate
                fig, ax = plt.subplots(2, 3, figsize=(15, 10))
                im0 = ax[0, 0].imshow(
                    target_response.abs().cpu().numpy()
                )
                ax[0, 0].set_title("Target Magnitude")
                fig.colorbar(im0, ax=ax[0, 0])
                im1 = ax[0, 1].imshow(
                    stitched_response.detach().abs().cpu().numpy()
                )
                ax[0, 1].set_title("Stitched Magnitude")
                fig.colorbar(im1, ax=ax[0, 1])
                im3 = ax[0, 2].imshow(
                    (stitched_response - target_response).detach().abs().cpu().numpy()
                )
                ax[0, 2].set_title("Difference Magnitude")
                fig.colorbar(im3, ax=ax[0, 2])

                im4 = ax[1, 0].imshow(
                    torch.angle(target_response).cpu().numpy()
                )
                ax[1, 0].set_title("Target Phase")
                fig.colorbar(im4, ax=ax[1, 0])
                im5 = ax[1, 1].imshow(
                    torch.angle(stitched_response.detach()).cpu().numpy()
                )
                ax[1, 1].set_title("Stitched Phase")
                fig.colorbar(im5, ax=ax[1, 1])
                im7 = ax[1, 2].imshow(
                    torch.angle(stitched_response - target_response).detach().cpu().numpy()
                )
                ax[1, 2].set_title("Difference Phase")
                fig.colorbar(im7, ax=ax[1, 2])
                plt.savefig(configs.plot.plot_root + f"epoch-{out_epoch}_convid-{i}.png")
                plt.close()
    if not finetune_entire:
        assert layer_wise_matching, "neither layer-wise matching nor entire matching is set while projection is on, please check the configs"
        current_pillar_width = torch.stack(current_pillar_width, dim=0) # shape (depth, num_atom)
        stitched_response = torch.stack(stitched_response_list, dim=0)
        lg.info(f"this is the current pillar width: {current_pillar_width}")
        return stitched_response.detach(), current_pillar_width
    # fine-tune the entire transfer matrix
    # by matching UTUT
    else:
        # reset the return value
        match_entire_stitched_response_list = []
        match_entire_current_pillar_width = []
        for i in range(configs.model.conv_cfg.path_depth):
            # if configs.invdes.project_init == "LPA" or prev_pillar_width is None:
            #     patched_metalens.set_target_phase_response(target_phase_response)
            #     patched_metalens.rebuild_param()
            # elif configs.invdes.project_init == "last_time":
            #     patched_metalens.direct_set_pillar_width(prev_pillar_width[i])
            # else:
            #     raise NotImplementedError(f"Unknown project_init: {configs.invdes.project_init}")
            if len(current_pillar_width) == 0: # which means that we are not using the layer-wise matching
                assert not layer_wise_matching, "layer-wise matching is set but the len of current_pillar_width is 0, something is wrong"
                assert configs.invdes.project_init != "LPA", "project_init can not be LPA since we don't know the layer wise response info"
                if prev_pillar_width is not None:
                    # if we have the previous pillar width, we can set it to the patched_metalens
                    # otherwise, we only have a entire transfer matrix, and we cannot layer-wise set each layer
                    patched_metalens_list[i].direct_set_pillar_width(prev_pillar_width[i])
            else: # which means that we are using the layer-wise matching and we can get the current_pillar_width from it
                patched_metalens_list[i].direct_set_pillar_width(current_pillar_width[i])
        invdes_optimizer, invdes_scheduler = reset_optimizer_and_scheduler(
            model=patched_metalens_list, 
            lr_init=invdes_lr[0] if configs.invdes.adaptive_finetune_lr else configs.invdes.finetune_lr_init,
            lr_final=invdes_lr[1] if configs.invdes.adaptive_finetune_lr else configs.invdes.finetune_lr_final,
            num_epoch=match_entire_epoch,
        )
        sharp_scheduler = SharpnessScheduler(
            initial_sharp=invdes_sharp[0], 
            final_sharp=invdes_sharp[1], 
            total_steps=match_entire_epoch,
        )
        for i in range(match_entire_epoch):
            sharpness = sharp_scheduler.get_sharpness()
            invdes_optimizer.zero_grad()
            curren_entire_transfer_matrix = None
            for j in range(configs.model.conv_cfg.path_depth):
                current_layer_tm = patched_metalens_list[j].forward(
                    sharpness=sharpness, 
                    in_down_sample_rate=in_downsample_rate,
                    out_down_sample_rate=out_downsample_rate,
                )
                current_ds_near2far_matrix = near2far_matrix[
                    in_downsample_rate//2::in_downsample_rate, 
                    :
                ]
                current_ds_near2far_matrix = current_ds_near2far_matrix.reshape(current_ds_near2far_matrix.shape[0], -1, out_downsample_rate).sum(dim=-1)
                
                current_ds_near2far_matrix = current_ds_near2far_matrix.to(current_layer_tm.dtype)
                if curren_entire_transfer_matrix is None:
                    curren_entire_transfer_matrix = current_ds_near2far_matrix @ current_layer_tm
                else:
                    curren_entire_transfer_matrix = current_ds_near2far_matrix @ current_layer_tm @ curren_entire_transfer_matrix

            if configs.invdes.criterion.name == "TMMatching":
                loss_fn_output = invdes_criterion(
                    curren_entire_transfer_matrix,
                    target_entire_transfer_matrix,
                    target_entire_transfer_matrix_phase_variants,
                    seperate_loss=configs.invdes.seperate_loss
                )
            elif configs.invdes.criterion.name == "ResponseMatching":
                loss_fn_output = invdes_criterion(
                    target_entire_transfer_matrix,
                    curren_entire_transfer_matrix,
                )
            else:
                raise NotImplementedError(f"Unknown criterion: {configs.invdes.criterion.name}")
            if isinstance(loss_fn_output, tuple):
                loss = loss_fn_output[0]
            else:
                loss = loss_fn_output
            loss.backward()
            for j in range(configs.model.conv_cfg.path_depth):
                patched_metalens_list[j].disable_solver_cache()
            # process = psutil.Process(os.getpid())
            # mem = process.memory_info().rss / 1024**2  # in MB
            # print(f"CPU memory usage: {mem:.2f} MB", flush=True)
            invdes_optimizer.step()
            invdes_scheduler.step()
            sharp_scheduler.step()
            print(f"matching entire epoch: {i}, loss: {loss.item()}", flush=True)
            if i == match_entire_epoch - 1:
                match_entire_current_pillar_width = [patched_metalens_list[idx].get_pillar_width() for idx in range(configs.model.conv_cfg.path_depth)]
                with torch.no_grad():
                    sharpness = sharp_scheduler.get_sharpness()
                    match_entire_stitched_response_list = [
                        patched_metalens_list[idx].forward(
                            sharpness=sharpness, 
                            in_down_sample_rate=in_downsample_rate,
                            out_down_sample_rate=out_downsample_rate,
                        ) for idx in range(configs.model.conv_cfg.path_depth)
                    ]

        stitched_response = torch.stack(match_entire_stitched_response_list, dim=0)
        current_pillar_width = torch.stack(match_entire_current_pillar_width, dim=0)
        for i in range(configs.model.conv_cfg.path_depth):
            model.features.conv1.conv._conv_pos.metalens[f"{i}_17"].set_param_from_target_matrix(match_entire_stitched_response_list[i])
        if probe_full_wave:
            for i in range(configs.model.conv_cfg.path_depth):
                high_res_response = probe_full_tm(device=device, patched_metalens=patched_metalens_list[i])
                # print("this is the keys in the model: ", list(model_test.features.conv1.conv._conv_pos.metalens.keys()))
                model_test.features.conv1.conv._conv_pos.metalens[f"{i}_17"].set_param_from_target_matrix(high_res_response)
                
                full_wave_response = high_res_response
                
                target_response = target_transfer_matrix[i]

                if target_response.shape != full_wave_response.shape:
                    assert full_wave_response.shape[-1] % target_response.shape[-1] == 0, f"{full_wave_response.shape[-1]} % {target_response.shape[-1]} != 0"
                    assert full_wave_response.shape[-2] % target_response.shape[-2] == 0, f"{full_wave_response.shape[-2]} % {target_response.shape[-2]} != 0"
                    assert full_wave_response.shape[-1] // target_response.shape[-1] == full_wave_response.shape[-2] // target_response.shape[-2]

                    ds_rate = full_wave_response.shape[-1] // target_response.shape[-1]

                    full_wave_response = full_wave_response[
                        ds_rate//2::ds_rate, 
                        :
                    ]
                    full_wave_response = full_wave_response.reshape(full_wave_response.shape[0], -1, ds_rate).sum(dim=-1)
                fig, ax = plt.subplots(2, 4, figsize=(20, 10))
                im0 = ax[0, 0].imshow(
                    target_response.abs().cpu().numpy()
                )
                ax[0, 0].set_title("Target Magnitude")
                fig.colorbar(im0, ax=ax[0, 0])
                im1 = ax[0, 1].imshow(
                    stitched_response[i].detach().abs().cpu().numpy()
                )
                ax[0, 1].set_title("Stitched Magnitude")
                fig.colorbar(im1, ax=ax[0, 1])
                im2 = ax[0, 2].imshow(
                    full_wave_response.abs().cpu().numpy()
                )
                ax[0, 2].set_title("Full Magnitude")
                fig.colorbar(im2, ax=ax[0, 2])
                im3 = ax[0, 3].imshow(
                    (
                        full_wave_response / torch.norm(full_wave_response, p=2) - target_response / torch.norm(target_response, p=2)
                    ).abs().cpu().numpy()
                )
                ax[0, 3].set_title("Difference Magnitude")
                fig.colorbar(im3, ax=ax[0, 3])

                im4 = ax[1, 0].imshow(
                    torch.angle(target_response).cpu().numpy()
                )
                ax[1, 0].set_title("Target Phase")
                fig.colorbar(im4, ax=ax[1, 0])
                im5 = ax[1, 1].imshow(
                    torch.angle(stitched_response[i].detach()).cpu().numpy()
                )
                ax[1, 1].set_title("Stitched Phase")
                fig.colorbar(im5, ax=ax[1, 1])
                im6 = ax[1, 2].imshow(
                    torch.angle(full_wave_response).cpu().numpy()
                )
                ax[1, 2].set_title("Full Phase")
                fig.colorbar(im6, ax=ax[1, 2])
                im7 = ax[1, 3].imshow(
                    torch.angle(
                        full_wave_response / torch.norm(full_wave_response, p=2) - target_response / torch.norm(target_response, p=2)
                    ).cpu().numpy()
                )
                ax[1, 3].set_title("Difference Phase")
                fig.colorbar(im7, ax=ax[1, 3])
                plt.savefig(configs.plot.plot_root + f"epoch-{out_epoch}_convid-{i}_full_lens.png")
                plt.close() 
        lg.info(f"this is the current pillar width: {current_pillar_width}")
        return stitched_response.detach(), current_pillar_width

def test(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    fp16: bool = False,
    stitched_response: torch.Tensor = None,
    plot: bool = False,
    test_train_loader: bool = False,
) -> None:
    model.eval()
    val_loss = 0
    correct = 0
    class_meter = AverageMeter("ce")
    with amp.autocast(enabled=fp16):
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if mixup_fn is not None:
                    data, target = mixup_fn(
                        data, target, random_state=i + 10000, vflip=False
                    )

                output = model(data)
                val_loss = criterion(output, target)
                # print("this is the criterion: ", criterion, flush=True)
                # print("this is the output: ", output[0], flush=True)
                # print("this is the target: ", target[0], flush=True)
                # print("this is the loss: ", val_loss, flush=True)
                # quit()
                class_meter.update(val_loss.item())
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

    loss_vector.append(class_meter.avg)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    accuracy_vector.append(accuracy)

    if not test_train_loader:
        lg.info(
            f"\nTest set: Average loss: {class_meter.avg:.4e}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
        )
        wandb.log(
            {
                "test_loss": class_meter.avg, 
                "test_acc": accuracy,
                "epoch": epoch,
            }
        )
    else:
        lg.info(
            f"\nFeasible train set: Average loss: {class_meter.avg:.4e}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
        )
        wandb.log(
            {
                "train_loss": class_meter.avg,
                "train_acc": accuracy,
                "epoch": epoch,
            }
        )

    if plot:
        for i in range(configs.model.conv_cfg.path_depth):
            # first, we need to read the transfer matrix of the metasurface trained in DONN
            if model.state_dict().get(f"features.conv1.conv._conv_pos.metalens.{i}_17.W_buffer") is not None:
                A = model.state_dict()[f"features.conv1.conv._conv_pos.metalens.{i}_17.W_buffer"]
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                im0 = ax[0].imshow(
                    A.abs().cpu().numpy()
                )
                ax[0].set_title("Magnitude")
                fig.colorbar(im0, ax=ax[0])
                im1 = ax[1].imshow(
                    torch.angle(A).cpu().numpy()
                )
                ax[1].set_title("Phase")
                fig.colorbar(im1, ax=ax[1])
                plt.savefig(configs.plot.plot_root + f"epoch-{epoch}_convid-{i}_test.png")
                plt.close()

    if test_train_loader:
        return class_meter.avg, accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if configs.model.linear_system:
        configs.model.optical_norm_cfg = None
        configs.model.optical_act_cfg = None

    if configs.model_test.linear_system:
        configs.model_test.optical_norm_cfg = None
        configs.model_test.optical_act_cfg = None

    if configs.model.hidden_channel_1 != 0 or configs.model.hidden_channel_2 != 0 or configs.model.hidden_channel_3 != 0:
        hidden_list = []
        if configs.model.hidden_channel_1 != 0:
            hidden_list.append(configs.model.hidden_channel_1)
        if configs.model.hidden_channel_2 != 0:
            hidden_list.append(configs.model.hidden_channel_2)
        if configs.model.hidden_channel_3 != 0:
            hidden_list.append(configs.model.hidden_channel_3)
        configs.model.hidden_list = hidden_list

    if configs.model_test.hidden_channel_1 != 0 or configs.model_test.hidden_channel_2 != 0 or configs.model_test.hidden_channel_3 != 0:
        hidden_list = []
        if configs.model_test.hidden_channel_1 != 0:
            hidden_list.append(configs.model_test.hidden_channel_1)
        if configs.model_test.hidden_channel_2 != 0:
            hidden_list.append(configs.model_test.hidden_channel_2)
        if configs.model_test.hidden_channel_3 != 0:
            hidden_list.append(configs.model_test.hidden_channel_3)
        configs.model_test.hidden_list = hidden_list
    assert configs.invdes.field_norm_condition == "wo_lens", "Only wo_lens is supported for now"
    lg.info(configs)
    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if bool(configs.run.deterministic):
        set_torch_deterministic()

    train_loader, validation_loader, test_loader = builder.make_dataloader(
        splits=["train", "valid", "test"]
    )

    if (
        configs.run.do_distill
        and configs.teacher is not None
        and os.path.exists(configs.teacher.checkpoint)
    ):
        teacher = builder.make_model(device, model_cfg=configs.teacher)
        load_model(teacher, path=configs.teacher.checkpoint)
        teacher.eval()
        lg.info(f"Load teacher model from {configs.teacher.checkpoint}")
    else:
        teacher = None

    model = builder.make_model(
        device,
        model_cfg=configs.model,
        random_state=(
            int(configs.run.random_state) if int(configs.run.deterministic) else None
        ),
    )
    lg.info(model)

    model_test = builder.make_model(
        device,
        model_cfg=configs.model_test,
        random_state=(
            int(configs.run.random_state) if int(configs.run.deterministic) else None
        ),
    )
    lg.info(model_test)

    # initialize the stitched_response
    with h5py.File(f"/home/pingchua/projects/MAPS/figs/metalens_TF_uniform_numA-{configs.invdes.num_atom}_wl-0.85_p-0.3_mat-Si/transfer_matrix.h5", "r") as f:
        transfer_matrix = f["transfer_matrix"][:]
        transfer_matrix = torch.tensor(transfer_matrix, dtype=torch.complex64)
        transfer_matrix = transfer_matrix.to(device)
        if configs.model.conv_cfg.max_tm_norm:
            transfer_matrix = transfer_matrix / torch.max(transfer_matrix.abs())
    stitched_response = []
    for i in range(configs.model.conv_cfg.path_depth):
        stitched_response.append(transfer_matrix)
    stitched_response = torch.stack(stitched_response, dim=0).to(device)

    # -----------------------------------------------
    # build the patch metalens for inverse projection
    csv_file = f"/home/pingchua/projects/MAPS/unitest/metaatom_phase_response_fsdx-0.3.csv"
    LUT = {}
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if float(row[0]) > 0.14:
                break
            LUT[float(row[0])] = float(row[1])

    patch_metalens = PatchMetalens(
        atom_period=0.3,
        patch_size=configs.invdes.patch_size,
        num_atom=configs.invdes.num_atom,
        probing_region_size=configs.invdes.patch_size,
        target_phase_response=None,
        LUT=LUT,
        device=device,
        target_dx=0.3,
        plot_root=configs.plot.plot_root,
        downsample_mode=configs.invdes.downsample_mode,
        downsample_method=configs.invdes.downsample_method,
        dz=configs.model.conv_cfg.delta_z_data,
        param_method=configs.invdes.param_method,
        tm_norm=configs.invdes.tm_norm,
        field_norm_condition=configs.invdes.field_norm_condition,
    )
    patch_metalens_list = [
        PatchMetalens(
            atom_period=0.3,
            patch_size=configs.invdes.patch_size,
            num_atom=configs.invdes.num_atom,
            probing_region_size=configs.invdes.patch_size,
            target_phase_response=None,
            LUT=LUT,
            device=device,
            target_dx=0.3,
            plot_root=configs.plot.plot_root,
            downsample_mode=configs.invdes.downsample_mode,
            downsample_method=configs.invdes.downsample_method,
            dz=configs.model.conv_cfg.delta_z_data,
            param_method=configs.invdes.param_method,
            tm_norm=configs.invdes.tm_norm,
            field_norm_condition=configs.invdes.field_norm_condition,
        ) for _ in range(configs.model.conv_cfg.path_depth)
    ]
        
    ensure_dir(configs.plot.plot_root)

    if configs.model.conv_cfg.near2far_method == "green_fn":
        near2far_matrix = probe_near2far_matrix(
            patch_metalens[0].total_opt if isinstance(patch_metalens, list) else patch_metalens.total_opt,
            configs.model.conv_cfg.lambda_data,
            device,
        )
        model_test.set_near2far_matrix(near2far_matrix)
        in_downsample_rate = configs.model.conv_cfg.in_downsample_rate
        out_downsample_rate = configs.model.conv_cfg.out_downsample_rate
        ds_near2far_matrix = near2far_matrix[
            out_downsample_rate//2::out_downsample_rate, 
            :
        ]
        ds_near2far_matrix = ds_near2far_matrix.reshape(ds_near2far_matrix.shape[0], -1, in_downsample_rate).sum(dim=-1)
        if configs.model.conv_cfg.calculate_in_hr:
            model.set_near2far_matrix(near2far_matrix)
        else:
            model.set_near2far_matrix(ds_near2far_matrix)

    # print(next(iter(test_loader))[0].shape)
    ## dummy forward to initialize quantizer
    model_test.set_test_mode()
    model(next(iter(test_loader))[0].to(device))
    model_test(next(iter(test_loader))[0].to(device))

    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(
        device
    )

    aux_criterions = dict()
    if configs.aux_criterion is not None:
        for name, config in configs.aux_criterion.items():
            if float(config.weight) > 0:
                try:
                    fn = builder.make_criterion(name, cfg=config)
                except NotImplementedError:
                    fn = name
                aux_criterions[name] = [fn, float(config.weight)]
    print(aux_criterions)
    if "mse_distill" in aux_criterions and teacher is not None:
        ## register hooks for teacher and student
        register_hidden_hooks(teacher)
        register_hidden_hooks(model)
        print(len([m for m in teacher.modules() if hasattr(m, "_recorded_hidden")]))
        print(len([m for m in teacher.modules() if hasattr(m, "_recorded_hidden")]))
        print("Register hidden state hooks for teacher and students")

    mixup_config = configs.dataset.augment
    mixup_fn = MixupAll(**mixup_config) if mixup_config is not None else None
    test_mixup_fn = (
        MixupAll(**configs.dataset.test_augment) if mixup_config is not None else None
    )
    grad_scaler = amp.GradScaler(enabled=getattr(configs.run, "fp16", False))
    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    wandb.login()
    tag = wandb.util.generate_id()
    group = f"{datetime.date.today()}"
    name = f"{configs.run.wandb.name}-{datetime.datetime.now().hour:02d}{datetime.datetime.now().minute:02d}{datetime.datetime.now().second:02d}-{tag}"
    configs.run.pid = os.getpid()
    run = wandb.init(
        project=configs.run.wandb.project,
        group=group,
        name=name,
        id=tag,
        config=configs,
    )

    lossv, accv = [0], [0]
    epoch = 0
    if configs.invdes.admm:
        # init the variables needed for ADMM
        admm_vars = {}
        admm_vars["rho_admm"] = configs.aux_criterion.admm_consistency.rho_admm
        admm_vars["z_admm"] = [
            stitched_response[i].clone().detach() for i in range(configs.model.conv_cfg.path_depth)
        ]
        admm_vars["u_admm"] = [
            torch.zeros_like(admm_vars["z_admm"][i]) for i in range(configs.model.conv_cfg.path_depth)
        ]
    else:
        admm_vars = None
    assert configs.invdes.reset_frequency in {"epoch", "proj"}, f"Unknown reset_frequency: {configs.invdes.reset_frequency}"
    try:
        if (
            int(configs.checkpoint.resume)
            and len(configs.checkpoint.restore_checkpoint) > 0
        ):
            load_model(
                model,
                configs.checkpoint.restore_checkpoint,
                ignore_size_mismatch=int(configs.checkpoint.no_linear),
            )

            lg.info("Validate resumed model...")
            test(
                model,
                validation_loader,
                0,
                criterion,
                lossv,
                accv,
                device,
                fp16=grad_scaler._enabled,
            )
            transfer_weights(model, model_test)
            # in previous validation, we have verify the loaded model is correct
            ls_knots = torch.tensor([[ 0.0456,  0.0490,  0.0296,  0.0349,  0.0358,  0.0420,  0.0444,  0.0611,
          0.0243,  0.0337, -0.0303,  0.0318,  0.0356,  0.0213,  0.0350,  0.0338,
          0.0476,  0.0220,  0.0334,  0.0335,  0.0435,  0.0219,  0.0337,  0.0307,
          0.0270,  0.0227,  0.0365,  0.0328,  0.0256,  0.0326,  0.0289,  0.0200],
        [ 0.0369,  0.0356,  0.0370,  0.0384,  0.0390,  0.0346,  0.0404,  0.0365,
          0.0312,  0.0392,  0.0399,  0.0326,  0.0411, -0.1342,  0.0323,  0.0351,
          0.0323,  0.0415,  0.0389,  0.0465,  0.0297,  0.0726,  0.0340,  0.0352,
          0.0384,  0.0325,  0.0349,  0.0375,  0.0344,  0.0377,  0.0205,  0.0311]]).to(device)
            with torch.no_grad():
                for i in range(configs.model.conv_cfg.path_depth):
                    _ = patch_metalens_list[i].forward(256)
            for i in range(configs.model_test.conv_cfg.path_depth):
                patch_metalens_list[i].direct_set_pillar_width(ls_knots[i])
                hr_response = probe_full_tm(
                    device=device,
                    patched_metalens=patch_metalens_list[i],
                )
                model_test.features.conv1.conv._conv_pos.metalens[f"{i}_17"].set_param_from_target_matrix(hr_response)
                ds_response = hr_response[
                    15//2::15, 
                    :
                ]
                ds_response = ds_response.reshape(ds_response.shape[0], -1, 15).sum(dim=-1)
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                im0 = ax[0].imshow(
                    ds_response.abs().cpu().numpy()
                )
                ax[0].set_title("ds_hr_tm")
                fig.colorbar(im0, ax=ax[0])
                im1 = ax[1].imshow(
                    model.state_dict()[f"features.conv1.conv._conv_pos.metalens.{i}_17.W_buffer"].detach().abs().cpu().numpy()
                )
                ax[1].set_title("lr_tm")
                fig.colorbar(im1, ax=ax[1])
                plt.savefig(f"./unitest/compare_{i}.png")
                plt.close()

            feasibleCE, feasibleAcc = test(
                model_test, # post project test on low res model
                train_loader,
                epoch,
                criterion,
                [],
                [],
                device,
                mixup_fn=test_mixup_fn,
                fp16=grad_scaler._enabled,
                stitched_response=stitched_response,
                plot = False,
                test_train_loader=True,
            )
            test(
                model_test, # post project test on low res model
                test_loader,
                epoch,
                criterion,
                [],
                [],
                device,
                mixup_fn=test_mixup_fn,
                fp16=grad_scaler._enabled,
                stitched_response=stitched_response,
                plot = False,
            )
        wandb.finish()
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()