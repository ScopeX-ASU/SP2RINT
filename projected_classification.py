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
from core import builder
from core.datasets.mixup import MixupAll
from core.utils import (
    get_parameter_group, 
    register_hidden_hooks, 
    probe_near2far_matrix, 
    CosSimLoss,
    reset_optimizer_and_scheduler,
    get_mid_weight,
)
from core.models.patch_metalens import PatchMetalens
import csv
from thirdparty.MAPS_old.core.utils import SharpnessScheduler
from thirdparty.MAPS_old.core.fdfd.pardiso_solver import pardisoSolver
from pyutils.general import ensure_dir
from matplotlib import pyplot as plt
import h5py
import copy
import numpy as np
import wandb
import datetime
import psutil
import torch

def overfit_single_batch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: Callable,
    device: torch.device = torch.device("cuda:0"),
    max_iters: int = 100,
    aux_criterions: Dict = {},
    grad_scaler: Optional[Callable] = None,
    teacher: Optional[nn.Module] = None,
):
    model.train()
    data_iter = iter(train_loader)
    data, target = next(data_iter)  # one batch only!
    data, target = data.to(device), target.to(device)

    for i in range(max_iters):
        with amp.autocast(enabled=grad_scaler._enabled):
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            class_loss = criterion(output, target)
            loss = class_loss

            for name, config in aux_criterions.items():
                aux_criterion, weight = config
                if name == "kd" and teacher is not None:
                    with torch.no_grad():
                        teacher_scores = teacher(data)
                    aux_loss = weight * aux_criterion(output, teacher_scores, target)
                    loss += aux_loss

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean().item()

        print(f"[Iter {i:3d}] Loss: {loss.item():.4f} | Acc: {acc*100:.2f}%", flush=True)

        if acc == 1.0:
            print(f"✅ Model successfully overfit the batch at iteration {i}!")
            break

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
                weight = patched_metalens.get_design_variables(
                    -0.05 * torch.ones_like(patched_metalens.get_pillar_width()),
                )
                _ = patched_metalens.total_opt(
                    sharpness=256, 
                    weight=weight,
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
                _ = patched_metalens.total_opt(
                    sharpness=256, 
                    weight=patched_metalens.get_design_variables(
                        patched_metalens.get_pillar_width(),
                    ),
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

def set_test_model_transfer_matrix(
    model, 
    model_test,
    patched_metalens,
    device,
):
    if model_test is not None:
        transfer_weights(model, model_test)

    for i in range(configs.model.conv_cfg.path_depth):
        current_width = model.state_dict()[f"features.conv1.conv._conv_pos.metalens.{i}_17.widths"]
        print(f"this is the current width: {current_width}", flush=True)
        # clip it to the range of [0.01, 0.28]
        current_width = torch.clamp(current_width, min=0.01, max=0.28)
        print(f"this is the current width after clipping: {current_width}", flush=True)
        current_weights = get_mid_weight(0.05, current_width)
        patched_metalens.direct_set_pillar_width(current_weights)
        hr_tm = probe_full_tm(
            device,
            patched_metalens,
            full_wave_down_sample_rate = 1,
        )
        model_test.features.conv1.conv._conv_pos.metalens[f"{i}_17"].set_param_from_target_matrix(hr_tm)

def design_using_LPA(
    model, 
    model_test,
    patched_metalens,
    device,
):
    if model_test is not None:
        transfer_weights(model, model_test)

    for i in range(configs.model.conv_cfg.path_depth):
        current_lr_tm = model.state_dict()[f"features.conv1.conv._conv_pos.metalens.{i}_1.W_buffer"]
        patched_metalens.set_target_phase_response(current_lr_tm)
        patched_metalens.rebuild_param()
        hr_tm = probe_full_tm(
            device,
            patched_metalens,
            full_wave_down_sample_rate = 1,
        )
        model_test.features.conv1.conv._conv_pos.metalens[f"{i}_1"].set_param_from_target_matrix(hr_tm)
    
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
        current_ds_near2far_matrix = near2far_matrix[
            in_downsample_rate//2::in_downsample_rate, 
            :
        ]
        current_ds_near2far_matrix = current_ds_near2far_matrix.reshape(current_ds_near2far_matrix.shape[0], -1, out_downsample_rate).sum(dim=-1)
        for i in range(configs.model.conv_cfg.path_depth):
            current_lens_tm = model.state_dict()[f"features.conv1.conv._conv_pos.metalens.{i}_17.W_buffer"]
            current_ds_near2far_matrix = current_ds_near2far_matrix.to(current_lens_tm.dtype)
            if target_entire_transfer_matrix is None:
                target_entire_transfer_matrix = current_ds_near2far_matrix @ current_lens_tm
            else:
                target_entire_transfer_matrix = current_ds_near2far_matrix @ current_lens_tm @ target_entire_transfer_matrix
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
                # if epoch == epoch_per_proj - 1:
                #     quit()
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
                        (i + 1) if getattr(configs.invdes.criterion, "weighted_response", False) else 0,
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

                # for p in patched_metalens.parameters():
                #     if p.grad is not None:
                #         print(f"this is the norm of the gradient: {p.grad.norm()}", flush=True)
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
                        (i + 1) if getattr(configs.invdes.criterion, "weighted_response", False) else 0
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
                    # assert full_wave_response.shape[-1] % target_response.shape[-1] == 0, f"{full_wave_response.shape[-1]} % {target_response.shape[-1]} != 0"
                    # assert full_wave_response.shape[-2] % target_response.shape[-2] == 0, f"{full_wave_response.shape[-2]} % {target_response.shape[-2]} != 0"
                    # assert full_wave_response.shape[-1] // target_response.shape[-1] == full_wave_response.shape[-2] // target_response.shape[-2]

                    # ds_rate = full_wave_response.shape[-1] // target_response.shape[-1]

                    # full_wave_response = full_wave_response[
                    #     ds_rate//2::ds_rate, 
                    #     :
                    # ]
                    # full_wave_response = full_wave_response.reshape(full_wave_response.shape[0], -1, ds_rate).sum(dim=-1)
                    upsampled_target_real = F.interpolate(target_response.real.unsqueeze(0).unsqueeze(0), size=full_wave_response.shape[-2:], mode="bilinear", align_corners=False).squeeze()
                    upsampled_target_imag = F.interpolate(target_response.imag.unsqueeze(0).unsqueeze(0), size=full_wave_response.shape[-2:], mode="bilinear", align_corners=False).squeeze()
                    upsampled_target_response = upsampled_target_real + 1j * upsampled_target_imag
                    ds_rate = full_wave_response.shape[-1] / target_response.shape[-1]
                    upsampled_target_response = upsampled_target_response / ds_rate
                
                else:
                    upsampled_target_response = target_response


                
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
                        full_wave_response / torch.norm(full_wave_response, p=2) - upsampled_target_response / torch.norm(upsampled_target_response, p=2)
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
                        full_wave_response / torch.norm(full_wave_response, p=2) - upsampled_target_response / torch.norm(upsampled_target_response, p=2)
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
                    1 if getattr(configs.invdes.criterion, "weighted_response", False) else 0,
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
                    # assert full_wave_response.shape[-1] % target_response.shape[-1] == 0, f"{full_wave_response.shape[-1]} % {target_response.shape[-1]} != 0"
                    # assert full_wave_response.shape[-2] % target_response.shape[-2] == 0, f"{full_wave_response.shape[-2]} % {target_response.shape[-2]} != 0"
                    # assert full_wave_response.shape[-1] // target_response.shape[-1] == full_wave_response.shape[-2] // target_response.shape[-2]

                    # ds_rate = full_wave_response.shape[-1] // target_response.shape[-1]

                    # full_wave_response = full_wave_response[
                    #     ds_rate//2::ds_rate, 
                    #     :
                    # ]
                    # full_wave_response = full_wave_response.reshape(full_wave_response.shape[0], -1, ds_rate).sum(dim=-1)
                    upsampled_target_real = F.interpolate(target_response.real.unsqueeze(0).unsqueeze(0), size=full_wave_response.shape[-2:], mode="bilinear", align_corners=False).squeeze()
                    upsampled_target_imag = F.interpolate(target_response.imag.unsqueeze(0).unsqueeze(0), size=full_wave_response.shape[-2:], mode="bilinear", align_corners=False).squeeze()
                    upsampled_target_response = upsampled_target_real + 1j * upsampled_target_imag
                    ds_rate = full_wave_response.shape[-1] / target_response.shape[-1]
                    upsampled_target_response = upsampled_target_response / ds_rate

                else:
                    upsampled_target_response = target_response

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
                        full_wave_response / torch.norm(full_wave_response, p=2) - upsampled_target_response / torch.norm(upsampled_target_response, p=2)
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
                        full_wave_response / torch.norm(full_wave_response, p=2) - upsampled_target_response / torch.norm(upsampled_target_response, p=2)
                    ).cpu().numpy()
                )
                ax[1, 3].set_title("Difference Phase")
                fig.colorbar(im7, ax=ax[1, 3])
                plt.savefig(configs.plot.plot_root + f"epoch-{out_epoch}_convid-{i}_full_lens.png")
                plt.close() 
        lg.info(f"this is the current pillar width: {current_pillar_width}")
        return stitched_response.detach(), current_pillar_width


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    aux_criterions: Dict,
    mixup_fn: Callable = None,
    device: torch.device = torch.device("cuda:0"),
    grad_scaler: Optional[Callable] = None,
    teacher: Optional[nn.Module] = None,
    stitched_response: torch.Tensor = None,
    plot: bool = False,
    # params for inverse design projection
    patched_metalens: Optional[PatchMetalens] = None,
    patched_metalens_list: Optional[List[PatchMetalens]] = None,
    invdes_criterion: Optional[Callable] = None,  
    current_pillar_width: Optional[torch.Tensor] = None,
    invdes_lr: tuple = None,
    invdes_sharp: tuple = None,
    in_downsample_rate_scheduler: Optional[Scheduler] = None,
    out_downsample_rate_scheduler: Optional[Scheduler] = None,
    near2far_matrix: Optional[torch.Tensor] = None,
    ds_near2far_matrix: Optional[torch.Tensor] = None,
    admm_vars = None,
) -> None:
    # the invdes_lr looks like (5e-3, 5e-5)
    # the invdes_sharp looks like (10, 256)
    assert len(invdes_lr) == 2, f"len(invdes_lr) != 2, {invdes_lr}"
    assert len(invdes_sharp) == 2, f"len(invdes_sharp) != 2, {invdes_sharp}"
    assert configs.invdes.num_epoch % configs.invdes.epoch_per_proj == 0, f"{configs.invdes.num_epoch} % {configs.invdes.epoch_per_proj} != 0"
    if configs.model.conv_cfg.TM_model_method == "end2end":
        end2end_sharpness_scheduler = builder.make_scheduler(
            optimizer=None, 
            name="end2end_sharpness",
            cfgs=configs.end2end_sharpness_scheduler,
        )
    else:
        end2end_sharpness_scheduler = None
    num_proj_during_train = round(configs.invdes.num_epoch // configs.invdes.epoch_per_proj) - 1 # we must do a projection at the end of the training
    proj_batch_interval = len(train_loader) // (num_proj_during_train + 1)
    current_project_count = 0
    invdes_array = np.linspace(invdes_lr[0], invdes_lr[1], num_proj_during_train + 2) # * get_learning_rate(optimizer) / configs.optimizer.lr
    invdes_sharp_array = np.linspace(invdes_sharp[0], invdes_sharp[1], num_proj_during_train + 2)
    # print("this is the invdes_array: ", invdes_array, flush=True)
    # print("this is the invdes_sharp_array: ", invdes_sharp_array, flush=True)
    model.train()
    step = epoch * len(train_loader)

    class_meter = AverageMeter("ce")
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}
    data_counter = 0
    correct = 0
    total_data = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        if configs.model.conv_cfg.TM_model_method == "end2end":
            sharpness = end2end_sharpness_scheduler.step()
            print(f"this is the sharpness: {sharpness}", flush=True)
            if sharpness >= configs.end2end_sharpness_scheduler.final_sharpness:
                break
        else:
            sharpness = None
        data = data.to(device, non_blocking=True)
        data_counter += data.shape[0]

        target = target.to(device, non_blocking=True)
        if mixup_fn is not None:
            data, target = mixup_fn(data, target)

        with amp.autocast(enabled=grad_scaler._enabled):
            output = model(data, sharpness)
            if isinstance(output, tuple):
                output, inner_fields = output
            class_loss = criterion(output, target)
            class_meter.update(class_loss.item())
            loss = class_loss
            # if batch_idx == len(train_loader) - 1:
            #     print("this is the criterion: ", criterion, flush=True)
            #     print("this is the output: ", output[0], flush=True)
            #     print("this is the target: ", target[0], flush=True)
            #     print("this is the loss: ", loss, flush=True)

            
            smoothing_loss = 0
            # if hasattr(model, "get_smoothing_loss"):
            #     # you can pass in a hyperparameter e.g. 1e-3
            #     smoothing_loss = model.get_smoothing_loss(lambda_smooth=1e-3)
            #     loss += smoothing_loss
            intensity_loss = 0
            # lambda_intensity = 1e-3
            # total_intensity = output.sum(dim=1).mean() 
            # intensity_loss = -lambda_intensity * total_intensity
            # loss += intensity_loss
            
            for name, config in aux_criterions.items():
                aux_criterion, weight = config
                aux_loss = 0
                if name in {"kd", "dkd"} and teacher is not None:
                    with torch.no_grad():
                        teacher_scores = teacher(data).data.detach()
                    aux_loss = weight * aux_criterion(output, teacher_scores, target)
                elif name == "mse_distill" and teacher is not None:
                    with torch.no_grad():
                        teacher(data).data.detach()
                    teacher_hiddens = [
                        m._recorded_hidden
                        for m in teacher.modules()
                        if hasattr(m, "_recorded_hidden")
                    ]
                    student_hiddens = [
                        m._recorded_hidden
                        for m in model.modules()
                        if hasattr(m, "_recorded_hidden")
                    ]

                    aux_loss = weight * sum(
                        F.mse_loss(h1, h2)
                        for h1, h2 in zip(teacher_hiddens, student_hiddens)
                    )
                elif name == "distance_constraint": # this prevent the wegiths from being too faraway from the initial weights
                    assert stitched_response is not None
                    aux_loss = weight * aux_criterion(stitched_response, model.features.conv1.conv._conv_pos.equivalent_W)
                elif name == "smooth_penalty":
                    aux_loss = weight * aux_criterion(model.features.conv1.conv._conv_pos.equivalent_W)
                elif name == "admm":
                    aux_loss = weight * aux_criterion(model.features.conv1.conv._conv_pos.equivalent_W, admm_vars)
                elif name == "activation_smooth":
                    aux_loss = weight * aux_criterion(inner_fields)
                loss = loss + aux_loss
                aux_meters[name].update(aux_loss)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).sum().item()

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                # print(f"the grad of name {name} is: {p.grad.data.norm()}", flush=True)
                if "features.conv1.conv._conv_pos.metalens" in name:
                    print(f"the grad of name {name} is: {p.grad.data.norm()}", flush=True)
                    print(f"the param of name {name} is: {p.data}", flush=True)
                    

        grad_scaler.unscale_(optimizer)
        if configs.run.grad_clip:
            torch.nn.utils.clip_grad_value_(
                [p for p in model.parameters() if p.requires_grad],
                float(configs.run.max_grad_value),
            )
        grad_scaler.step(optimizer)
        grad_scaler.update()
        step += 1
        # print(list(model.named_parameters()))
        # Clip all weight into -pi to pi
        # model.phase_rounding()
        # model.update_lambda_pixel_size()
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Parameter: {name}")
        #         print(f"Grad: {param.grad.data.norm()}")
        if batch_idx % int(configs.run.log_interval) == 0:
            log = "Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4e} class Loss: {:.4e}".format(
                epoch,
                data_counter,
                total_data,
                100.0 * data_counter / total_data,
                loss.data.item(),
                class_loss.data.item(),
            )
            if smoothing_loss != 0.0:
                log += f" smooth: {float(smoothing_loss):.4e}"
            if intensity_loss != 0.0:
                log += f" intens: {float(intensity_loss):.4e}"
            for name, aux_meter in aux_meters.items():
                log += f" {name}: {aux_meter.val:.4e}"
            lg.info(log)

        # begin to project the weights to the implementable subspace
        if batch_idx % proj_batch_interval == 0 and batch_idx != 0 and current_project_count < num_proj_during_train and configs.run.project_GD:
            if configs.invdes.reset_frequency == "epoch":
                invdes_lr_to_pass = tuple(invdes_array[current_project_count: current_project_count + 2])
                invdes_sharp_to_pass = tuple(invdes_sharp_array[current_project_count: current_project_count + 2])
            elif configs.invdes.reset_frequency == "proj":
                invdes_lr_to_pass = invdes_lr
                invdes_sharp_to_pass = invdes_sharp
            else:
                raise NotImplementedError(f"Unknown reset_frequency: {configs.invdes.reset_frequency}")
            # print("passing lr to the inverse design: ", invdes_lr_to_pass, flush=True)
            # print("passing sharp to the inverse design: ", invdes_sharp_to_pass, flush=True)
            current_project_count += 1
            stitched_response, current_pillar_width = project_to_implementable_subspace(
                model, 
                model_test=None,
                patched_metalens=patched_metalens,
                patched_metalens_list=patched_metalens_list,
                invdes_criterion=invdes_criterion,
                prev_pillar_width=current_pillar_width,
                out_epoch=epoch,
                proj_idx=current_project_count, 
                in_downsample_rate_scheduler=in_downsample_rate_scheduler,
                out_downsample_rate_scheduler=out_downsample_rate_scheduler,
                near2far_matrix=near2far_matrix,
                ds_near2far_matrix=ds_near2far_matrix,
                device=device,
                probe_full_wave=False,
                invdes_lr=invdes_lr_to_pass,
                invdes_sharp=invdes_sharp_to_pass if configs.invdes_sharpness_scheduler.mode != "per_proj" else (configs.invdes_sharpness_scheduler.init_sharpness, configs.invdes_sharpness_scheduler.final_sharpness),
                layer_wise_matching=configs.invdes.layer_wise_matching,
                epoch_per_proj=configs.invdes.epoch_per_proj,
                finetune_entire = False if configs.invdes.layer_wise_matching else True,
                match_entire_epoch=configs.invdes.epoch_per_proj,
                admm_vars=admm_vars,
            )

    scheduler.step()
    avg_class_loss = class_meter.avg
    if sharpness is None:
        accuracy = 100.0 * correct / total_data
    else:
        accuracy = 100.0 * correct / data_counter
    if model.darcy:
        lg.info(
            f"Train Loss: {avg_class_loss:.4e}"
        )
    else:
        lg.info(
            f"Train class Loss: {avg_class_loss:.4e}, Accuracy: {correct}/{total_data if sharpness is None else data_counter} ({accuracy:.2f}%)"
        )
    wandb.log(
        {
            "train_loss": avg_class_loss,
            "train_acc": accuracy,
            "epoch": epoch,
            "lr": get_learning_rate(optimizer),
        }
    )
    if plot:
        # plot the transfer matrix trained in DONN
        if model.darcy:
            num_plot_sample = min(3, data.shape[0])
            fig, ax = plt.subplots(num_plot_sample, 3, figsize=(num_plot_sample * 5, 15))
            for i in range(num_plot_sample):
                im0 = ax[i, 0].imshow(
                    data[i, 0].cpu().numpy()
                )
                ax[i, 0].set_title("Input")
                fig.colorbar(im0, ax=ax[i, 0])
                im1=ax[i, 1].imshow(
                    output[i, 0].detach().cpu().numpy()
                )
                ax[i, 1].set_title("Output")
                fig.colorbar(im1, ax=ax[i, 1])
                im2=ax[i, 2].imshow(
                    target[i, 0].cpu().numpy()
                )
                ax[i, 2].set_title("Target")
                fig.colorbar(im2, ax=ax[i, 2])
            plt.savefig(configs.plot.plot_root + f"darcy_field_epoch-{epoch}_train.png")
            plt.close()
        else:
            pass

        # for i in range(configs.model.conv_cfg.path_depth):
        #     # first, we need to read the transfer matrix of the metasurface trained in DONN
        #     if model.state_dict().get(f"features.conv1.conv._conv_pos.metalens.{i}_17.W_buffer") is not None:
        #         A = model.state_dict()[f"features.conv1.conv._conv_pos.metalens.{i}_17.W_buffer"]
        #         fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        #         im0 = ax[0].imshow(
        #             A.abs().cpu().numpy()
        #         )
        #         ax[0].set_title("Magnitude")
        #         fig.colorbar(im0, ax=ax[0])
        #         im1 = ax[1].imshow(
        #             torch.angle(A).cpu().numpy()
        #         )
        #         ax[1].set_title("Phase")
        #         fig.colorbar(im1, ax=ax[1])
        #         plt.savefig(configs.plot.plot_root + f"epoch-{epoch}_convid-{i}_train.png")
        #         plt.close()

    if not configs.run.project_GD:
        return invdes_lr, invdes_sharp, stitched_response, current_pillar_width, avg_class_loss, accuracy
    elif configs.invdes.reset_frequency == "epoch" and configs.run.project_GD:
        remaining_invdes_lr = tuple(invdes_array[current_project_count:])
        remaining_invdes_sharp = tuple(invdes_sharp_array[current_project_count:])
        assert len(remaining_invdes_lr) == 2, f"len(remaining_invdes_lr) != 2, {remaining_invdes_lr}"
        assert len(remaining_invdes_sharp) == 2, f"len(remaining_invdes_sharp) != 2, {remaining_invdes_sharp}"
        return remaining_invdes_lr, remaining_invdes_sharp, stitched_response, current_pillar_width, avg_class_loss, accuracy
    elif configs.invdes.reset_frequency == "proj":
        return invdes_lr, invdes_sharp, stitched_response, current_pillar_width, avg_class_loss, accuracy
    else:
        raise NotImplementedError(f"Unknown reset_frequency: {configs.invdes.reset_frequency}")


def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    aux_criterions: Dict,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    fp16: bool = False,
    stitched_response: torch.Tensor = None,
    plot: bool = False,
) -> None:
    model.eval()
    if configs.model.conv_cfg.TM_model_method == "end2end":
        model.set_test_mode(True)
    val_loss = 0
    correct = 0
    class_meter = AverageMeter("ce")
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}
    with amp.autocast(enabled=fp16):
        with torch.no_grad():
            for i, (data, target) in enumerate(validation_loader):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if mixup_fn is not None:
                    data, target = mixup_fn(data, target, random_state=i, vflip=False)

                output = model(
                    data,
                    sharpness=None if configs.model.conv_cfg.TM_model_method != "end2end" else 256
                )
                if isinstance(output, tuple):
                    output, inner_fields = output

                val_loss = criterion(output, target)
                class_meter.update(val_loss.item())
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # for name, config in aux_criterions.items():
                #     aux_criterion, weight = config
                #     aux_loss = 0
                #     if name == "distance_constraint": # this prevent the wegiths from being too faraway from the initial weights
                #         assert stitched_response is not None
                #         aux_loss = weight * aux_criterion(stitched_response, model.features.conv1.conv._conv_pos.equivalent_W)
                #     else:
                #         raise NotImplementedError
                #     val_loss = val_loss + aux_loss
                #     aux_meters[name].update(aux_loss)

    loss_vector.append(class_meter.avg)
    accuracy = 100.0 * correct / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    if model.darcy:
        log = f"\nValidation set: Average loss: {class_meter.avg:.4e}\n"
    else:
        log = f"\nValidation set: Average loss: {class_meter.avg:.4e}, Accuracy: {correct}/{len(validation_loader.dataset)} ({accuracy:.2f}%)\n"
    for name, aux_meter in aux_meters.items():
        log += f" {name}: {aux_meter.avg:.4e}"
    lg.info(
        log
    )

    if configs.model.conv_cfg.TM_model_method == "end2end":
        model.set_test_mode(False)

    wandb.log(
        {
            "val_loss": class_meter.avg,
            "val_acc": accuracy,
            "epoch": epoch,
        }
    )
    if plot:
        if model.darcy:
            num_plot_sample = min(3, data.shape[0])
            fig, ax = plt.subplots(num_plot_sample, 3, figsize=(num_plot_sample * 5, 15))
            for i in range(num_plot_sample):
                im0 = ax[i, 0].imshow(
                    data[i, 0].cpu().numpy()
                )
                ax[i, 0].set_title("Input")
                fig.colorbar(im0, ax=ax[i, 0])
                im1=ax[i, 1].imshow(
                    output[i, 0].detach().cpu().numpy()
                )
                ax[i, 1].set_title("Output")
                fig.colorbar(im1, ax=ax[i, 1])
                im2=ax[i, 2].imshow(
                    target[i, 0].cpu().numpy()
                )
                ax[i, 2].set_title("Target")
                fig.colorbar(im2, ax=ax[i, 2])
            plt.savefig(configs.plot.plot_root + f"darcy_field_epoch-{epoch}_valid.png")
            plt.close()
        # for i in range(configs.model.conv_cfg.path_depth):
        #     # first, we need to read the transfer matrix of the metasurface trained in DONN
        #     if model.state_dict().get(f"features.conv1.conv._conv_pos.metalens.{i}_17.W_buffer") is not None:
        #         A = model.state_dict()[f"features.conv1.conv._conv_pos.metalens.{i}_17.W_buffer"]
        #         fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        #         im0 = ax[0].imshow(
        #             A.abs().cpu().numpy()
        #         )
        #         ax[0].set_title("Magnitude")
        #         fig.colorbar(im0, ax=ax[0])
        #         im1 = ax[1].imshow(
        #             torch.angle(A).cpu().numpy()
        #         )
        #         ax[1].set_title("Phase")
        #         fig.colorbar(im1, ax=ax[1])
        #         plt.savefig(configs.plot.plot_root + f"epoch-{epoch}_convid-{i}_valid.png")
        #         plt.close()


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
    if configs.model.conv_cfg.TM_model_method == "end2end":
        model.set_test_mode(True)
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

                output = model(
                    data,
                    sharpness=None if configs.model.conv_cfg.TM_model_method != "end2end" else 256
                )
                if isinstance(output, tuple):
                    output, inner_fields = output
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

    if configs.model.conv_cfg.TM_model_method == "end2end":
        model.set_test_mode(False)

    if not test_train_loader:
        if model.darcy:
            lg.info(
                f"\nTest set: Average loss: {class_meter.avg:.4e}\n"
            )
        else:
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
        if model.darcy:
            lg.info(
                f"\nFeasible train set: Average loss: {class_meter.avg:.4e}\n"
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
        if model.darcy:
            num_plot_sample = min(3, data.shape[0])
            fig, ax = plt.subplots(num_plot_sample, 3, figsize=(num_plot_sample * 5, 15))
            for i in range(num_plot_sample):
                im0 = ax[i, 0].imshow(
                    data[i, 0].cpu().numpy()
                )
                ax[i, 0].set_title("Input")
                fig.colorbar(im0, ax=ax[i, 0])
                im1=ax[i, 1].imshow(
                    output[i, 0].detach().cpu().numpy()
                )
                ax[i, 1].set_title("Output")
                fig.colorbar(im1, ax=ax[i, 1])
                im2=ax[i, 2].imshow(
                    target[i, 0].cpu().numpy()
                )
                ax[i, 2].set_title("Target")
                fig.colorbar(im2, ax=ax[i, 2])
            plt.savefig(configs.plot.plot_root + f"darcy_field_epoch-{epoch}_test.png")
            plt.close()
        # for i in range(configs.model.conv_cfg.path_depth):
        #     # first, we need to read the transfer matrix of the metasurface trained in DONN
        #     if model.state_dict().get(f"features.conv1.conv._conv_pos.metalens.{i}_17.W_buffer") is not None:
        #         A = model.state_dict()[f"features.conv1.conv._conv_pos.metalens.{i}_17.W_buffer"]
        #         fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        #         im0 = ax[0].imshow(
        #             A.abs().cpu().numpy()
        #         )
        #         ax[0].set_title("Magnitude")
        #         fig.colorbar(im0, ax=ax[0])
        #         im1 = ax[1].imshow(
        #             torch.angle(A).cpu().numpy()
        #         )
        #         ax[1].set_title("Phase")
        #         fig.colorbar(im1, ax=ax[1])
        #         plt.savefig(configs.plot.plot_root + f"epoch-{epoch}_convid-{i}_test.png")
        #         plt.close()

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
    csv_file = f"core/metaatom_response_fsdx-0.3.csv"
    LUT = {}
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if float(row[0]) > 0.14:
                break
            LUT[float(row[0])] = float(row[1])

    LUT_height = {}
    csv_file_height = f"core/metaatom_response_fsdx-0.3_height.csv"
    with open(csv_file_height, mode='r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if float(row[0]) <= 0.189:
                continue
            LUT_height[float(row[0])] = float(row[2])

    patch_metalens = PatchMetalens(
        atom_period=0.3,
        patch_size=configs.invdes.patch_size,
        num_atom=configs.invdes.num_atom,
        probing_region_size=configs.invdes.patch_size,
        target_phase_response=None,
        LUT=LUT if configs.invdes.design_var_type == "width" else LUT_height,
        device=device,
        target_dx=0.3,
        plot_root=configs.plot.plot_root,
        downsample_mode=configs.invdes.downsample_mode,
        downsample_method=configs.invdes.downsample_method,
        dz=configs.model.conv_cfg.delta_z_data,
        param_method=configs.invdes.param_method,
        tm_norm=configs.invdes.tm_norm,
        field_norm_condition=configs.invdes.field_norm_condition,
        design_var_type=configs.invdes.design_var_type, # width or height
        atom_width=configs.invdes.atom_width,
    )
    if configs.invdes.finetune_entire:
        patch_metalens_list = [
            PatchMetalens(
                atom_period=0.3,
                patch_size=configs.invdes.patch_size,
                num_atom=configs.invdes.num_atom,
                probing_region_size=configs.invdes.patch_size,
                target_phase_response=None,
                LUT=LUT if configs.invdes.design_var_type == "width" else LUT_height,
                device=device,
                target_dx=0.3,
                plot_root=configs.plot.plot_root,
                downsample_mode=configs.invdes.downsample_mode,
                downsample_method=configs.invdes.downsample_method,
                dz=configs.model.conv_cfg.delta_z_data,
                param_method=configs.invdes.param_method,
                tm_norm=configs.invdes.tm_norm,
                field_norm_condition=configs.invdes.field_norm_condition,
                design_var_type=configs.invdes.design_var_type, # width or height
                atom_width=configs.invdes.atom_width,
            ) for _ in range(configs.model.conv_cfg.path_depth)
        ]
    else:
        patch_metalens_list = None
        
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

        # plt.figure()
        # plt.imshow(near2far_matrix.abs().cpu().numpy())
        # plt.colorbar()
        # plt.savefig(f"./figs/near2far_matrix.png")
        # plt.close()

        # plt.figure()
        # plt.imshow(ds_near2far_matrix.abs().cpu().numpy())
        # plt.colorbar()
        # plt.savefig(f"./figs/near2far_matrix_ds.png")
        # plt.close()
        # quit()

    # print(next(iter(test_loader))[0].shape)
    ## dummy forward to initialize quantizer
    model_test.set_test_mode()
    model(
        next(iter(test_loader))[0].to(device),
        256 if configs.model.conv_cfg.TM_model_method == "end2end" else None
    )
    model_test(
        next(iter(test_loader))[0].to(device),
        # 256 if configs.model.conv_cfg.TM_model_method == "end2end" else None
    )

    # if int(configs.checkpoint.resume) or getattr(configs.run, "uniform_metasurface", False): # we freeze the training of features if we transfer learning
    if getattr(configs.run, "uniform_metasurface", False): # we freeze the training of features if we transfer learning
        assert not configs.run.project_GD, "You must not set project_GD to True when resuming the training"
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model_test.features.parameters():
            param.requires_grad = False
        optimizer = builder.make_optimizer(
            [p for p in model_test.parameters() if p.requires_grad],
            name=configs.optimizer.name,
            opt_configs=configs.optimizer,
        )
    else:
        optimizer = builder.make_optimizer(
            get_parameter_group(model, weight_decay=float(configs.optimizer.weight_decay)),
            name=configs.optimizer.name,
            opt_configs=configs.optimizer,
        )
    scheduler = builder.make_scheduler(optimizer)
    in_downsample_rate_scheduler = builder.make_scheduler(optimizer=None, name="downsample_rate", cfgs=configs.in_downsample_rate_scheduler)
    out_downsample_rate_scheduler = builder.make_scheduler(optimizer=None, name="downsample_rate", cfgs=configs.out_downsample_rate_scheduler)
    invdes_sharpness_scheduler = builder.make_scheduler(optimizer=None, name="invdes_sharpness", cfgs=configs.invdes_sharpness_scheduler)
    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(
        device
    )
    invdes_criterion = builder.make_criterion(configs.invdes.criterion.name, configs.invdes.criterion).to(
        device
    )
    CosSim_criterion = builder.make_criterion("cosine_similarity")

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
    saver = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=True if not model.darcy else False,
        truncate=2,
        metric_name="acc",
        format="{:.2f}",
    )
    test_model_saver = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=True if not model.darcy else False,
        truncate=2,
        metric_name="acc",
        format="{:.2f}",
    )
    grad_scaler = amp.GradScaler(enabled=getattr(configs.run, "fp16", False))
    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    # mlflow.set_experiment(configs.run.experiment)
    # experiment = mlflow.get_experiment_by_name(configs.run.experiment)

    # run_id_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # mlflow.start_run(run_name=model_name)
    # mlflow.log_params(
    #     {
    #         "exp_name": configs.run.experiment,
    #         "exp_id": experiment.experiment_id,
    #         "run_id": mlflow.active_run().info.run_id,
    #         "init_lr": configs.optimizer.lr,
    #         "checkpoint": checkpoint,
    #         "restore_checkpoint": configs.checkpoint.restore_checkpoint,
    #         "pid": os.getpid(),
    #     }
    # )
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
    current_pillar_width = None
    worsenCE_r = []
    worsenCE_a = []
    worsenAcc_r = []
    worsenAcc_a = []

    worsenCE_r_lr_model = []
    worsenCE_a_lr_model = []
    worsenAcc_r_lr_model = []
    worsenAcc_a_lr_model = []
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
        lg.info(
            f"Experiment {name} starts. Group: {group}, Run ID: ({run.id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})"
            # f"Experiment {configs.run.experiment} ({experiment.experiment_id}) starts. Run ID: ({mlflow.active_run().info.run_id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})"
        )
        if (
            int(configs.checkpoint.resume)
            and len(configs.checkpoint.restore_checkpoint) > 0
        ):
            load_model(
                model,
                configs.checkpoint.restore_checkpoint,
                ignore_size_mismatch=int(configs.checkpoint.no_linear),
            )
            load_model(
                model_test,
                configs.checkpoint.restore_test_checkpoint,
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
            current_pillar_width = [[ 0.0288,  0.0349,  0.0217,  0.0293,  0.0301,  0.0369,  0.0208,  0.0279,
          0.0295,  0.0370,  0.0291, -0.0057,  0.0300,  0.0329,  0.0154,  0.0255,
          0.0376,  0.0226,  0.0343,  0.0179,  0.0305,  0.0308,  0.0348,  0.0065,
          0.0353,  0.0211,  0.0295,  0.0372,  0.0304,  0.0144,  0.0374,  0.0251],
        [ 0.0296,  0.0355,  0.0218,  0.0383,  0.0364,  0.0303,  0.0333,  0.0396,
          0.0350,  0.0354,  0.0323,  0.0303,  0.0358,  0.0387,  0.0287,  0.0302,
          0.0353,  0.0308,  0.0338,  0.0339,  0.0374,  0.0333,  0.0393,  0.0244,
          0.0414,  0.0311,  0.0360,  0.0301,  0.0330,  0.0293,  0.0194,  0.0304]]
            current_pillar_width = torch.tensor(current_pillar_width, dtype=torch.float32).to(device)

            # we swap the model and test and only train the test model digital part
            # dummy_model = model
            # model = model_test
            # model_test = dummy_model
            # assert not configs.run.project_GD, "You must not set project_GD to True when resuming the training"
        elif getattr(configs.run, "uniform_metasurface", False):
            uniform_ls_knots = 0.05 * torch.ones(configs.model.conv_cfg.length).to(device)
            patch_metalens.direct_set_pillar_width(uniform_ls_knots)
            uniform_metasurface_tm = probe_full_tm(
                device=device,
                patched_metalens=patch_metalens,
                full_wave_down_sample_rate = 1,
            )
            for i in range(configs.model.conv_cfg.path_depth):
                model_test.features.conv1.conv._conv_pos.metalens[f"{i}_17"].set_param_from_target_matrix(uniform_metasurface_tm)
            dummy_model = model
            model = model_test
            model_test = dummy_model
            assert not configs.run.project_GD, "You must not set project_GD to True when resuming the training"
        # overfit_single_batch(
        #     model=model,
        #     train_loader=train_loader,
        #     optimizer=optimizer,
        #     criterion=criterion,
        #     device=device,
        #     max_iters=10000,
        #     aux_criterions=aux_criterions,
        #     grad_scaler=grad_scaler,
        #     teacher=None,
        # )
        # quit()
        for epoch in range(1, int(configs.run.n_epochs) + 1):
            # reset the optimizer and scheduler every epoch
            if configs.invdes.adaptive_invdes_lr:
                invdes_lr = (
                    configs.invdes.lr * (get_learning_rate(optimizer) / configs.optimizer.lr), 
                    configs.invdes.lr * 1e-2 * (get_learning_rate(optimizer) / configs.optimizer.lr)
                )
            else:
                invdes_lr = (configs.invdes.lr, configs.invdes.lr * 1e-2)
            print(f"in epoch: {epoch}, this is the invdes_lr about to use: ", invdes_lr, flush=True)
            invdes_sharp = invdes_sharpness_scheduler.step()
            print(f"in epoch: {epoch}, this is the invdes_sharpness about to use: ", invdes_sharp, flush=True)
            with torch.autograd.set_detect_anomaly(True):
                invdes_lr, invdes_sharp, stitched_response, current_pillar_width, idealCE, idealAcc = train(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    epoch,
                    criterion,
                    aux_criterions,
                    mixup_fn,
                    device,
                    grad_scaler=grad_scaler,
                    teacher=teacher,
                    stitched_response=stitched_response,
                    plot = True,
                    # parameters for inverse design
                    patched_metalens=patch_metalens,
                    patched_metalens_list=patch_metalens_list,
                    invdes_criterion=invdes_criterion,
                    current_pillar_width=current_pillar_width,
                    invdes_lr=invdes_lr,
                    invdes_sharp=invdes_sharp,
                    in_downsample_rate_scheduler=in_downsample_rate_scheduler,
                    out_downsample_rate_scheduler=out_downsample_rate_scheduler,
                    near2far_matrix=near2far_matrix,
                    ds_near2far_matrix=ds_near2far_matrix,
                    admm_vars=admm_vars,
                )
            if configs.run.project_GD:
                if validation_loader is not None:
                    validate( # pre project valid
                        model,
                        validation_loader,
                        epoch,
                        criterion,
                        aux_criterions,
                        [],
                        [],
                        device,
                        mixup_fn=test_mixup_fn,
                        fp16=grad_scaler._enabled,
                        stitched_response=stitched_response,
                        plot = False,
                    )
                test( # pre project test
                    model,
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
                stitched_response, current_pillar_width = project_to_implementable_subspace(
                    model, 
                    model_test,
                    patch_metalens,
                    patch_metalens_list,
                    invdes_criterion,
                    current_pillar_width,
                    epoch,
                    "post_train", 
                    in_downsample_rate_scheduler,
                    out_downsample_rate_scheduler,
                    near2far_matrix,
                    ds_near2far_matrix,
                    device,
                    probe_full_wave=True,
                    invdes_lr=invdes_lr,
                    invdes_sharp=invdes_sharp if configs.invdes_sharpness_scheduler.mode != "per_proj" else (configs.invdes_sharpness_scheduler.init_sharpness, configs.invdes_sharpness_scheduler.final_sharpness),
                    layer_wise_matching=configs.invdes.layer_wise_matching,
                    epoch_per_proj=configs.invdes.epoch_per_proj,
                    finetune_entire=configs.invdes.finetune_entire,
                    match_entire_epoch=5 if configs.invdes.layer_wise_matching else configs.invdes.epoch_per_proj,
                    admm_vars=admm_vars,
                )
                feasibleCE_lr_model, feasibleAcc_lr_model = test(
                    model, # post project test on low res model
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
                if not model.darcy:
                    worsenCE_r_lr_model.append((feasibleCE_lr_model - idealCE) / idealCE)
                    worsenCE_a_lr_model.append((feasibleCE_lr_model - idealCE))
                    worsenAcc_r_lr_model.append((idealAcc - feasibleAcc_lr_model) / idealAcc)
                    worsenAcc_a_lr_model.append((idealAcc - feasibleAcc_lr_model))
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
                    worsenCE_r.append((feasibleCE - idealCE) / idealCE)
                    worsenCE_a.append((feasibleCE - idealCE))
                    worsenAcc_r.append((idealAcc - feasibleAcc) / idealAcc)
                    worsenAcc_a.append((idealAcc - feasibleAcc))
                    if epoch % 10 == 0 or epoch == int(configs.run.n_epochs):
                        avg_worsenCE_r = np.mean(worsenCE_r)
                        avg_worsenAcc_r = np.mean(worsenAcc_r)
                        avg_worsenCE_a = np.mean(worsenCE_a)
                        avg_worsenAcc_a = np.mean(worsenAcc_a)
                        avg_worsenCE_r_lr_model = np.mean(worsenCE_r_lr_model)
                        avg_worsenAcc_r_lr_model = np.mean(worsenAcc_r_lr_model)
                        avg_worsenCE_a_lr_model = np.mean(worsenCE_a_lr_model)
                        avg_worsenAcc_a_lr_model = np.mean(worsenAcc_a_lr_model)
                        lg.info(f"[LR] rel avg_worsenCE: {avg_worsenCE_r_lr_model}, rel avg_worsenAcc: {avg_worsenAcc_r_lr_model}")
                        lg.info(f"[LR] abs avg_worsenCE: {avg_worsenCE_a_lr_model}, abs avg_worsenAcc: {avg_worsenAcc_a_lr_model}")
                        lg.info(f"[HR] rel avg_worsenCE: {avg_worsenCE_r}, rel avg_worsenAcc: {avg_worsenAcc_r}")
                        lg.info(f"[HR] abs avg_worsenCE: {avg_worsenCE_a}, abs avg_worsenAcc: {avg_worsenAcc_a}")
            if validation_loader is not None:
                validate( # post project valid on low res model
                    model,
                    validation_loader,
                    epoch,
                    criterion,
                    aux_criterions,
                    lossv,
                    accv,
                    device,
                    mixup_fn=test_mixup_fn,
                    fp16=grad_scaler._enabled,
                    stitched_response=stitched_response,
                    plot = False if not configs.run.project_GD else True,
                )
                if configs.run.project_GD:
                    validate( # post project valid on high res model
                        model_test,
                        validation_loader,
                        epoch,
                        criterion,
                        aux_criterions,
                        [],
                        [], # should not rely on the high res result to determine the best model
                        # lossv,
                        # accv,
                        device,
                        mixup_fn=test_mixup_fn,
                        fp16=grad_scaler._enabled,
                        stitched_response=stitched_response,
                        plot = True,
                    )
            test(
                model, # post project test on low res model
                test_loader,
                epoch,
                criterion,
                lossv if validation_loader is None else [],
                accv if validation_loader is None else [],
                device,
                mixup_fn=test_mixup_fn,
                fp16=grad_scaler._enabled,
                stitched_response=stitched_response,
                plot = True if not configs.run.project_GD else False,
            )
            if configs.run.project_GD:
                test( # post project test on high res
                    model_test,
                    test_loader,
                    epoch,
                    criterion,
                    [],
                    [],
                    device,
                    mixup_fn=test_mixup_fn,
                    fp16=grad_scaler._enabled,
                    stitched_response=stitched_response,
                    plot = True,
                )
            if configs.invdes.admm:
                # update the ADMM variables
                update_admm_dual_variable(
                    model=model,
                    stitched_response=stitched_response,
                    admm_vars=admm_vars,
                )
            
            if getattr(configs.run, "projection_once", False) and epoch == int(configs.run.n_epochs):
                # do the projection only once at the end of the training
                stitched_response, current_pillar_width = project_to_implementable_subspace(
                    model, 
                    model_test,
                    patch_metalens,
                    patch_metalens_list,
                    invdes_criterion,
                    current_pillar_width,
                    epoch,
                    "post_train", 
                    in_downsample_rate_scheduler,
                    out_downsample_rate_scheduler,
                    near2far_matrix,
                    ds_near2far_matrix,
                    device,
                    probe_full_wave=True,
                    invdes_lr=(5e-3, 5e-5),
                    invdes_sharp=(10, 256),
                    layer_wise_matching=configs.invdes.layer_wise_matching,
                    epoch_per_proj=50,
                    finetune_entire=configs.invdes.finetune_entire,
                    match_entire_epoch=50,
                    admm_vars=admm_vars,
                )
                test(
                    model_test,
                    test_loader,
                    epoch,
                    criterion,
                    lossv,
                    accv,
                    device,
                    mixup_fn=test_mixup_fn,
                    fp16=grad_scaler._enabled,
                    stitched_response=stitched_response,
                    plot = False,
                )
            elif getattr(configs.run, "conv_LPA", False) and epoch == int(configs.run.n_epochs):
                set_test_model_transfer_matrix(
                    model=model, 
                    model_test=model_test,
                    patched_metalens=patch_metalens,
                    device=device,
                )
                test(
                    model_test,
                    test_loader,
                    epoch,
                    criterion,
                    lossv,
                    accv,
                    device,
                    mixup_fn=test_mixup_fn,
                    fp16=grad_scaler._enabled,
                    stitched_response=stitched_response,
                    plot = False,
                )
            elif getattr(configs.run, "LPA", False) and epoch == int(configs.run.n_epochs):
                # need to extract the transfer matrix from the model
                # then use the look up table to get the pillar width
                # set it to patch_metalens and probe the transfer matrix and set to test model
                design_using_LPA(
                    model=model, 
                    model_test=model_test,
                    patched_metalens=patch_metalens,
                    device=device,
                )
                test(
                    model_test,
                    test_loader,
                    epoch,
                    criterion,
                    lossv,
                    accv,
                    device,
                    mixup_fn=test_mixup_fn,
                    fp16=grad_scaler._enabled,
                    stitched_response=stitched_response,
                    plot = False,
                )
            saver.save_model(
                getattr(model, "_orig_mod", model),  # remove compiled wrapper
                accv[-1] if not model.darcy else lossv[-1],
                epoch=epoch,
                path=checkpoint,
                save_model=False,
                print_msg=True,
            )
            test_model_saver.save_model(
                getattr(model_test, "_orig_mod", model_test),  # remove compiled wrapper
                accv[-1] if not model_test.darcy else lossv[-1],
                epoch=epoch,
                path=checkpoint.replace(".pt", "_test.pt"),
                save_model=False,
                print_msg=True,
            )
        wandb.finish()
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()