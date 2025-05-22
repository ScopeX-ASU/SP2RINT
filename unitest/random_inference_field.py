'''
in this script, we will plot the inference field of a hybrid model and show that why phase mask does not work

to do this, we will use the following steps:
0. simulate all the diffraction near2far matrix and save them to a h5 file
1. read a batch of iamges from data loader and unfold the images to feed into the model
2. find the one with appropriate values
3. 
'''
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from torch.cuda import amp
import csv
from torch.utils.data import DataLoader
from pyutils.general import ensure_dir, print_stat
from pyutils.config import configs
from pyutils.general import logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
import argparse
import datetime
import time
import random
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from thirdparty.MAPS_old.core.invdes.models import (
    MetaLensOptimization,
)
from thirdparty.MAPS_old.core.invdes.models.base_optimization import DefaultSimulationConfig
from thirdparty.MAPS_old.core.invdes.models.layers import MetaLens
from core import builder
from core.datasets.mixup import MixupAll
from core.models.patch_metalens import PatchMetalens
from core.utils import (
    get_parameter_group, 
    register_hidden_hooks, 
    probe_near2far_matrix, 
    draw_diffraction_region,
    CosSimLoss,
    reset_optimizer_and_scheduler,
    get_mid_weight,
    DeterministicCtx,
    insert_zeros_after_every_N_except_last,
)
sys.path.pop()

def probe_full_tm(
    device,
    patched_metalens,
    full_wave_down_sample_rate = 1,
    num_atom=None,
    norm_tm=True,
):
    # time_start = time.time()
    number_atoms = num_atom
    sources = torch.eye(number_atoms * round(15 // full_wave_down_sample_rate), device=device)

    sim_key = list(patched_metalens.total_opt.objective.sims.keys())
    assert len(sim_key) == 1, f"there should be only one sim key, but we got {sim_key}"
    if hasattr(patched_metalens.total_opt.objective.sims[sim_key[0]].solver, "clear_solver_cache"):
        patched_metalens.total_opt.objective.sims[sim_key[0]].solver.clear_solver_cache()
        patched_metalens.total_opt.objective.sims[sim_key[0]].solver.set_cache_mode(True)
    # we first need to run the normalizer
    if norm_tm and (patched_metalens.total_normalizer_list is None or len(patched_metalens.total_normalizer_list) < number_atoms * round(15 // full_wave_down_sample_rate)):
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
            weight = patched_metalens.get_design_variables(
                -0.05 * torch.ones_like(patched_metalens.get_pillar_width()),
            )
            _ = patched_metalens.total_opt(
                sharpness=256, 
                weight=weight,
                custom_source=custom_source
            )

            source_field = patched_metalens.total_opt.objective.response[('in_slice_1', 'in_slice_1', 0.85, "Hz1", 300)]["fz"].squeeze()
            total_normalizer_list.append(source_field[boolean_source_mask].mean())
            if idx == number_atoms * round(15 // full_wave_down_sample_rate) - 1:
                patched_metalens.set_total_normalizer_list(total_normalizer_list)
    # now we already have the normalizer, we can run the full wave response
    if hasattr(patched_metalens.total_opt.objective.sims[sim_key[0]].solver, "clear_solver_cache"):
        patched_metalens.total_opt.objective.sims[sim_key[0]].solver.clear_solver_cache()
        patched_metalens.total_opt.objective.sims[sim_key[0]].solver.set_cache_mode(False)
        
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
            _ = patched_metalens.total_opt(
                sharpness=256, 
                weight=patched_metalens.get_design_variables(
                    patched_metalens.get_pillar_width(),
                ),
                custom_source=custom_source
            )
            response = patched_metalens.total_opt.objective.response[('in_slice_1', 'nearfield_1', 0.85, "Hz1", 300)]["fz"].squeeze()
            response = response[full_wave_down_sample_rate // 2 :: full_wave_down_sample_rate]
            assert len(response) == number_atoms * round(15 // full_wave_down_sample_rate), f"{len(response)}!= {number_atoms * round(15 // full_wave_down_sample_rate)}"
            full_wave_response[idx] = response
        full_wave_response = full_wave_response.transpose(0, 1)
        if norm_tm: 
            normalizer = torch.stack(patched_metalens.total_normalizer_list, dim=0).to(device)
            normalizer = normalizer.unsqueeze(1)
            full_wave_response = full_wave_response / normalizer

    if hasattr(patched_metalens.total_opt.objective.sims[sim_key[0]].solver, "clear_solver_cache"):
        patched_metalens.total_opt.objective.sims[sim_key[0]].solver.clear_solver_cache()
        patched_metalens.total_opt.objective.sims[sim_key[0]].solver.set_cache_mode(False)

    # time_end = time.time()
    # print(f"this is the time for probing the full wave response: {time_end - time_start}", flush=True)
    return full_wave_response


def print_unique_counts(tensor: torch.Tensor):
    """
    Given a 2D tensor of shape (batch_size, L), print the number of unique values
    in each row.
    """
    assert tensor.dim() == 2, "Input tensor must be 2D (batch_size, L)"
    
    for i, row in enumerate(tensor):
        unique_vals = torch.unique(row)
        print(f"Sample {i}: {len(unique_vals)} unique values")

def calculate_diffraction(near2far_matrices, source, device, mode = "near2far", diffract_opt=None, num_atom=32):
    if mode == "near2far":
        total_field = draw_diffraction_region(
            total_opt=diffract_opt, 
            wl=0.85, # in um
            source=source.squeeze(),
            device=device,
        )
        print("this is the total field shape: ", total_field.shape, flush=True)
        assert near2far_matrices.ndim == 2, "near2far matrices should be 2D"
        field = torch.matmul(source, near2far_matrices.T.to(source.dtype))
        field = field.squeeze()

    elif mode == "simulation":
        assert diffract_opt is not None, "diffract_opt should not be None when mode is simulation"
        ls_knots = -0.05 * torch.ones(2 * num_atom + 1, device=device)
        src_zero_padding = torch.zeros(int(0.5 * 50), device=device).to(source.dtype)
        source = torch.cat(
            [
                src_zero_padding,
                source,
                src_zero_padding,
            ],
            dim=-1,
        )
        with torch.no_grad():
            _ = diffract_opt.forward(
                sharpness=256,
                weight={"design_region_0": ls_knots.unsqueeze(0)},
                custom_source=dict(
                    source=source,
                    slice_name="nearfield_1",
                    mode="Hz1",
                    wl=0.85,
                    direction="x+",
                ),
            )
        total_field = diffract_opt.objective.solutions[("nearfield_1", 0.85, "Hz1", 300)]["Hz"]
        total_field = total_field[
            -round(res*(0.5 + diffract_opt.device.farfield_dxs[0][0] - 0.3)):-round(res*0.5),
            round(res*0.5):-round(res*0.5),
        ].T
        field = diffract_opt.objective.response[('nearfield_1', 'nearfield_2', 0.85, "Hz1", 300)]["fz"].squeeze()
        print("this is the shape of the field: ", field.shape, flush=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return total_field, field
    
def get_near2far_matrices(path, device, num_atom=32, distance=4):
    res = 50
    wl = 0.85
    atom_period = 0.3
    plot_root = "./unitest/inference_field/dummy_plot"
    assert isinstance(distance, int), "distance should be an integer"
    ensure_dir(plot_root)
    if os.path.exists(path):
        with h5py.File(path, "r") as f:
            near2far_matrices = f["near2far_matrices"][:]
            near2far_matrices = torch.from_numpy(near2far_matrices).to(device)
            assert near2far_matrices.ndim == 2, "near2far matrices should be 2D"
        return near2far_matrices
    else:
        sim_cfg = DefaultSimulationConfig()
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
                plot_root=plot_root,
            )
        )
        metalens = MetaLens(
            material_bg="Air",
            material_r = "Si",
            material_sub="SiO2",
            sim_cfg=sim_cfg,
            aperture=atom_period * num_atom,
            port_len=(1, 1),
            port_width=(atom_period * num_atom, atom_period),
            substrate_depth=0,
            ridge_height_max=0.75,
            nearfield_dx=0.3,
            nearfield_size=atom_period * num_atom,
            farfield_dxs=((distance, distance + 2/res),),
            farfield_sizes=(atom_period * num_atom,),
            device=device,
            design_var_type="width",
            atom_period=atom_period,
            atom_width=None,
        )
        hr_metalens = metalens.copy(resolution=200)
        opt = MetaLensOptimization(
            device=metalens,
            hr_device=hr_metalens,
            sim_cfg=sim_cfg,
            operation_device=device,
            design_region_param_cfgs={},
        )
        with torch.inference_mode():
            near2far_matrix = probe_near2far_matrix(
                opt, 
                wl, # in um
                device,
                normalize=False,
            )

        near2far_matrices = near2far_matrix.to(torch.complex128)
        
        with h5py.File(path, "w") as f:
            f.create_dataset("near2far_matrices", data=near2far_matrices.cpu().numpy())

        print("near2far matrices saved to", path)
        return near2far_matrices

def get_metasurface_tm(path, device, num_atom=32, LUT=None, num_surfaces=10, norm_tm=True, patch_size=17):
    res = 50
    wl = 0.85
    atom_period = 0.3
    plot_root = "./unitest/random_inference_field/dummy_plot"
    ensure_dir(plot_root)
    if os.path.exists(path):
        with h5py.File(path, "r") as f:
            metasurface_tm_15 = torch.tensor(f["metasurface_tm_15"][:], device=device)
            metasurface_tm_1 = torch.tensor(f["metasurface_tm_1"][:], device=device)
            full_surface_tm = torch.tensor(f["full_surface_tm"][:], device=device)
            ls_knots = torch.tensor(f["metasurface_ls_knots"][:], device=device)
        return metasurface_tm_15, metasurface_tm_1, full_surface_tm, ls_knots
    else:
        # need to simulate the metasurface tm
        surface_calculator_15 = PatchMetalens(
            atom_period=0.3,
            patch_size=patch_size, # hard code to be 17
            num_atom=num_atom,
            probing_region_size=patch_size, # hard code to be 17
            target_phase_response=None,
            LUT=LUT,
            device=device,
            target_dx=0.3,
            plot_root=plot_root,
            downsample_mode="both",
            downsample_method="avg",
            dz=4.0,
            param_method="level_set",
            tm_norm="field" if norm_tm else "none",
            field_norm_condition="wo_lens",
            design_var_type="width", # width or height
            atom_width=0.12,
        )
        surface_calculator_1 = PatchMetalens(
            atom_period=0.3,
            patch_size=patch_size, # hard code to be 17
            num_atom=num_atom,
            probing_region_size=patch_size, # hard code to be 17
            target_phase_response=None,
            LUT=LUT,
            device=device,
            target_dx=0.3,
            plot_root=plot_root,
            downsample_mode="both",
            downsample_method="avg",
            dz=4.0,
            param_method="level_set",
            tm_norm="field" if norm_tm else "none",
            field_norm_condition="wo_lens",
            design_var_type="width", # width or height
            atom_width=0.12,
        )
        patched_surface_tm_15_list = []
        patched_surface_tm_1_list = []
        full_surface_tm_list = []
        with DeterministicCtx(seed=41):
            ls_knots = 0.27 * torch.rand(num_surfaces, num_atom, device=device) + 0.01 # this is the width of the atom
            ls_knots = get_mid_weight(0.05, ls_knots) # this is actually the height of the level set height
        for idx in tqdm(range(num_surfaces)):
            surface_calculator_15.disable_solver_cache()
            surface_calculator_15.direct_set_pillar_width(ls_knots[idx])
            surface_calculator_1.disable_solver_cache()
            surface_calculator_1.direct_set_pillar_width(ls_knots[idx])
            with torch.no_grad():
                time_start = time.time()
                patched_surface_tm_15 = surface_calculator_15.forward(
                    sharpness=256, 
                    in_down_sample_rate=15,
                    out_down_sample_rate=1,
                )
                time_end = time.time()
                print(f"this is the time for probing the TM with 15 down sample rate: {time_end - time_start}", flush=True)
                patched_surface_tm_1 = surface_calculator_1.forward(
                    sharpness=256, 
                    in_down_sample_rate=1,
                    out_down_sample_rate=1,
                )
                patched_surface_tm_15_list.append(patched_surface_tm_15)
                patched_surface_tm_1_list.append(patched_surface_tm_1)
                time_start = time.time()
                full_surface_tm = probe_full_tm(
                    device=device,
                    patched_metalens=surface_calculator_15,
                    full_wave_down_sample_rate = 1,
                    num_atom=num_atom,
                )
                time_end = time.time()
                print(f"this is the time for probing the full wave response: {time_end - time_start}", flush=True)
                quit()
                full_surface_tm_list.append(full_surface_tm)
        with h5py.File(path, "w") as f:
            f.create_dataset("metasurface_tm_15", data=torch.stack(patched_surface_tm_15_list, dim=0).cpu().numpy())
            f.create_dataset("metasurface_tm_1", data=torch.stack(patched_surface_tm_1_list, dim=0).cpu().numpy())
            f.create_dataset("full_surface_tm", data=torch.stack(full_surface_tm_list, dim=0).cpu().numpy())
            f.create_dataset("metasurface_ls_knots", data=ls_knots.cpu().numpy())
        return torch.stack(patched_surface_tm_15_list, dim=0), torch.stack(patched_surface_tm_1_list, dim=0), torch.stack(full_surface_tm_list, dim=0), ls_knots



if __name__ == "__main__":
    exp_dir = "./unitest/random_inference_field"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensure_dir(exp_dir)
    # num_atom = 32
    num_atom = 32
    num_surfaces = 10
    distance = 4 #* round(num_atom / 32)
    calculate_diffraction_method = "near2far"
    norm_tm = False
    # patch_size = 53
    patch_size = 17
    near2far_matrices_path = os.path.join(exp_dir, f"near2far_matrices_{num_atom}_distance_{int(distance)}.h5")
    if calculate_diffraction_method == "near2far":
        near2far_matrices = get_near2far_matrices(near2far_matrices_path, device, num_atom=num_atom, distance=distance)
        
    elif calculate_diffraction_method == "simulation":
        near2far_matrices = None
    else:
        raise ValueError(f"Invalid diffraction method: {calculate_diffraction_method}")
    # now we have to simulate the transfer matrix of random metasurface
    csv_file = f"core/metaatom_response_fsdx-0.3.csv"
    LUT = {}
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            # if float(row[0]) > 0.14:
            #     break
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
    if norm_tm:
        metasurface_tm_path = os.path.join(exp_dir, f"metasurface_tm_{num_atom}_ps_{patch_size}.h5")
    else:
        metasurface_tm_path = os.path.join(exp_dir, f"metasurface_tm_{num_atom}_no_norm_ps_{patch_size}.h5")
    metasurface_tm_15, metasurface_tm_1, full_surface_tm, metasurface_ls_knots = get_metasurface_tm(metasurface_tm_path, device, num_atom=num_atom, LUT=LUT, num_surfaces=num_surfaces, norm_tm=norm_tm, patch_size=patch_size)
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # im0=axs[0].imshow(metasurface_tm_1[0].abs().cpu().detach().numpy())
    # fig.colorbar(im0, ax=axs[0])
    # im1=axs[1].imshow(full_surface_tm[0].abs().cpu().detach().numpy())
    # fig.colorbar(im1, ax=axs[1])
    # im2=axs[2].imshow((full_surface_tm.abs() - metasurface_tm_1.abs())[0].cpu().detach().numpy())
    # fig.colorbar(im2, ax=axs[2])
    # plt.savefig(os.path.join(exp_dir, f"metasurface_tm_{num_atom}.png"))
    # plt.close()
    quit()
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

    if bool(configs.run.deterministic):
        set_torch_deterministic()

    train_loader, validation_loader, test_loader = builder.make_dataloader(
        splits=["train", "valid", "test"]
    )

    # now we have successfully loaded the model
    batch_size = 10
    wl = 0.85
    atom_period = 0.3
    res = 50
    num_layers = 6
    num_diff_sys = 20
    # random pick batch_size samples from the train_loader
    with DeterministicCtx(seed=41):
        samples = [train_loader.dataset[i] for i in random.sample(range(len(train_loader.dataset)), batch_size)]
    imgs = [sample[0] for sample in samples]
    imgs = torch.stack(imgs, dim=0).to(device)

    x = imgs
    total_sys_lens = round(num_atom * res * atom_period)
    input_wg_width_px = round(configs.model.input_wg_width * res)
    input_wg_interval_px = round(configs.model.input_wg_interval * res) * (num_atom // 32)
    # [B, 1, 28, 28]
    bs, C, H, W = x.shape
    p = 3
    # Ensure image size is sufficient
    assert H >= p and W >= p, "Image smaller than patch size"
    # Unfold: turn image into sliding patches
    patches = x.unfold(2, p, 1).unfold(3, p, 1)  # (bs, C, H-p+1, W-p+1, p, p)
    out_H = patches.shape[2]
    out_W = patches.shape[3]
    patches = patches.contiguous().view(bs, C, out_H * out_W, p * p)  # (bs, C, N, p*p)
    patches = patches.reshape(bs * C * out_H * out_W, p * p).unsqueeze(1)  # (bs*C*out_H*out_W, 1, p*p)
    source_mask = torch.ones_like(patches, dtype=torch.bool, device=patches.device)
    x = insert_zeros_after_every_N_except_last(patches, input_wg_width_px, input_wg_interval_px)
    source_mask = insert_zeros_after_every_N_except_last(source_mask, input_wg_width_px, input_wg_interval_px)
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
    print("this is the shape of the x: ", x.shape, flush=True)

    source_mask = x[..., 1].to(torch.bool)
    x = x[..., 0]
    bs, inC, L = x.shape
    
    eps = 1e-8
    # meanval = x.square().mean(dim=[-1], keepdim=True).add(eps).sqrt()  # shape [bs, inC, 1]
    # x = x / meanval  # [bs, inC, L]
    
    # print(x.shape)
    
    x = torch.exp(1j * x)
    x[~source_mask] = 0
    print("this is the shape of the x: ", x.shape, flush=True) # bs, 1, L
    x = x.squeeze(1)
    # print_unique_counts(torch.angle(x))
    # plt.figure()
    # plt.plot(torch.angle(x[6092].cpu().detach().numpy()))
    # plt.plot(torch.abs(x[6092].cpu().detach().numpy()))
    # plt.savefig(os.path.join(exp_dir, "angle_abs.png"))
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].plot(torch.angle(x[6092]).cpu().detach().numpy())
    # ax[0].set_title("Angle")
    # ax[1].plot(torch.abs(x[6092]).cpu().detach().numpy())
    # ax[1].set_title("Abs")
    # plt.savefig(os.path.join(exp_dir, "angle_abs.png"))
    # quit()
    plot_root = "./unitest/random_inference_field"
    ensure_dir(plot_root)

    sim_cfg = DefaultSimulationConfig()
    sim_cfg.update(
        dict(
            solver="ceviche_torch",
            numerical_solver="solve_direct",
            use_autodiff=False,
            neural_solver=None,
            border_width=[0, 0, 0.5, 0.5],
            PML=[0.5, 0.5],
            resolution=res,
            wl_cen=wl,
            plot_root=plot_root,
        )
    )
    metalens = MetaLens(
        material_bg="Air",
        material_r = "Si",
        material_sub="SiO2",
        sim_cfg=sim_cfg,
        aperture=atom_period * num_atom,
        port_len=(1, distance + 0.5),
        port_width=(atom_period * num_atom, atom_period),
        substrate_depth=0,
        ridge_height_max=0.75,
        nearfield_dx=0.3,
        nearfield_size=atom_period * num_atom,
        # farfield_dxs=((distance, distance + 2/res),),
        farfield_dxs=((0.3, distance),),
        farfield_sizes=(atom_period * num_atom,),
        device=device,
        design_var_type="width",
        atom_period=atom_period,
        atom_width=None,
    )
    hr_metalens = metalens.copy(resolution=200)
    obj_cfgs=dict(
                near_field_response_record=dict(
                    weight=0,
                    #### objective is evaluated at this port
                    in_slice_name="in_slice_1",
                    out_slice_name="nearfield_1",
                    #### objective is evaluated at all points by sweeping the wavelength and modes
                    temp=[300],
                    wl=[0.85],
                    in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                    out_modes=("Hz1",),
                    type="response_record",
                    direction="x+",
                ),
                far_field_response_record=dict(
                    weight=0,
                    #### objective is evaluated at this port
                    in_slice_name="in_slice_1",
                    out_slice_name="nearfield_2",
                    #### objective is evaluated at all points by sweeping the wavelength and modes
                    temp=[300],
                    wl=[0.85],
                    in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                    out_modes=("Hz1",),
                    type="response_record",
                    direction="x+",
                ),
                source_field_response_record=dict(
                    weight=0,
                    #### objective is evaluated at this port
                    in_slice_name="in_slice_1",
                    out_slice_name="in_slice_1",
                    #### objective is evaluated at all points by sweeping the wavelength and modes
                    temp=[300],
                    wl=[0.85],
                    in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                    out_modes=("Hz1",),
                    type="response_record",
                    direction="x+",
                ),
            )
    diffract_obj_cfgs=dict(
                near_field_response_record=dict(
                    weight=0,
                    #### objective is evaluated at this port
                    in_slice_name="nearfield_1",
                    out_slice_name="nearfield_1",
                    #### objective is evaluated at all points by sweeping the wavelength and modes
                    temp=[300],
                    wl=[0.85],
                    in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                    out_modes=("Hz1",),
                    type="response_record",
                    direction="x+",
                ),
                source_field_response_record=dict(
                    weight=0,
                    #### objective is evaluated at this port
                    in_slice_name="nearfield_1",
                    out_slice_name="nearfield_1",
                    #### objective is evaluated at all points by sweeping the wavelength and modes
                    temp=[300],
                    wl=[0.85],
                    in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                    out_modes=("Hz1",),
                    type="response_record",
                    direction="x+",
                ),
            )
    opt = MetaLensOptimization(
        device=metalens,
        hr_device=hr_metalens,
        sim_cfg=sim_cfg,
        operation_device=device,
        design_region_param_cfgs={},
        obj_cfgs=obj_cfgs,
    )
    sim_key = list(opt.objective.sims.keys())
    assert len(sim_key) == 1, f"there should be only one sim key, but we got {sim_key}"
    opt.objective.sims[sim_key[0]].solver.clear_solver_cache()
    opt.objective.sims[sim_key[0]].solver.set_cache_mode(False)

    diffract_metalens = MetaLens(
        material_bg="Air",
        material_r = "Si",
        material_sub="SiO2",
        sim_cfg=sim_cfg,
        aperture=atom_period * num_atom,
        port_len=(1, distance + 0.5),
        port_width=(atom_period * num_atom, atom_period),
        substrate_depth=0,
        ridge_height_max=0.75,
        nearfield_dx=0.3,
        nearfield_size=atom_period * num_atom,
        farfield_dxs=((0.3, distance),),
        farfield_sizes=(atom_period * num_atom,),
        device=device,
        design_var_type="width",
        atom_period=atom_period,
        atom_width=None,
    )
    diffract_hr_metalens = metalens.copy(resolution=200)
    diffract_opt = MetaLensOptimization(
        device=diffract_metalens,
        hr_device=diffract_hr_metalens,
        sim_cfg=sim_cfg,
        operation_device=device,
        design_region_param_cfgs={},
        obj_cfgs=diffract_obj_cfgs,
    )
    diffract_opt.objective.sims[sim_key[0]].solver.clear_solver_cache()
    diffract_opt.objective.sims[sim_key[0]].solver.set_cache_mode(False)
    
    region_1, x1 = calculate_diffraction(near2far_matrices, x[6092], device, mode=calculate_diffraction_method, diffract_opt=diffract_opt, num_atom=num_atom)
    # print("this is the dtype of x1: ", x1.dtype, flush=True)
    # quit()
    print("this is the shape of the region_1: ", region_1.shape, flush=True)
    print("this is the shape of the x1: ", x1.shape, flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(torch.abs(region_1).cpu().detach().numpy())
    ax[0].set_title("Abs")
    ax[1].imshow(torch.angle(region_1).cpu().detach().numpy())
    ax[1].set_title("Angle")
    plt.savefig(os.path.join(exp_dir, f"region_1_{calculate_diffraction_method}_num_atom_{num_atom}.png"), dpi = 300)
    N_L2norm_ours_list = []
    N_L2norm_LPA_list = []
    for diff_sys_idx in tqdm(range(num_diff_sys)):
        x_gt = x1.clone()
        x_ours = x1.clone()
        x_LPA = x1.clone() 
        receiver_gt_list = []
        receiver_ours_list = []
        receiver_LPA_list = []
        diff_sys_plot_dir = os.path.join(exp_dir, f"diff_sys_{diff_sys_idx}_num_atom_{num_atom}_layer_{num_layers}")
        ensure_dir(diff_sys_plot_dir)
        with DeterministicCtx(seed=41 + diff_sys_idx):
            # random generate a non-repeating list of indices from [0, bs) of length num_layers
            surface_pool = list(range(len(metasurface_ls_knots)))
            assert len(surface_pool) >= num_layers, f"the number of surfaces is less than the number of layers, {len(surface_pool)} < {num_layers}"
            # random pick num_layers indices from surface_pool, non-repeating
            surface_idx = random.sample(surface_pool, num_layers)
        for i in range(num_layers):
            ls_knots_to_be_set = -0.05 * torch.ones(2 * num_atom + 1, device=device)
            ls_knots_to_be_set[1::2] = metasurface_ls_knots[surface_idx[i]]
            weight = {
                "design_region_0": ls_knots_to_be_set.unsqueeze(0),
            }
            source_padding = torch.zeros(int(0.5 * res), device=device).to(x1.dtype)
            x_gt = torch.cat(
                [
                    source_padding,
                    x_gt,
                    source_padding,
                ],
                dim=-1,
            )
            x_gt = x_gt / torch.norm(x_gt, p=2) # normalize the input signal
            with torch.no_grad():
                _ = opt.forward(
                    sharpness=256,
                    weight=weight,
                    custom_source=dict(
                        source=x_gt,
                        slice_name="in_slice_1",
                        mode="Hz1",
                        wl=0.85,
                        direction="x+",
                    ),
                )
            x_gt = opt.objective.response[('in_slice_1', 'nearfield_2', 0.85, "Hz1", 300)]["fz"].squeeze()
            receiver_gt_list.append(
                x_gt / torch.norm(x_gt, p=2) # record the normalized receiver field
            )
            opt.plot(
                plot_filename=f"region_{i+2}_gt.png",
                eps_map=opt._eps_map,
                obj=None,
                field_key=("in_slice_1", 0.85, "Hz1", 300),
                field_component="Hz",
                in_slice_name="in_slice_1",
                exclude_slice_names=[],
                plot_dir=diff_sys_plot_dir,
            )
            Ez_gt = opt.objective.solutions[("in_slice_1", 0.85, "Hz1", 300)]["Hz"]
            Ez_gt = Ez_gt[
                -round(res*(0.5 + distance - 0.3)):-round(res*0.5),
                round(res*0.5):-round(res*0.5),
            ]
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(torch.abs(Ez_gt.T).cpu().detach().numpy())
            ax[0].set_title("Abs")
            ax[1].imshow((Ez_gt.T.real).cpu().detach().numpy())
            ax[1].set_title("real")
            plt.savefig(os.path.join(diff_sys_plot_dir, f"region_{i+2}.png"), dpi = 300)
            plt.close()
            # here we calculate the diffraction field using our model
            tm = metasurface_tm_1[surface_idx[i]]
            # tm = full_surface_tm[surface_idx[i]]
            x_ours = x_ours / torch.norm(x_ours, p=2) # normalize the input signal
            x_near_field = torch.matmul(x_ours.to(tm.dtype), tm.T)
            diffraction_region_ours, x_ours = calculate_diffraction(near2far_matrices, x_near_field, device, mode=calculate_diffraction_method, diffract_opt=diffract_opt, num_atom=num_atom)
            receiver_ours_list.append(
                x_ours / torch.norm(x_ours, p=2) # record the normalized receiver field
            )
            diffraction_region_ours = diffraction_region_ours / torch.max(diffraction_region_ours.abs())
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            im0 = ax[0].imshow(
                torch.abs(diffraction_region_ours).cpu().detach().numpy(), 
                vmax=1,
            )
            ax[0].set_title("Abs")
            # plt.colorbar(im0, ax=ax[0])
            ax[1].imshow(diffraction_region_ours.real.cpu().detach().numpy())
            ax[1].set_title("real")
            plt.savefig(os.path.join(diff_sys_plot_dir, f"region_{i+2}_ours_{calculate_diffraction_method}.png"), dpi = 300)
            plt.close()

            # here we calculate the diffraction field using LPA
            pillar_width = metasurface_ls_knots[surface_idx[i]]
            pillar_width = (0.3 * pillar_width) / (0.05 + pillar_width)
            # for width in pillar_width:
            #     if width < 0.14:
            #         width = 0.14
            pillar_width = torch.clamp(pillar_width, min=0.01, max=0.28)
            LPA_tm = torch.zeros((len(pillar_width), len(pillar_width)), device=device, dtype=torch.complex64)
            for j in range(len(pillar_width)):
                phase_shift = torch.tensor(LUT[round(pillar_width[j].item(), 3)], device=device)
                # print("this is the phase shift: ", phase_shift, flush=True)
                # print("this is the type of phase shift: ", type(phase_shift), flush=True)
                LPA_tm[j, j] = 1 * torch.exp(1j * phase_shift)
            LPA_tm_real = F.interpolate(LPA_tm.real.unsqueeze(0).unsqueeze(0), size=(total_sys_lens, total_sys_lens), mode="bilinear", align_corners=False).squeeze()
            LPA_tm_imag = F.interpolate(LPA_tm.imag.unsqueeze(0).unsqueeze(0), size=(total_sys_lens, total_sys_lens), mode="bilinear", align_corners=False).squeeze()
            LPA_tm = LPA_tm_real + 1j * LPA_tm_imag
            x_LPA = x_LPA / torch.norm(x_LPA, p=2) # normalize the input signal
            x_near_field_LPA = torch.matmul(x_LPA.to(LPA_tm.dtype), LPA_tm.T)
            diffraction_region_LPA, x_LPA = calculate_diffraction(near2far_matrices, x_near_field_LPA, device, mode=calculate_diffraction_method, diffract_opt=diffract_opt, num_atom=num_atom)
            receiver_LPA_list.append(
                x_LPA / torch.norm(x_LPA, p=2) # record the normalized receiver field
            )
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(torch.abs(diffraction_region_LPA).cpu().detach().numpy())
            ax[0].set_title("Abs")
            ax[1].imshow(diffraction_region_LPA.real.cpu().detach().numpy())
            ax[1].set_title("real")
            plt.savefig(os.path.join(diff_sys_plot_dir, f"region_{i+2}_LPA_{calculate_diffraction_method}.png"), dpi = 300)
            plt.close()

        receiver_gt_amp = torch.stack(receiver_gt_list, dim=0).abs() # 6, 480
        receiver_gt_phase = torch.stack(receiver_gt_list, dim=0).angle()
        receiver_gt_phase = receiver_gt_phase - receiver_gt_phase[:, receiver_gt_amp.shape[1]//2].unsqueeze(1)
        receiver_gt = receiver_gt_amp * torch.exp(1j * receiver_gt_phase)
        receiver_ours_amp = torch.stack(receiver_ours_list, dim=0).abs()
        receiver_ours_phase = torch.stack(receiver_ours_list, dim=0).angle()
        receiver_ours_phase = receiver_ours_phase - receiver_ours_phase[:, receiver_ours_amp.shape[1]//2].unsqueeze(1)
        receiver_ours = receiver_ours_amp * torch.exp(1j * receiver_ours_phase)
        receiver_LPA_amp = torch.stack(receiver_LPA_list, dim=0).abs()
        receiver_LPA_phase = torch.stack(receiver_LPA_list, dim=0).angle()
        receiver_LPA_phase = receiver_LPA_phase - receiver_LPA_phase[:, receiver_LPA_amp.shape[1]//2].unsqueeze(1)
        receiver_LPA = receiver_LPA_amp * torch.exp(1j * receiver_LPA_phase)
        error_ours = receiver_gt_amp - receiver_ours_amp
        error_LPA = receiver_gt_amp - receiver_LPA_amp
        N_L2norm_ours = torch.norm(error_ours, p=2, dim=-1) / torch.norm(receiver_gt_amp, p=2, dim=-1) # 6
        N_L2norm_LPA = torch.norm(error_LPA, p=2, dim=-1) / torch.norm(receiver_gt_amp, p=2, dim=-1)
        N_L2norm_ours_list.append(N_L2norm_ours)
        N_L2norm_LPA_list.append(N_L2norm_LPA)
        
        if diff_sys_idx == 0:
            for i in range(len(receiver_gt)):
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].plot(receiver_gt[i].abs().cpu().detach().numpy(), label="receiver_gt")
                ax[0].plot(receiver_ours[i].abs().cpu().detach().numpy(), label="receiver_ours")
                ax[0].plot(receiver_LPA[i].abs().cpu().detach().numpy(), label="receiver_LPA")
                ax[0].legend()
                ax[1].plot(receiver_gt[i].angle().cpu().detach().numpy(), label="receiver_gt")
                ax[1].plot(receiver_ours[i].angle().cpu().detach().numpy(), label="receiver_ours")
                ax[1].plot(receiver_LPA[i].angle().cpu().detach().numpy(), label="receiver_LPA")
                ax[1].legend()
                plt.savefig(os.path.join(diff_sys_plot_dir, f"receiver_{i}.png"), dpi = 300)
                plt.close()
            for i in range(receiver_gt.shape[0]):
                with open(os.path.join(diff_sys_plot_dir, f"field_receiver_{i}.csv"), "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["idx", "gt_amp", "ours_amp", "LPA_amp"])
                    for j in range(receiver_gt.shape[1]):
                        writer.writerow([j, receiver_gt[i, j].abs().cpu().detach().numpy(), receiver_ours[i, j].abs().cpu().detach().numpy(), receiver_LPA[i, j].abs().cpu().detach().numpy()])

    N_L2norm_ours_list = torch.stack(N_L2norm_ours_list, dim=0) # 20, 6
    N_L2norm_LPA_list = torch.stack(N_L2norm_LPA_list, dim=0)
    N_L2norm_ours_mean = torch.mean(N_L2norm_ours_list, dim=0) # 6
    N_L2norm_LPA_mean = torch.mean(N_L2norm_LPA_list, dim=0) # 6
    N_L2norm_ours_std = torch.std(N_L2norm_ours_list, dim=0) # 6
    N_L2norm_LPA_std = torch.std(N_L2norm_LPA_list, dim=0) # 6
    with open(os.path.join(exp_dir, f"N_L2norm_ours_mean_LPA_mean_std_num_atom_{num_atom}_num_layers_{num_layers}_diff_method_{calculate_diffraction_method}_norm_tm_{int(norm_tm)}.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "N_L2norm_ours_mean", "N_L2norm_LPA_mean", "N_L2norm_ours_std", "N_L2norm_LPA_std"])
        for i in range(len(N_L2norm_ours_mean)):
            writer.writerow([i, N_L2norm_ours_mean[i].item(), N_L2norm_LPA_mean[i].item(), N_L2norm_ours_std[i].item(), N_L2norm_LPA_std[i].item()])
    quit()