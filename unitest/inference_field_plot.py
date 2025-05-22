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
from pyutils.general import ensure_dir
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
    CosSimLoss,
    reset_optimizer_and_scheduler,
    get_mid_weight,
    DeterministicCtx,
    insert_zeros_after_every_N_except_last,
)
sys.path.pop()

def print_unique_counts(tensor: torch.Tensor):
    """
    Given a 2D tensor of shape (batch_size, L), print the number of unique values
    in each row.
    """
    assert tensor.dim() == 2, "Input tensor must be 2D (batch_size, L)"
    
    for i, row in enumerate(tensor):
        unique_vals = torch.unique(row)
        print(f"Sample {i}: {len(unique_vals)} unique values")

def calculate_diffraction(near2far_matrices, source, device, mode = "near2far", diffract_opt=None):
    if mode == "near2far":
        total_field = [source.unsqueeze(1)]

        for i in range(near2far_matrices.shape[0]):
            field = torch.matmul(source, near2far_matrices[i].T)
            total_field.append(field.unsqueeze(1))
        total_field = torch.cat(total_field, dim=1)

    elif mode == "simulation":
        assert diffract_opt is not None, "diffract_opt should not be None when mode is simulation"
        ls_knots = -0.05 * torch.ones(2 * 32 + 1, device=device)
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
            -round(res*(0.5 + 3.7)):-round(res*0.5),
            round(res*0.5):-round(res*0.5),
        ].T
        field = diffract_opt.objective.response[('nearfield_1', 'nearfield_2', 0.85, "Hz1", 300)]["fz"].squeeze()
        print("this is the shape of the field: ", field.shape, flush=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return total_field, field
    

def get_near2far_matrices(path, device):
    res = 50
    wl = 0.85
    atom_period = 0.3
    num_atom = 32
    plot_root = "./unitest/inference_field/dummy_plot"
    ensure_dir(plot_root)
    if os.path.exists(path):
        with h5py.File(path, "r") as f:
            near2far_matrices = f["near2far_matrices"][:]
        return near2far_matrices
    else:
        # need to simulate the near2far matrices
        near_distance_px = int(0.3 * res)
        far_distance_px = int(4 * res)
        # far_distance_px = near_distance_px + 1
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
        near2far_matrices = []
        for distance in tqdm(range(near_distance_px + 1, far_distance_px + 1)):
            distance = distance / res
            distance = round(distance, 2)
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
            near2far_matrix = probe_near2far_matrix(
                opt, 
                wl, # in um
                device
            )

            plt.figure()
            plt.imshow(
                near2far_matrix.abs().cpu().detach().numpy(),
            )
            plt.colorbar()
            plt.title(f"near2far matrix at {distance} um")
            plt.savefig(os.path.join(plot_root, f"near2far_matrix_{distance}.png"))
            plt.close()
            near2far_matrices.append(near2far_matrix)
        near2far_matrices = torch.stack(near2far_matrices, dim=0)
        with h5py.File(path, "w") as f:
            f.create_dataset("near2far_matrices", data=near2far_matrices.cpu().numpy())

        print("near2far matrices saved to", path)
        return near2far_matrices

if __name__ == "__main__":
    exp_dir = "./unitest/inference_field"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensure_dir(exp_dir)
    near2far_matrices_path = os.path.join(exp_dir, "near2far_matrices.h5")
    near2far_matrices = get_near2far_matrices(near2far_matrices_path, device)
    near2far_matrices = torch.from_numpy(near2far_matrices).to(device)
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
    # -----------------------------------------------
    # build the patch metalens for inverse projection
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
        
    ensure_dir(configs.plot.plot_root)

    # print(next(iter(test_loader))[0].shape)
    ## dummy forward to initialize quantizer
    model.set_near2far_matrix(near2far_matrices[-1])
    model_test.set_near2far_matrix(near2far_matrices[-1])
    model_test.set_test_mode()
    model(next(iter(test_loader))[0].to(device))
    model_test(next(iter(test_loader))[0].to(device))

    grad_scaler = amp.GradScaler(enabled=getattr(configs.run, "fp16", False))
    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")
    assert configs.invdes.reset_frequency in {"epoch", "proj"}, f"Unknown reset_frequency: {configs.invdes.reset_frequency}"



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

        # now we have successfully loaded the model
        batch_size = 10
        # random pick batch_size samples from the train_loader
        with DeterministicCtx(seed=41):
            samples = [train_loader.dataset[i] for i in random.sample(range(len(train_loader.dataset)), batch_size)]
        imgs = [sample[0] for sample in samples]
        imgs = torch.stack(imgs, dim=0).to(device)

        with torch.no_grad():
            _ = model_test.forward(imgs)

        x = imgs
        total_sys_lens = round(model.kernel_size_list[-1] * model.conv_cfg["resolution"] * model.conv_cfg["pixel_size_data"])
        # [B, 1, 28, 28]
        bs, C, H, W = x.shape
        p = 3
        calculate_diffraction_method = "near2far"
        # Ensure image size is sufficient
        assert H >= p and W >= p, "Image smaller than patch size"
        # Unfold: turn image into sliding patches
        patches = x.unfold(2, p, 1).unfold(3, p, 1)  # (bs, C, H-p+1, W-p+1, p, p)
        out_H = patches.shape[2]
        out_W = patches.shape[3]
        patches = patches.contiguous().view(bs, C, out_H * out_W, p * p)  # (bs, C, N, p*p)
        patches = patches.reshape(bs * C * out_H * out_W, p * p).unsqueeze(1)  # (bs*C*out_H*out_W, 1, p*p)
        source_mask = torch.ones_like(patches, dtype=torch.bool, device=patches.device)
        x = insert_zeros_after_every_N_except_last(patches, model.input_wg_width_px, model.input_wg_interval_px)
        source_mask = insert_zeros_after_every_N_except_last(source_mask, model.input_wg_width_px, model.input_wg_interval_px)
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

        current_pillar_width = [[ 0.0288,  0.0349,  0.0217,  0.0293,  0.0301,  0.0369,  0.0208,  0.0279,
          0.0295,  0.0370,  0.0291, -0.0057,  0.0300,  0.0329,  0.0154,  0.0255,
          0.0376,  0.0226,  0.0343,  0.0179,  0.0305,  0.0308,  0.0348,  0.0065,
          0.0353,  0.0211,  0.0295,  0.0372,  0.0304,  0.0144,  0.0374,  0.0251],
        [ 0.0296,  0.0355,  0.0218,  0.0383,  0.0364,  0.0303,  0.0333,  0.0396,
          0.0350,  0.0354,  0.0323,  0.0303,  0.0358,  0.0387,  0.0287,  0.0302,
          0.0353,  0.0308,  0.0338,  0.0339,  0.0374,  0.0333,  0.0393,  0.0244,
          0.0414,  0.0311,  0.0360,  0.0301,  0.0330,  0.0293,  0.0194,  0.0304]]
        wl = 0.85
        atom_period = 0.3
        num_atom = 32
        distance = 4
        plot_root = "./unitest/inference_field"
        ensure_dir(plot_root)
        res = 50

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
            port_len=(1, 4.5),
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
                    far_field_response_record=dict(
                        weight=0,
                        #### objective is evaluated at this port
                        in_slice_name="nearfield_1",
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
            port_len=(1, 4.5),
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
        
        region_1, x1 = calculate_diffraction(near2far_matrices, x[6092], device, mode=calculate_diffraction_method, diffract_opt=diffract_opt)
        print("this is the shape of the region_1: ", region_1.shape, flush=True)
        print("this is the shape of the x1: ", x1.shape, flush=True)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(torch.abs(region_1).cpu().detach().numpy())
        ax[0].set_title("Abs")
        ax[1].imshow(torch.angle(region_1).cpu().detach().numpy())
        ax[1].set_title("Angle")
        plt.savefig(os.path.join(exp_dir, f"region_1_{calculate_diffraction_method}.png"), dpi = 300)

        x_gt = x1.clone()
        x_ours = x1.clone()
        x_LPA = x1.clone()
        for i in range(len(current_pillar_width)):
            # here we calculate the diffraction field using FDFD simulation
            ls_knots = -0.05 * torch.ones(2 * len(current_pillar_width[i]) + 1, device=device)
            ls_knots[1::2] = torch.tensor(current_pillar_width[i], device=device)
            weight = {
                "design_region_0": ls_knots.unsqueeze(0),
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
            print("this is the shape of the x1: ", x1.shape, flush=True)
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
            # opt.plot(
            #     plot_filename=f"region_{i+2}.png",
            #     eps_map=opt._eps_map,
            #     obj=None,
            #     field_key=("in_slice_1", 0.85, "Hz1", 300),
            #     field_component="Hz",
            #     in_slice_name="in_slice_1",
            #     exclude_slice_names=[],
            # )
            Ez_gt = opt.objective.solutions[("in_slice_1", 0.85, "Hz1", 300)]["Hz"]
            Ez_gt = Ez_gt[
                -round(res*(0.5 + 3.7)):-round(res*0.5),
                round(res*0.5):-round(res*0.5),
            ]
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(torch.abs(Ez_gt.T).cpu().detach().numpy())
            ax[0].set_title("Abs")
            ax[1].imshow((Ez_gt.T.real).cpu().detach().numpy())
            ax[1].set_title("real")
            plt.savefig(os.path.join(exp_dir, f"region_{i+2}.png"), dpi = 300)
            plt.close()
            # here we calculate the diffraction field using our model
            tm = model_test.state_dict()[f"features.conv1.conv._conv_pos.metalens.{i}_17.W_buffer"]
            print("this is the shape of x_ours: ", x_ours.shape, flush=True)
            x_near_field = torch.matmul(x_ours.to(tm.dtype), tm.T)
            diffraction_region_ours, x_ours = calculate_diffraction(near2far_matrices, x_near_field, device, mode=calculate_diffraction_method, diffract_opt=diffract_opt)
            diffraction_region_ours = diffraction_region_ours / torch.max(diffraction_region_ours.abs())
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            im0 = ax[0].imshow(
                torch.abs(diffraction_region_ours).cpu().detach().numpy(), 
                vmax=0.5 if calculate_diffraction_method == "near2far" else 1,
            )
            ax[0].set_title("Abs")
            plt.colorbar(im0, ax=ax[0])
            ax[1].imshow(diffraction_region_ours.real.cpu().detach().numpy())
            ax[1].set_title("real")
            plt.savefig(os.path.join(exp_dir, f"region_{i+2}_ours_{calculate_diffraction_method}.png"), dpi = 300)
            plt.close()

            # here we calculate the diffraction field using LPA
            pillar_width = torch.tensor(current_pillar_width[i])
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
            LPA_tm_real = F.interpolate(LPA_tm.real.unsqueeze(0).unsqueeze(0), size=(480, 480), mode="bilinear", align_corners=False).squeeze()
            LPA_tm_imag = F.interpolate(LPA_tm.imag.unsqueeze(0).unsqueeze(0), size=(480, 480), mode="bilinear", align_corners=False).squeeze()
            LPA_tm = LPA_tm_real + 1j * LPA_tm_imag
            x_near_field_LPA = torch.matmul(x_LPA.to(LPA_tm.dtype), LPA_tm.T)
            diffraction_region_LPA, x_LPA = calculate_diffraction(near2far_matrices, x_near_field_LPA, device, mode=calculate_diffraction_method, diffract_opt=diffract_opt)
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(torch.abs(diffraction_region_LPA).cpu().detach().numpy())
            ax[0].set_title("Abs")
            ax[1].imshow(diffraction_region_LPA.real.cpu().detach().numpy())
            ax[1].set_title("real")
            plt.savefig(os.path.join(exp_dir, f"region_{i+2}_LPA_{calculate_diffraction_method}.png"), dpi = 300)
            plt.close()