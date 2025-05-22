'''
in this script, we will test the patch size of the surface
1. we will first generate 25 random metasurfaces with size of 32
2. then we will probe the full tranfer matrix of them
3. then we will use different patch size to get the transfer matrix
3.5 then we can save all the transfer matrix to a h5 file so that we don't need to recompute them if we need to rerun the experiment
4. then we will compare the probed transfer matrix with the full transfer matrix (ground truth)
5. we can also columnwisely compare the transfer matrix with the full transfer matrix (ground truth) to see the error distribution
'''

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from thirdparty.MAPS_local.core.invdes.models.base_optimization import DefaultSimulationConfig
from thirdparty.MAPS_local.core.invdes.models import (
    MetaLensOptimization,
)
from thirdparty.MAPS_local.core.invdes.models.layers import MetaLens
from thirdparty.MAPS_local.core.utils import SharpnessScheduler
from pyutils.general import ensure_dir
from core.utils import (
    probe_near2far_matrix, 
    DeterministicCtx,
    get_mid_weight,
    TransferMatrixMatchingLoss,
)
from core.models.patch_metalens import PatchMetalens
sys.path.pop()
from pyutils.torch_train import (
    set_torch_deterministic,
)
import csv
import numpy
import h5py
import matplotlib.pyplot as plt

def probe_full_tm(
    device,
    patched_metalens,
    full_wave_down_sample_rate = 1,
    number_atoms = 32,
):
    # time_start = time.time()
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
            _ = patched_metalens.total_opt(
                sharpness=256, 
                weight={"design_region_0": -0.05 * torch.ones_like(patched_metalens.level_set_knots.unsqueeze(0))},
                custom_source=custom_source
            )

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
            _ = patched_metalens.total_opt(
                sharpness=256, 
                weight={"design_region_0": patched_metalens.level_set_knots.unsqueeze(0)},
                custom_source=custom_source
            )

            response = patched_metalens.total_opt.objective.response[('in_slice_1', 'nearfield_1', 0.85, "Hz1", 300)]["fz"].squeeze()
            response = response[full_wave_down_sample_rate // 2 :: full_wave_down_sample_rate]
            assert len(response) == number_atoms * round(15 // full_wave_down_sample_rate), f"{len(response)}!= {number_atoms * round(15 // full_wave_down_sample_rate)}"
            full_wave_response[idx] = response
        full_wave_response = full_wave_response.transpose(0, 1)
        normalizer = torch.stack(patched_metalens.total_normalizer_list, dim=0).to(device)
        normalizer = normalizer.unsqueeze(1)
        full_wave_response = full_wave_response / normalizer

    if hasattr(patched_metalens.total_opt.objective.sims[sim_key[0]].solver, "clear_solver_cache"):
        patched_metalens.total_opt.objective.sims[sim_key[0]].solver.clear_solver_cache()
        patched_metalens.total_opt.objective.sims[sim_key[0]].solver.set_cache_mode(False)

    # time_end = time.time()
    # print(f"this is the time for probing the full wave response: {time_end - time_start}", flush=True)
    return full_wave_response

def exp_patch_size():
    # ------------prepare the necessary things starts----------------
    batch_size = 100
    patch_size_list = [5, 7, 9, 11, 13, 15, 17, 19, 21]
    # patch_size_list = [5,7,9,17]
    down_sample_rate = 15
    exp_root = "./unitest/patch_size_exp"
    ensure_dir(exp_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    with DeterministicCtx(seed=41):
        ls_knots = 0.27 * torch.rand(batch_size - 2, 32, device=device) + 0.01
        uniform_surface = 0.1 * torch.ones(32, device=device).unsqueeze(0)
        phase_changing = torch.tensor([0.06, 0.119] * 16, device=device).unsqueeze(0)
        ls_knots = torch.cat([ls_knots, uniform_surface, phase_changing], dim=0)
        ls_knots = get_mid_weight(0.05, ls_knots)
    # ------------prepare the necessary things end----------------
    for patch_size in patch_size_list:
        plot_root = exp_root + f"/patch_size_{patch_size}"
        ensure_dir(plot_root)
        opts = None
        total_opt = None
        if os.path.exists(os.path.join(exp_root, f"full_surface_tm_{batch_size}.h5")):
            print("the full surface tm has already been computed, skip this step")
        else:
            full_surface_tm_list = []
        if os.path.exists(os.path.join(plot_root, f"patched_surface_tm_{batch_size}.h5")):
            print("the patched surface tm has already been computed, skip this step")
            continue
        else:
            patched_surface_tm_list = []
        surface_calculator = PatchMetalens(
            atom_period=0.3,
            patch_size=patch_size,
            num_atom=32,
            probing_region_size=patch_size,
            target_phase_response=None,
            LUT=LUT,
            device=device,
            target_dx=0.3,
            plot_root=plot_root,
            downsample_mode="both",
            downsample_method="avg",
            dz=4.0,
            param_method="level_set",
            tm_norm="field",
            field_norm_condition="wo_lens",
        )
        for idx in range(batch_size):
            surface_calculator.disable_solver_cache()
            surface_calculator.direct_set_pillar_width(ls_knots[idx])
            if os.path.exists(os.path.join(plot_root, f"patched_surface_tm_{batch_size}.h5")):
                with h5py.File(os.path.join(plot_root, f"patched_surface_tm_{batch_size}.h5"), "r") as f:
                    patched_surface_tm = torch.tensor(f["patched_surface_tm"][idx], device=device)
            else:
                with torch.no_grad():
                    patched_surface_tm = surface_calculator.forward(sharpness=256, down_sample_rate=down_sample_rate)
                    patched_surface_tm_list.append(patched_surface_tm)
            if os.path.exists(os.path.join(exp_root, f"full_surface_tm_{batch_size}.h5")):
                with h5py.File(os.path.join(exp_root, f"full_surface_tm_{batch_size}.h5"), "r") as f:
                    full_surface_tm = torch.tensor(f["full_surface_tm"][idx], device=device)
            else:
                full_surface_tm = probe_full_tm(
                    device,
                    patched_metalens=surface_calculator,
                    full_wave_down_sample_rate=down_sample_rate,
                )
                full_surface_tm_list.append(full_surface_tm)
            max_abs_value = max(
                full_surface_tm.abs().max(),
                patched_surface_tm.abs().max(),
            )
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"patch size: {patch_size}")
            im0 = ax[0].imshow(
                full_surface_tm.abs().detach().cpu().numpy(), vmin=0, vmax=max_abs_value
            )
            ax[0].set_title("full surface tm")
            fig.colorbar(im0, ax=ax[1])
            im1 = ax[1].imshow(
                patched_surface_tm.abs().detach().cpu().numpy(), vmin=0, vmax=max_abs_value
            )
            ax[1].set_title("patched surface tm")
            fig.colorbar(im1, ax=ax[0])
            im2 = ax[2].imshow(
                (patched_surface_tm - full_surface_tm).detach().abs().cpu().numpy(), vmin=0, vmax=max_abs_value
            )
            ax[2].set_title("diff tm")
            fig.colorbar(im2, ax=ax[2])
            plt.savefig(os.path.join(plot_root, f"full_vs_patched_tm_{idx}.png"))
            plt.close(fig)
        if not os.path.exists(os.path.join(plot_root, f"patched_surface_tm_{batch_size}.h5")):
            patched_surface_tm = torch.stack(patched_surface_tm_list, dim=0)
            # save it to a h5 file
            with h5py.File(os.path.join(plot_root, f"patched_surface_tm_{batch_size}.h5"), "w") as f:
                f.create_dataset("patched_surface_tm", data=patched_surface_tm.detach().cpu().numpy())

        if not os.path.exists(os.path.join(exp_root, f"full_surface_tm_{batch_size}.h5")):
            full_surface_tm = torch.stack(full_surface_tm_list, dim=0)
            # save it to a h5 file
            with h5py.File(os.path.join(exp_root, f"full_surface_tm_{batch_size}.h5"), "w") as f:
                f.create_dataset("full_surface_tm", data=full_surface_tm.detach().cpu().numpy())

    with h5py.File(os.path.join(exp_root, f"full_surface_tm_{batch_size}.h5"), "r") as f:
        full_surface_tm = torch.tensor(f["full_surface_tm"], device=device)
        avg_full_surface_tm = full_surface_tm.mean(dim=0)
        plt.figure(figsize=(12, 6))
        plt.title(f"full surface tm")
        plt.imshow(avg_full_surface_tm.abs().detach().cpu().numpy())
        plt.colorbar()
        plt.savefig(os.path.join(exp_root, f"full_surface_tm.png"), dpi=300)
        plt.close()
    quit()

    error_dict = {}
    for patch_size in patch_size_list:
        plot_root = exp_root + f"/patch_size_{patch_size}"
        with h5py.File(os.path.join(plot_root, f"patched_surface_tm_{batch_size}.h5"), "r") as f:
            patched_surface_tm = torch.tensor(f["patched_surface_tm"], device=device)
        with h5py.File(os.path.join(exp_root, f"full_surface_tm_{batch_size}.h5"), "r") as f:
            full_surface_tm = torch.tensor(f["full_surface_tm"], device=device)

        error = patched_surface_tm - full_surface_tm
        error_l2_norm = error.norm(p=2, dim=[-1, -2])
        ground_truth_l2_norm = full_surface_tm.norm(p=2, dim=[-1, -2])
        normalized_l2_norm_mean, normalized_l2_norm_std = (error_l2_norm / ground_truth_l2_norm).mean(), (error_l2_norm / ground_truth_l2_norm).std()
        error_dict[patch_size] = (normalized_l2_norm_mean.item(), normalized_l2_norm_std.item())
    print(error_dict)
    with open(os.path.join(exp_root, "error_dict.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["patch_size", "normalized_l2_norm_mean", "normalized_l2_norm_std"])
        for patch_size, (mean, std) in error_dict.items():
            writer.writerow([patch_size, mean, std])

    column_wise_error = {}
    for patch_size in patch_size_list:
        plot_root = exp_root + f"/patch_size_{patch_size}"
        with h5py.File(os.path.join(plot_root, f"patched_surface_tm_{batch_size}.h5"), "r") as f:
            patched_surface_tm = torch.tensor(f["patched_surface_tm"], device=device)
        with h5py.File(os.path.join(exp_root, f"full_surface_tm_{batch_size}.h5"), "r") as f:
            full_surface_tm = torch.tensor(f["full_surface_tm"], device=device)

        error = patched_surface_tm - full_surface_tm # bs, H, W
        error_col_wise = error.norm(p=2, dim=[-2])
        ground_truth_col_wise = full_surface_tm.norm(p=2, dim=[-2])
        normalized_col_wise_mean, normalized_col_wise_std = (error_col_wise / ground_truth_col_wise).mean(dim=0), (error_col_wise / ground_truth_col_wise).std(dim=0)
        column_wise_error[patch_size] = (normalized_col_wise_mean.cpu().numpy(), normalized_col_wise_std.cpu().numpy())
        # print(f"this is the size of normalized_col_wise_mean: {normalized_col_wise_mean.shape}") [32]
    plt.figure(figsize=(12, 6))
    plt.title(f"column wise error")
    plt.xlabel("Position")
    plt.ylabel("Column N-L2 Norm")
    plt.xticks(numpy.arange(0, 32, 1))
    plt.xlim(0, 32)
    plt.grid()
    for patch_size, (mean, std) in column_wise_error.items():
        plt.plot(mean, label=f"patch size: {patch_size}")
        plt.fill_between(numpy.arange(0, 32, 1), mean - std, mean + std, alpha=0.2)
    plt.legend()
    plt.savefig(os.path.join(exp_root, "column_wise_error.png"), dpi=300)
    plt.close()




if __name__ == "__main__":
    exp_patch_size()