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
import time

def exp_patch_size():
    # ------------prepare the necessary things starts----------------
    batch_size = 1
    patch_size_list = [5, 7, 9, 11, 13, 15, 17, 19, 21]
    # patch_size_list = [5,7,9]
    down_sample_rate = 15
    exp_root = "./unitest/patch_size_exp_speed"
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
        ls_knots = 0.27 * torch.rand(batch_size, 32, device=device) + 0.01
        ls_knots = get_mid_weight(0.05, ls_knots)
    # ------------prepare the necessary things end----------------
    patched_surface_tm_time_dict = {}
    for patch_size in patch_size_list:
        plot_root = exp_root + f"/patch_size_{patch_size}"
        ensure_dir(plot_root)
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
            with torch.no_grad():
                time_start = time.time()
                patched_surface_tm = surface_calculator.forward(sharpness=256, in_down_sample_rate=down_sample_rate)
                time_end = time.time()
                patched_surface_tm_time = (time_end - time_start) / 2 # norm run and the probing time are included
                patched_surface_tm_time_dict[patch_size] = patched_surface_tm_time
    with open(os.path.join(exp_root, "patched_surface_tm_time.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["patch_size", "patched_surface_tm_time"])
        for patch_size, patched_surface_tm_time in patched_surface_tm_time_dict.items():
            writer.writerow([patch_size, patched_surface_tm_time])
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