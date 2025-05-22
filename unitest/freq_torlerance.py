'''
we need to determine what is the frequency tolerance of the near field of metasurface T_{i-1}

1. we randomly generate one or multiple metasurfaces
2. we probe the high resolution transfer matrix and down sample it on the column dimension
3. use fourier basis to probe the output of the hr transfer matrix and the downsampled transfer matrix
4. compare the normalized L2 norm of the two outputs --> determine what is the torlerance of the freq of the incident light
5. we use another fourier basis and after diffraction, we can see what is the tolerance of freq of the output near field of the metasurface T_{i-1}
'''

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
):
    # time_start = time.time()
    number_atoms = 32
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
            ls_knot = -0.05 * torch.ones_like(patched_metalens.level_set_knots)
            ls_knot[1::2] = patched_metalens.weights.data
            _ = patched_metalens.total_opt(
                sharpness=256, 
                weight={"design_region_0": ls_knot.unsqueeze(0)},
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
    patch_size = 17
    down_sample_rate = 1
    exp_root = "./unitest/freq_tolerance"
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
    plot_root = exp_root
    opts = None
    total_opt = None
    if os.path.exists(os.path.join(exp_root, f"full_surface_tm_{batch_size}.h5")):
        print("the full surface tm has already been computed, skip this step")
        with h5py.File(os.path.join(exp_root, f"full_surface_tm_{batch_size}.h5"), "r") as f:
            full_surface_tm = torch.tensor(f["full_surface_tm"], device=device)
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
    else:
        full_surface_tm_list = []
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
        if not hasattr(surface_calculator, "level_set_knots"):
            surface_calculator.level_set_knots = -0.05 * torch.ones(32*2+1, device=device)
        for idx in range(batch_size):
            surface_calculator.disable_solver_cache()
            surface_calculator.direct_set_pillar_width(ls_knots[idx])
            full_surface_tm = probe_full_tm(
                device,
                patched_metalens=surface_calculator,
                full_wave_down_sample_rate=down_sample_rate,
            )
            full_surface_tm_list.append(full_surface_tm)
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            im0 = ax[0].imshow(
                full_surface_tm.abs().detach().cpu().numpy()
            )
            ax[0].set_title("full surface tm magnitude")
            fig.colorbar(im0, ax=ax[1])
            im1 = ax[1].imshow(
                torch.angle(full_surface_tm).detach().cpu().numpy()
            )
            ax[1].set_title("full surface tm phase")
            fig.colorbar(im1, ax=ax[0])
            plt.savefig(os.path.join(plot_root, f"full_vs_patched_tm_{idx}.png"))
            plt.close(fig)

        if not os.path.exists(os.path.join(exp_root, f"full_surface_tm_{batch_size}.h5")):
            full_surface_tm = torch.stack(full_surface_tm_list, dim=0)
            # save it to a h5 file
            with h5py.File(os.path.join(exp_root, f"full_surface_tm_{batch_size}.h5"), "w") as f:
                f.create_dataset("full_surface_tm", data=full_surface_tm.detach().cpu().numpy())
        quit()

    # need to downsample the full surface tm
    ds_surface_tm = full_surface_tm.view(full_surface_tm.shape[0], full_surface_tm.shape[1], -1, 15).sum(dim=-1)

    fourier_src = torch.eye(full_surface_tm.shape[-1], device=device)
    fourier_src = torch.fft.ifft(fourier_src, dim=-1).to(full_surface_tm.dtype) # bs, 480

    ds_fourier_src = torch.eye(ds_surface_tm.shape[-1], device=device)
    ds_fourier_src = torch.fft.ifft(ds_fourier_src, dim=-1).to(ds_surface_tm.dtype) # bs, 32

    response_list = []
    for i in range(batch_size):
        response_list.append(
            torch.matmul(fourier_src, full_surface_tm[i].T)
        )
    response = torch.stack(response_list, dim=0)
    print("this is the size of the response", response.shape)

    ds_response_list = []
    for i in range(batch_size):
        ds_response_list.append(
            torch.matmul(ds_fourier_src, ds_surface_tm[i].T)
        )
    ds_response = torch.stack(ds_response_list, dim=0)
    ds_response = ds_response / torch.norm(ds_response, p=2, dim=-1, keepdim=True)
    print("this is the size of the ds response", ds_response.shape)

    hr_response = torch.cat(
        [
            response[
                :,
                :ds_response.shape[1] // 2,
                :,
            ],
            response[
                :,
                -ds_response.shape[1] // 2:,
                :,
            ]
        ],
        dim=1
    )
    hr_response = hr_response / torch.norm(hr_response, p=2, dim=-1, keepdim=True)

    error = hr_response - ds_response
    n_l2norm = torch.norm(error, p=2, dim=-1) / torch.norm(hr_response, p=2, dim=-1)
    n_l2norm = n_l2norm.mean(dim=0)
    print("this is the normalized L2 norm of the error", n_l2norm)


    for i in range(10):
        fig, ax = plt.subplots(2, 2, figsize=(10, 5))
        im0 = ax[0, 0].plot(
            ds_fourier_src[i].real.detach().cpu().numpy()
        )
        ax[0, 0].set_title("ds fourier src real")
        im1 = ax[0, 1].plot(
            fourier_src[i].real.detach().cpu().numpy()
        )
        ax[0, 1].set_title("fourier src real")
        im2 = ax[1, 0].plot(
            ds_fourier_src[i].imag.detach().cpu().numpy()
        )
        ax[1, 0].set_title("ds fourier src imag")
        im3 = ax[1, 1].plot(
            fourier_src[i].imag.detach().cpu().numpy()
        )
        ax[1, 1].set_title("fourier src imag")
        plt.savefig(os.path.join(plot_root, f"ds_fourier_src_{i}.png"))
        plt.close(fig)

        fig, ax = plt.subplots(2, 2, figsize=(10, 5))
        im0 = ax[0, 0].plot(
            ds_response[5][i].real.detach().cpu().numpy()
        )
        ax[0, 0].set_title("ds response real")
        im1 = ax[0, 1].plot(
            ds_response[5][i].imag.detach().cpu().numpy()
        )
        ax[0, 1].set_title("ds response imag")
        im2 = ax[1, 0].plot(
            hr_response[5][i].real.detach().cpu().numpy()
        )
        ax[1, 0].set_title("hr response real")
        im3 = ax[1, 1].plot(
            hr_response[5][i].imag.detach().cpu().numpy()
        )
        ax[1, 1].set_title("hr response imag")
        plt.savefig(os.path.join(plot_root, f"hr_response_{i}.png"))
        plt.close(fig)

    with open(os.path.join(plot_root, "incident_light_error.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Freq idx", "Response N-L2Norm"])
        for i in range(len(n_l2norm)):
            writer.writerow([i, n_l2norm[i].item()])

    last_surface_near_field_src = torch.eye(full_surface_tm.shape[-1], device=device)
    last_surface_near_field_src = torch.fft.ifft(last_surface_near_field_src, dim=-1).to(full_surface_tm.dtype) # bs, 480

    near2far_matrix = probe_near2far_matrix(
        surface_calculator.total_opt,
        0.85, # in um
        device,
    ).to(full_surface_tm.dtype)
    ds_near2far_matrix = near2far_matrix[
        15 // 2::15,
        :,
    ]
    post_diffraction_src = torch.matmul(
        last_surface_near_field_src,
        near2far_matrix.T
    ) # bs, 480
    ds_post_diffraction_src = torch.matmul(
        last_surface_near_field_src,
        ds_near2far_matrix.T
    ) # bs, 32

    ds_post_diffraction_src_spectrum = torch.fft.fft(ds_post_diffraction_src, dim=-1)
    high_freq_percentage = torch.sum(
        ds_post_diffraction_src_spectrum[:, 5:-5].abs().square(),
        dim=-1
    ) / torch.sum(
        ds_post_diffraction_src_spectrum.abs().square(),
        dim=-1
    )
    print("this is the high freq percentage", high_freq_percentage)

    end2end_response_list = []
    for i in range(batch_size):
        end2end_response_list.append(
            torch.matmul(post_diffraction_src, full_surface_tm[i].T)
        )
    end2end_response = torch.stack(end2end_response_list, dim=0)

    ds_end2end_response_list = []
    for i in range(batch_size):
        ds_end2end_response_list.append(
            torch.matmul(ds_post_diffraction_src, ds_surface_tm[i].T)
        )
    ds_end2end_response = torch.stack(ds_end2end_response_list, dim=0)

    ds_end2end_response = ds_end2end_response / torch.norm(ds_end2end_response, p=2, dim=-1, keepdim=True)
    print("this is the size of the ds end2end response", ds_end2end_response.shape)

    hr_end2end_response = torch.cat(
        [
            end2end_response[
                :,
                :ds_end2end_response.shape[1] // 2,
                :,
            ],
            end2end_response[
                :,
                -ds_end2end_response.shape[1] // 2:,
                :,
            ]
        ],
        dim=1
    )
    hr_end2end_response = hr_end2end_response / torch.norm(hr_end2end_response, p=2, dim=-1, keepdim=True)

    error = hr_end2end_response - ds_end2end_response
    n_l2norm = torch.norm(error, p=2, dim=-1) / torch.norm(hr_end2end_response, p=2, dim=-1)
    n_l2norm = n_l2norm.mean(dim=0)
    print("this is the normalized L2 norm of the error", n_l2norm)
    plt.figure()
    plt.plot(n_l2norm.detach().cpu().numpy())
    plt.title("end2end NL2norm of the error")
    plt.savefig(os.path.join(plot_root, "end2end_error.png"))
    plt.close()


if __name__ == "__main__":
    exp_patch_size()