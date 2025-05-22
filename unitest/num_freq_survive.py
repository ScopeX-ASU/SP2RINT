import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.models.patch_metalens import PatchMetalens
from core.utils import DeterministicCtx
from core.utils import probe_near2far_matrix
sys.path.pop()

if __name__ == '__main__':
    device = torch.device("cuda:1")
    down_sample_rate = 15
    downsample_method = "avg"
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
    # Define the model
    model = PatchMetalens(
        atom_period=0.3,
        patch_size=17,
        num_atom=32,
        probing_region_size=17,
        target_phase_response=None,
        LUT=LUT,
        device=device,
        target_dx=0.3,
        plot_root="./figs/patched_metalens",
        downsample_mode="both",
        downsample_method=downsample_method,
        dz=0.3 + 1e-12,
        param_method="level_set",
        tm_norm="field",
        field_norm_condition="wo_lens",
    )

    near2far_matrix = probe_near2far_matrix(
        model.total_opt,
        0.85,
        device,
    )

    plt.figure(figsize=(12, 6))
    plt.imshow(
        torch.abs(near2far_matrix).cpu().numpy(),
    )
    plt.title("Near to Far Field Matrix")
    plt.colorbar()
    plt.savefig("./unitest/near2far_matrix.png", dpi=300)
    print("this is the near2far_matrix:", near2far_matrix, flush=True)
    # quit()


    ds_near2far_matrix = near2far_matrix[
        down_sample_rate // 2::down_sample_rate,
    ]

    # randomly generate a batch of random curves
    # Generate random sources using PyTorch
    bs = 1000
    length = 480
    spectrum = torch.randn((bs, length), device=device, dtype=near2far_matrix.dtype)
    sources = torch.fft.ifft(spectrum, dim=1)
    # Compute FFT (energy spectrum) for each source
    fft_sources = torch.fft.fft(sources, dim=1)
    energy_spectrum = torch.abs(fft_sources) ** 2 # 1000， 480

    # Compute mean energy spectrum across the batch dimension
    mean_energy_spectrum = torch.mean(energy_spectrum, dim=0).cpu().numpy()

    post_diffraction_field = torch.matmul(sources, near2far_matrix.T)
    print("this is the shape of post_diffraction_field:", post_diffraction_field.shape)

    post_diffraction_energy_spectrum = (torch.abs(torch.fft.fft(post_diffraction_field, dim=1)) ** 2) / energy_spectrum
    mean_post_diffraction_energy_spectrum = torch.mean(post_diffraction_energy_spectrum, dim=0).cpu().numpy()
    print("this is the mean_post_diffraction_energy_spectrum:", mean_post_diffraction_energy_spectrum / mean_post_diffraction_energy_spectrum.max())

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(mean_energy_spectrum, label='Mean Energy Spectrum')
    ax[0].set_title('Mean Energy Spectrum')
    ax[0].set_xlabel('Frequency')
    ax[0].set_ylabel('Energy')
    ax[0].legend()
    ax[1].plot(mean_post_diffraction_energy_spectrum, label='Mean Post Diffraction Energy Spectrum')
    ax[1].set_title('Mean Post Diffraction Energy Spectrum')
    ax[1].set_xlabel('Frequency')
    ax[1].set_ylabel('Energy')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("./unitest/energy_spectrum.png", dpi=300)

    

    