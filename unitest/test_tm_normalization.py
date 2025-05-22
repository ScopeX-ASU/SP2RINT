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

def generate_smooth_curves(bs, length=480, num_waves=5, device="cpu"):
    """
    Generate a batch of smooth random curves using sine and cosine waves in PyTorch.
    
    Parameters:
        bs (int): Batch size (number of curves).
        length (int): Length of each curve.
        num_waves (int): Number of sine/cosine components combined.
        device (str): Device to use ("cpu" or "cuda").
    
    Returns:
        torch.Tensor: A tensor of shape (bs, length) containing smooth random curves in [-1, 1].
    """
    x = torch.linspace(0, 2 * torch.pi, length, device=device).unsqueeze(0)  # Shape: (1, length)
    curves = torch.zeros((bs, length), device=device)  # Initialize output tensor

    for _ in range(num_waves):
        freq = torch.rand((bs, 1), device=device) * 4.5 + 0.5  # Frequencies in [0.5, 5.0]
        amp = torch.rand((bs, 1), device=device) * 0.8 + 0.2   # Amplitudes in [0.2, 1.0]
        phase = torch.rand((bs, 1), device=device) * 2 * torch.pi  # Random phase shifts

        wave = amp * torch.sin(freq * x + phase) + amp * torch.cos(freq * x + phase)
        curves += wave

    # Normalize each curve independently to stay within [-1,1]
    curves /= curves.abs().max(dim=1, keepdim=True)[0]

    return curves

if __name__ == '__main__':
    device = torch.device("cuda:0")
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
        dz=4.0,
        param_method="level_set",
        tm_norm="field",
    )
    # hr_transfer_matrix = model.forward(sharpness=256, down_sample_rate=1)
    transfer_matrix = model.forward(sharpness=256, down_sample_rate=down_sample_rate)

    with DeterministicCtx(seed=0):
        random_source_real = generate_smooth_curves(1, 480, 5, device)
        random_source_imag = generate_smooth_curves(1, 480, 5, device)
        random_source = random_source_real + 1j * random_source_imag
    random_source = random_source.squeeze()
    additional_zero_padding = torch.zeros(25).to(device)
    random_source = torch.cat((additional_zero_padding, random_source, additional_zero_padding), dim=0)
    custom_source = dict(
        source=random_source,
        slice_name="in_slice_1",
        mode="Hz1",
        wl=0.85,
        direction="x+",
    )
    _ = model.total_opt(
        sharpness=256,
        weight={"design_region_0": model.level_set_knots.unsqueeze(0)},
        custom_source=custom_source,
    )

    gt_near_field = model.total_opt.objective.response[('in_slice_1', 'nearfield_1', 0.85, "Hz1", 300)]["fz"].squeeze()
    hr_gt_energy = gt_near_field.abs().square().sum()
    source_field = model.total_opt.objective.response[('in_slice_1', 'in_slice_1', 0.85, "Hz1", 300)]["fz"].squeeze()
    source_field = source_field[25:-25]
    hr_source_energy = source_field.abs().square().sum()
    print(f"HR Source energy: {hr_source_energy}")
    print(f"HR GT energy: {hr_gt_energy}")
    print(f"HR Energy ratio: {hr_gt_energy / hr_source_energy}")

    # use a kernel size of 15 to avg the source field
    if downsample_method == "avg":
        source_field_real = F.avg_pool1d(source_field.real.unsqueeze(0).unsqueeze(0), kernel_size=down_sample_rate, stride=down_sample_rate).squeeze()
        source_field_imag = F.avg_pool1d(source_field.imag.unsqueeze(0).unsqueeze(0), kernel_size=down_sample_rate, stride=down_sample_rate).squeeze()
        source_field = source_field_real + 1j * source_field_imag
    elif downsample_method == "point":
        source_field = source_field[down_sample_rate//2::down_sample_rate] * down_sample_rate
    lr_source_energy = source_field.abs().square().sum()
    lr_gt_near_field = gt_near_field[down_sample_rate//2::down_sample_rate]
    lr_gt_energy = lr_gt_near_field.abs().square().sum()
    print(f"LR Source energy: {lr_source_energy}")
    print(f"LR GT energy: {lr_gt_energy}")
    print(f"LR Energy ratio: {lr_gt_energy / lr_source_energy}")

    calculated_near_field = (transfer_matrix @ source_field)
    plt.figure()
    plt.plot(lr_gt_near_field.abs().detach().cpu().numpy(), label="GT Near Field")
    plt.plot(calculated_near_field.abs().detach().cpu().numpy(), label="Calculated Near Field")
    plt.legend()
    plt.savefig(f"./figs/cpr_near_field_mag_ds{down_sample_rate}.png")
    plt.close()

    plt.figure()
    plt.plot(torch.angle(lr_gt_near_field).detach().cpu().numpy(), label="GT Near Field")
    plt.plot(torch.angle(calculated_near_field).detach().cpu().numpy(), label="Calculated Near Field")
    plt.legend()
    plt.savefig(f"./figs/cpr_near_field_phase_ds{down_sample_rate}.png")
    plt.close()

    plt.figure()
    plt.plot((calculated_near_field.abs() / lr_gt_near_field.abs()).detach().cpu().numpy())
    plt.savefig(f"./figs/cpr_near_field_ratio_ds{down_sample_rate}.png")
    plt.close()

    near2far_matrix = probe_near2far_matrix(
        model.total_opt,
        0.85,
        device,
    ).to(gt_near_field.dtype)
    ds_near2far_matrix = near2far_matrix[
        down_sample_rate//2::down_sample_rate, 
        :
    ]
    ds_near2far_matrix = ds_near2far_matrix.reshape(ds_near2far_matrix.shape[0], -1, down_sample_rate).sum(dim=-1)
    print(ds_near2far_matrix.shape)

    gt_far_field = near2far_matrix @ gt_near_field
    gt_far_field = gt_far_field[down_sample_rate//2::down_sample_rate]
    calculated_far_field = ds_near2far_matrix @ calculated_near_field

    plt.figure()
    plt.plot(gt_far_field.abs().detach().cpu().numpy(), label="GT Far Field")
    plt.plot(calculated_far_field.abs().detach().cpu().numpy(), label="Calculated Far Field")
    plt.legend()
    plt.savefig(f"./figs/cpr_far_field_mag_ds{down_sample_rate}.png")
    plt.close()

    plt.figure()
    plt.plot(torch.angle(gt_far_field).detach().cpu().numpy(), label="GT Far Field")
    plt.plot(torch.angle(calculated_far_field).detach().cpu().numpy(), label="Calculated Far Field")
    plt.legend()
    plt.savefig(f"./figs/cpr_far_field_phase_ds{down_sample_rate}.png")
    plt.close()

    plt.figure()
    plt.plot((calculated_far_field.abs() / gt_far_field.abs()).detach().cpu().numpy())
    plt.savefig(f"./figs/cpr_far_field_ratio_ds{down_sample_rate}.png")
    plt.close()