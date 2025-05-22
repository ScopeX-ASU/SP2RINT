'''
this script is to sweep the following parameters:
1. window size for equivalent convolution kernel 
2. number of the output channel for the equivalent convolution kernel
3. the gap between waveguides
4. the width of the waveguide

To inverse design a random transfer matrix, we need to implement the following steps:
1. for a given M and N, randomly generate a complex transfer matrix T of size MxN
2. calculate the transfer matrix of the two metasurfaces
3. inject unit stimulus into the first waveguide
4. calculate the output field on the receiver using the transfer matrix got in step 2 and the diffraction matrix
5. repeat step 3 and 4 for all the input waveguides
6. calculate the loss and gradient descent
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
)
sys.path.pop()
from pyutils.torch_train import (
    set_torch_deterministic,
)
import csv
import numpy

def normalize_transfer_matrix(T: torch.Tensor) -> torch.Tensor:
    col_mean = T.mean(dim=0, keepdim=True)
    col_std = T.std(dim=0, keepdim=True) + 1e-6
    T = (T - col_mean) / col_std
    T = T - T.min(dim=0, keepdim=True).values
    col_mean = T.mean(dim=0, keepdim=True)
    col_std = T.std(dim=0, keepdim=True) + 1e-6
    T = (T - col_mean) / col_std
    T = T - T.min(dim=0, keepdim=True).values
    return T

class ColumnWiseCosSimLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target: torch.Tensor, probed: torch.Tensor) -> torch.Tensor:
        """
        Compare two transfer matrices of shape (H, W) using column-wise cosine similarity.
        Returns 1 - average cosine similarity.
        """
        assert target.shape == probed.shape and target.ndim == 2, f"target_shape is {target.shape}, probed_shape is {probed.shape}, target_ndim is {target.ndim}, probed_ndim is {probed.ndim}"

        # Unit-norm each column
        target_norm = target / (target.norm(dim=0, keepdim=True) + 1e-6)
        probed_norm = probed / (probed.norm(dim=0, keepdim=True) + 1e-6)

        # Cosine similarity per column
        cos_sim = torch.sum(target_norm * probed_norm, dim=0)  # (W,)
        mean_sim = cos_sim.mean()  # scalar

        return 1.0 - mean_sim
    
def normalize_transfer_matrix(T: torch.Tensor) -> torch.Tensor:
    """
    Normalize a transfer matrix (shape [..., H, W]) such that:
        - Each column (dim=-2) has std=1
        - All values are positive (min=0 per column)
    Supports batched or non-batched input.
    """
    # Step 1: standardize column-wise
    col_mean = T.mean(dim=-2, keepdim=True)
    col_std = T.std(dim=-2, keepdim=True) + 1e-6
    T = (T - col_mean) / col_std

    # Step 2: shift so all values are ≥ 0
    col_min = T.min(dim=-2, keepdim=True).values
    T = T - col_min

    # Step 3: re-standardize (optional, but keeps consistent std=1 after shift)
    col_mean = T.mean(dim=-2, keepdim=True)
    col_std = T.std(dim=-2, keepdim=True) + 1e-6
    T = (T - col_mean) / col_std
    T = T - T.min(dim=-2, keepdim=True).values

    return T
def build_donn_input_sources(
    num_waveguides: int,
    resolution: int,            # e.g., 50 px/µm
    input_wg_width: float,      # in µm
    input_wg_gap: float,        # in µm
    total_pixels: int = 480,    # total width of metasurface
    device=None
):
    # Convert width and gap from µm to pixels
    wg_width_px = int(round(input_wg_width * resolution))
    wg_gap_px = int(round(input_wg_gap * resolution))

    # Calculate total span of all waveguides and gaps
    total_input_width = num_waveguides * wg_width_px + (num_waveguides - 1) * wg_gap_px
    if total_input_width > total_pixels:
        raise ValueError("Input waveguides + gaps exceed metasurface width!")

    # Center start position
    start = (total_pixels - total_input_width) // 2

    # Create tensor
    inputs = torch.zeros((num_waveguides, total_pixels), device=device, dtype=torch.complex128)

    for i in range(num_waveguides):
        left = start + i * (wg_width_px + wg_gap_px)
        right = left + wg_width_px
        # Inject uniform 1s over the width (can replace with 1-hot if needed)
        inputs[i, left:right] = 1.0

    return inputs


class MetaSurface(nn.Module):
    def __init__(
            self, 
            num_atom, 
            atom_period, 
            dz,
            normalizer_list=None,
            plot_root='./unitest/plot/',
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        ):
        super(MetaSurface, self).__init__()
        self.wl = 0.85
        self.res = 50
        self.device = device
        self.plot_root = plot_root

        self.num_atom = num_atom
        self.atom_period = atom_period
        self.dz = dz

        self.build_total_sim()
        self.build_parameter()
        if normalizer_list is None:
            self.normalizer_list = self.calculate_normalizer()
        else:
            self.normalizer_list = normalizer_list

    def build_total_sim(self):
        total_sim_cfg = DefaultSimulationConfig()
        total_sim_cfg.update(
            dict(
                solver="ceviche_torch",
                numerical_solver="solve_direct",
                use_autodiff=False,
                neural_solver=None,
                border_width=[0, 0, 0.5, 0.5],
                PML=[0.5, 0.5],
                resolution=self.res,
                wl_cen=self.wl,
                plot_root=self.plot_root,
            )
        )
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
            farfield_dxs=((self.dz, self.dz + 2/self.res),),
            farfield_sizes=(self.atom_period * self.num_atom,),
            device=self.device,
        )
        hr_total_metalens = total_metalens.copy(resolution=200)
        total_opt = MetaLensOptimization(
            device=total_metalens,
            hr_device=hr_total_metalens,
            sim_cfg=total_sim_cfg,
            operation_device=self.device,
            design_region_param_cfgs={},
        )
        self.opt = total_opt

    def build_parameter(self):
        self.weights = nn.Parameter(
            0.05 * torch.ones((self.num_atom), device=self.device)
        )
    
    def build_ls_knots(self):
        ls_knots = -0.05 * torch.ones((2 * self.num_atom + 1), device=self.device)
        ls_knots[1::2] = self.weights

        return ls_knots
    
    def calculate_normalizer(self):
        full_wave_down_sample_rate = 1
        sources = torch.eye(self.num_atom * round(15 // full_wave_down_sample_rate), device=self.device)
        total_normalizer_list = []
        with torch.no_grad():
            for idx in range(self.num_atom * round(15 // full_wave_down_sample_rate)):
                source_i = sources[idx].repeat_interleave(full_wave_down_sample_rate)
                source_zero_padding = torch.zeros(int(0.5 * 50), device=self.device)
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
                _ = self.opt(
                    sharpness=256, 
                    weight={"design_region_0": -0.05 * torch.ones((2 * self.num_atom + 1), device=self.device).unsqueeze(0)},
                    custom_source=custom_source
                )

                source_field = self.opt.objective.response[('in_slice_1', 'in_slice_1', 0.85, "Hz1", 300)]["fz"].squeeze()
                total_normalizer_list.append(source_field[boolean_source_mask].mean())
        return total_normalizer_list

    def forward(
            self, 
            sharpness, 
        ):
        # Implement the forward pass for the metasurface
        # TODO return the hr transfer matrix
        full_wave_down_sample_rate = 1
        sources = torch.eye(self.num_atom * round(15 // full_wave_down_sample_rate), device=self.device)
        ls_knots = self.build_ls_knots().unsqueeze(0)

        sim_key = list(self.opt.objective.sims.keys())
        assert len(sim_key) == 1, f"there should be only one sim key, but we got {sim_key}"
        if hasattr(self.opt.objective.sims[sim_key[0]].solver, "clear_solver_cache"):
            self.opt.objective.sims[sim_key[0]].solver.clear_solver_cache()
            self.opt.objective.sims[sim_key[0]].solver.set_cache_mode(True)

        full_wave_response = torch.zeros(
            (
                self.num_atom * round(15 // full_wave_down_sample_rate),
                self.num_atom * round(15 // full_wave_down_sample_rate),
            ),
            device=self.device, 
            dtype=torch.complex128
        )
        for idx in range(self.num_atom * round(15 // full_wave_down_sample_rate)):
            source_i = sources[idx].repeat_interleave(full_wave_down_sample_rate)
            source_zero_padding = torch.zeros(int(0.5 * 50), device=self.device)
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
            _ = self.opt(
                sharpness=sharpness, 
                weight={"design_region_0": ls_knots},
                custom_source=custom_source
            )

            response = self.opt.objective.response[('in_slice_1', 'nearfield_1', 0.85, "Hz1", 300)]["fz"].squeeze()
            response = response[full_wave_down_sample_rate // 2 :: full_wave_down_sample_rate]
            assert len(response) == self.num_atom * round(15 // full_wave_down_sample_rate), f"{len(response)}!= {self.num_atom * round(15 // full_wave_down_sample_rate)}"
            full_wave_response[idx] = response
        full_wave_response = full_wave_response.transpose(0, 1)

        normalizer = torch.stack(self.normalizer_list, dim=0).to(self.device)
        normalizer = normalizer.unsqueeze(1)
        full_wave_response = full_wave_response / normalizer

        if hasattr(self.opt.objective.sims[sim_key[0]].solver, "clear_solver_cache"):
            self.opt.objective.sims[sim_key[0]].solver.clear_solver_cache()
            self.opt.objective.sims[sim_key[0]].solver.set_cache_mode(False)
        return full_wave_response

def matching_T(
    num_metasurfaces=2,
    num_atom=32,
    atom_period=0.3,
    dz=4,
    kernel_area=2,
    input_wg_width=0.3,
    input_wg_gap=1,
    out_channles=3,
    target_matrixs=None, # N, H, W
):
    plot_root = './unitest/plot/'
    ensure_dir(plot_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metasurfaces = []
    metasurfaces.append(
        MetaSurface(
            num_atom=num_atom,
            atom_period=atom_period,
            dz=dz,
            plot_root=plot_root,
            device=device,
        )
    )
    for _ in range(num_metasurfaces - 1):
        metasurfaces.append(
            MetaSurface(
                num_atom=num_atom,
                atom_period=atom_period,
                dz=dz,
                normalizer_list=metasurfaces[0].normalizer_list,
                plot_root=plot_root,
                device=device,
            )
        )
    near2far_matrix = probe_near2far_matrix(
        metasurfaces[0].opt,
        dz,
        device,
    ).to(torch.complex128)

    # we need to calculate the source whose shape should be (kernel_area, 480)
    input_sources = build_donn_input_sources(
        num_waveguides=kernel_area,
        resolution=50,
        input_wg_width=input_wg_width,
        input_wg_gap=input_wg_gap,
        total_pixels=480,
        device=device,
    )

    lr = 0.005
    lr_final = lr * 1e-2
    num_epoch = 20
    initial_sharp = 10
    final_sharp = 256
    all_params = []
    for i in range(num_metasurfaces):
        all_params += list(metasurfaces[i].parameters())
    optimizer = torch.optim.Adam(
        all_params, lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_final)
    sharp_scheduler = SharpnessScheduler(
        initial_sharp=initial_sharp, 
        final_sharp=final_sharp, 
        total_steps=num_epoch,
    )
    creterion = ColumnWiseCosSimLoss()
    # next we need to probe the transfer matrix
    for i in range(num_epoch):
        sharpness = sharp_scheduler.get_sharpness()
        optimizer.zero_grad()
        transfer_matrix_list = [metasurface(sharpness) for metasurface in metasurfaces]
        total_transfer_matrix = near2far_matrix
        for j in range(num_metasurfaces):
            total_transfer_matrix = near2far_matrix @ transfer_matrix_list[j] @ total_transfer_matrix
        total_resoponse = torch.matmul(input_sources, total_transfer_matrix.t())
        assert 480 % out_channles == 0, f"480 % {out_channles} != 0"
        total_resoponse = total_resoponse.reshape(
            -1, out_channles, 480 // out_channles
        ) # bs, out_channles, 480 // out_channles
        total_resoponse = ((torch.abs(total_resoponse)) ** 2).sum(dim=-1).t() # out_channles, bs
        total_resoponse = normalize_transfer_matrix(total_resoponse)
        # normalize it column-wise using std
        loss = creterion(target_matrixs, total_resoponse)
        print(f"epoch {i}, loss: {loss.item()}", flush=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        sharp_scheduler.step()

    return loss.item()


if __name__ == "__main__":
    set_torch_deterministic()
    batch_size = 10
    input_wg_width = 0.3
    input_wg_gap = 1
    out_channles = 8
    window_size = 2
    kernel_area = window_size**2
    kernel_area = [2, 3, 4, 5, 6, 7]
    # kernel_area = [2]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cos_sim = {}
    for ka in kernel_area:
        print(f"kernel_area: {ka}", flush=True)
        with DeterministicCtx(41):
            # random generate a batch of transfer matrix whose shape is bs, out_channles, kernel_area
            target_matrixs = torch.rand(
                (batch_size, out_channles, ka), device=device
            )
            for i in range(len(target_matrixs)):
                target_matrixs[i] = normalize_transfer_matrix(target_matrixs[i])
        # inverse design to match a batch of random transfer matrix
        current_cos_sim = []
        for i in range(batch_size):
            print(f"batch {i}", flush=True)
            target_matrix = target_matrixs[i]
            current_cos_sim_loss = matching_T(
                num_metasurfaces=2,
                num_atom=32,
                atom_period=0.3,
                dz=4,
                kernel_area=ka,
                input_wg_width=input_wg_width,
                input_wg_gap=input_wg_gap,
                out_channles=out_channles,
                target_matrixs=target_matrix, # N, H, W
            )
            current_cos_sim.append(current_cos_sim_loss)
        current_cos_sim = numpy.array(current_cos_sim)
        current_cos_sim = current_cos_sim.mean()
        cos_sim[ka] = current_cos_sim
    # store it to csv
    with open(f"unitest/cossim_wgw-{input_wg_width}_wgg-{input_wg_gap}_oC-{out_channles}_swka.csv", mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["kernel_area", "cos_sim"])
        for key, value in cos_sim.items():
            writer.writerow([key, value])