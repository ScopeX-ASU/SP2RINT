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
    get_mid_weight,
    TransferMatrixMatchingLoss,
)
sys.path.pop()
from pyutils.torch_train import (
    set_torch_deterministic,
)
import csv
import numpy
import h5py

class MetaSurface(nn.Module):
    def __init__(
            self, 
            num_atom, 
            atom_period, 
            dz,
            bundle_number=1,
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
        self.bundle_number = bundle_number
        assert self.num_atom % self.bundle_number == 0, f"num_atom {self.num_atom} should be divisible by bundle_number {self.bundle_number}"

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
        hr_total_metalens = total_metalens.copy(resolution=100)
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
    
    def build_ls_knots(self, custom_ls_knots=None):
        ls_knots = -0.05 * torch.ones((2 * self.num_atom + 1), device=self.device)
        if custom_ls_knots is not None:
            weights = custom_ls_knots.reshape(self.num_atom // self.bundle_number, self.bundle_number)
        else:
            print("we are now using self.weights to build the ls_knots", flush=True)
            weights = self.weights.reshape(self.num_atom // self.bundle_number, self.bundle_number)
        weights = torch.mean(weights, dim=1).squeeze()
        weights = weights.repeat_interleave(self.bundle_number)
        ls_knots[1::2] = weights

        return ls_knots
    
    def calculate_normalizer(self):
        sim_key = list(self.opt.objective.sims.keys())
        assert len(sim_key) == 1, f"there should be only one sim key, but we got {sim_key}"
        if hasattr(self.opt.objective.sims[sim_key[0]].solver, "clear_solver_cache"):
            self.opt.objective.sims[sim_key[0]].solver.clear_solver_cache()
            self.opt.objective.sims[sim_key[0]].solver.set_cache_mode(True)

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

        if hasattr(self.opt.objective.sims[sim_key[0]].solver, "clear_solver_cache"):
            self.opt.objective.sims[sim_key[0]].solver.clear_solver_cache()
            self.opt.objective.sims[sim_key[0]].solver.set_cache_mode(False)
        return total_normalizer_list

    def forward(
            self, 
            sharpness, 
            custom_ls_knots=None,
        ):
        # Implement the forward pass for the metasurface
        # TODO return the hr transfer matrix
        full_wave_down_sample_rate = 1
        sources = torch.eye(self.num_atom * round(15 // full_wave_down_sample_rate), device=self.device)
        ls_knots = self.build_ls_knots(custom_ls_knots).unsqueeze(0)

        sim_key = list(self.opt.objective.sims.keys())
        assert len(sim_key) == 1, f"there should be only one sim key, but we got {sim_key}"
        if hasattr(self.opt.objective.sims[sim_key[0]].solver, "clear_solver_cache"):
            self.opt.objective.sims[sim_key[0]].solver.clear_solver_cache()
            self.opt.objective.sims[sim_key[0]].solver.set_cache_mode(False)

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
    target_matrixs=None, # N, H, W
    bundle_number=1,
    near2far_matrix=None,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
):
    plot_root = './unitest/plot/'
    ensure_dir(plot_root)
    metasurfaces = []
    metasurfaces.append(
        MetaSurface(
            num_atom=num_atom,
            atom_period=atom_period,
            dz=dz,
            bundle_number=bundle_number,
            plot_root=plot_root,
            device=device,
        )
    )
    if num_metasurfaces > 1:
        for _ in range(num_metasurfaces - 1):
            metasurfaces.append(
                MetaSurface(
                    num_atom=num_atom,
                    atom_period=atom_period,
                    dz=dz,
                    bundle_number=bundle_number,
                    normalizer_list=metasurfaces[0].normalizer_list,
                    plot_root=plot_root,
                    device=device,
                )
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
    creterion = TransferMatrixMatchingLoss()
    target_matrix_phase = torch.angle(target_matrixs)
    target_matrix_phase_variants = torch.stack([
        target_matrix_phase,         # Original
        target_matrix_phase + 2 * torch.pi,
        target_matrix_phase - 2 * torch.pi,
    ], dim=0)
    # next we need to probe the transfer matrix
    for i in range(num_epoch):
        sharpness = sharp_scheduler.get_sharpness()
        optimizer.zero_grad()
        transfer_matrix_list = []
        for metasurface in metasurfaces:
            transfer_matrix_list.append(
                metasurface(
                    sharpness=sharpness,
                    custom_ls_knots=None,
                )
            )
        if len(transfer_matrix_list) == 1:
            total_transfer_matrix = transfer_matrix_list[0]
        else:
            total_transfer_matrix = near2far_matrix
            for j in range(num_metasurfaces):
                total_transfer_matrix = near2far_matrix @ transfer_matrix_list[j] @ total_transfer_matrix

        loss = creterion(
            total_response=total_transfer_matrix, 
            target_response=target_matrixs, 
            target_phase_variants=target_matrix_phase_variants, 
            seperate_loss=False,
        )[0]
        print(f"epoch {i}, loss: {loss.item()}", flush=True)
        loss.backward()
        for p in all_params:
            print("this is the parameter: ", p, flush=True)
            print(f"this is the grad of p before step: {p.grad}", flush=True)
        optimizer.step()
        for p in all_params:
            print(f"this is the grad of p after step: {p.grad}", flush=True)
        scheduler.step()
        sharp_scheduler.step()

    # delete the metasurfaces
    for metasurface in metasurfaces:
        del metasurface
    return loss.item()


if __name__ == "__main__":
    set_torch_deterministic()
    batch_size = 10
    bundle_number_list = [1, 2, 4, 8, 16, 32]
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # need to first generate random implementable transfer matrix
    random_target_matrix_path = f"unitest/random_target_matrix.h5"
    if os.path.exists(random_target_matrix_path):
        with h5py.File(random_target_matrix_path, 'r') as f:
            transfer_matrix = f['transfer_matrix'][:]
            transfer_matrix = torch.from_numpy(transfer_matrix).to(device)
            near2far_matrix = f['near2far_matrix'][:]
            near2far_matrix = torch.from_numpy(near2far_matrix).to(device)
    else: 
        transfer_matrix = None
        near2far_matrix = None
    if transfer_matrix is not None and len(transfer_matrix) == batch_size:
        print("load transfer matrix from file", flush=True)
    else:
        with DeterministicCtx(41):
            ls_knots = 0.27 * torch.rand(batch_size, 32, device=device) + 0.01
            ls_knots = get_mid_weight(0.05, ls_knots)
            surface_calculator = MetaSurface(
                num_atom=32,
                atom_period=0.3,
                dz=4,
                bundle_number=1,
                normalizer_list=None,
                plot_root='./unitest/plot/',
                device=device,
            )
            with torch.no_grad():
                transfer_matrix = []
                for i in range(len(ls_knots)):
                    print(f"simulating {i} target metasurface...", flush=True)
                    ls_knots_i = ls_knots[i]
                    transfer_matrix_i = surface_calculator(
                        sharpness=256,
                        custom_ls_knots=ls_knots_i,
                    )
                    transfer_matrix.append(transfer_matrix_i)
                transfer_matrix = torch.stack(transfer_matrix, dim=0)
                near2far_matrix = probe_near2far_matrix(
                    surface_calculator.opt,
                    0.85,
                    device,
                ).to(torch.complex128)
                with h5py.File(random_target_matrix_path, 'w') as f:
                    f.create_dataset('transfer_matrix', data=transfer_matrix.cpu().numpy())
                    f.create_dataset('near2far_matrix', data=near2far_matrix.cpu().numpy())

    # total_transfer_matrix_list = []
    # for i in range(batch_size):
    #     total_transfer_matrix = near2far_matrix
    #     for j in range(2):
    #         total_transfer_matrix = near2far_matrix @ transfer_matrix[i * 2 + j] @ total_transfer_matrix
    #     total_transfer_matrix_list.append(total_transfer_matrix)
    # total_transfer_matrix = torch.stack(total_transfer_matrix_list, dim=0)
    total_transfer_matrix = transfer_matrix

    print("total transfer matrix shape: ", total_transfer_matrix.shape, flush=True)

    n_L2norm_loss = {}
    for bundle_number in bundle_number_list:
        print(f"bundle_number: {bundle_number}", flush=True)

        current_n_L2norm = []
        for i in range(batch_size):
            print(f"batch {i}", flush=True)
            target_matrix = total_transfer_matrix[i]
            current_n_L2norm_loss = matching_T(
                num_metasurfaces=1,
                num_atom=32,
                atom_period=0.3,
                dz=4,
                bundle_number=bundle_number,
                target_matrixs=target_matrix, # N, H, W
                near2far_matrix=near2far_matrix,
                device=device,
            )
            current_n_L2norm.append(current_n_L2norm_loss)
        current_n_L2norm = numpy.array(current_n_L2norm).mean()
        n_L2norm_loss[bundle_number] = current_n_L2norm
    # store it to csv
    csv_path = os.path.join('./unitest/', f"bundle_number.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["bundle_number", "loss"])
        for bundle_number, loss in n_L2norm_loss.items():
            writer.writerow([bundle_number, loss])