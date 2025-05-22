import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
import numpy as np
import random
import math
from thirdparty.MAPS_old.core.fdfd.fdfd import fdfd_hz
from thirdparty.MAPS_old.core.fdfd.near2far import (
    get_farfields_GreenFunction,
)
from ceviche.constants import *
import copy
import matplotlib.pyplot as plt
import h5py
class DeterministicCtx:
    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.random_state = None
        self.numpy_random_state = None
        self.torch_random_state = None
        self.torch_cuda_random_state = None

    def __enter__(self):
        # Save the current states
        self.random_state = random.getstate()
        self.numpy_random_state = np.random.get_state()
        self.torch_random_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            self.torch_cuda_random_state = torch.cuda.get_rng_state()

        # Set deterministic behavior based on the seed
        set_torch_deterministic(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the saved states
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_random_state)
        torch.random.set_rng_state(self.torch_random_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(self.torch_cuda_random_state)


def set_torch_deterministic(seed: int = 0) -> None:
    seed = int(seed) % (2**32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)

def hidden_register_hook(m, input, output):
    m._recorded_hidden = output


def register_hidden_hooks(model):
    for name, m in model.named_modules():
        if isinstance(m, (nn.ReLU, nn.GELU, nn.Hardswish, nn.ReLU6)):
            m.register_forward_hook(hidden_register_hook)


def get_parameter_group(model, weight_decay=0.0):
    """set weigh_decay to Normalization layers to 0"""
    all_parameters = set(model.parameters())
    group_no_decay = set()
    special_params = set()
    swap_alm_multiplier = set()

    for m in model.modules():
        if isinstance(m, _BatchNorm):
            if m.weight is not None:
                group_no_decay.add(m.weight)
            if m.bias is not None:
                group_no_decay.add(m.bias)
        else:
            for param_name, param in m.named_parameters():
                # print(param_name, param)
                if any(key in param_name for key in ["swap_permutation"]):
                    # print(param_name, param)
                    special_params.add(param)
                if any(key in param_name for key in ["swap_alm_multiplier"]):
                    # print(param_name, param)
                    swap_alm_multiplier.add(param)

                elif hasattr(param, "not_decay") and param.not_decay:
                    # print(param_name, param)
                    group_no_decay.add(param)

    group_decay = all_parameters - group_no_decay - swap_alm_multiplier

    if not special_params:
        print("Not train perm")
        return [
            {"params": list(group_no_decay), "weight_decay": 0.0},
            {"params": list(group_decay), "weight_decay": weight_decay},
        ]
    else:
        print("Train perm")
        return [
            {"params": list(group_no_decay), "weight_decay": 0.0},
            {"params": list(group_decay), "weight_decay": weight_decay},
        ], [{"params": list(special_params), "weight_decay": 0.0}]
    
class EnergyConservationLoss(nn.Module):
    def __init__(self, loss_coeff=1.0):
        super(EnergyConservationLoss, self).__init__()
        self.loss_coeff = loss_coeff

    def forward(self, transfer_matrix):
        # this should be a unitary matrix
        assert transfer_matrix.shape[0] == transfer_matrix.shape[1]
        transfer_matrix_H = transfer_matrix.conj().T
        identity = torch.eye((transfer_matrix.shape[0], transfer_matrix.shape[1]), device=transfer_matrix.device)
        response_matmul = torch.matmul(transfer_matrix, transfer_matrix_H)
        response_matmul = response_matmul / response_matmul.abs().max()
        response_matmul = response_matmul * self.loss_coeff
        loss = torch.norm(torch.matmul(transfer_matrix, transfer_matrix_H) - self.loss * identity, 2)
        return loss
    
# class NL2norm(nn.Module):
#     def __init__(self):
#         super(NL2norm, self).__init__()

#     def forward(self, target, transfer_matrix):
#         # interpolate the target to the same size as the transfer matrix
#         if target.shape != transfer_matrix.shape:
#             target_real = nn.functional.interpolate(target.unsqueeze(1).real, size=transfer_matrix.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
#             target_imag = nn.functional.interpolate(target.unsqueeze(1).imag, size=transfer_matrix.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
#             target = target_real + 1j * target_imag
#         # first normalize the transfer matrix along the batch dimension
#         transfer_matrix = transfer_matrix / torch.max(transfer_matrix.abs(), dim=1, keepdim=True)[0]
#         target = target / torch.max(target.abs(), dim=1, keepdim=True)[0]
#         loss = torch.norm(transfer_matrix - target, 2) / torch.norm(target, 2)

#         return loss
    
class NL2norm(nn.Module):
    def __init__(self):
        super(NL2norm, self).__init__()

    def forward(self, pred, target):
        error = pred - target
        error_energy = torch.norm(error, 2, dim=(-1, -2))
        target_energy = torch.norm(target, 2, dim=(-1, -2))
        return (error_energy / target_energy).mean()

    
class TransferMatrixMatchingLoss(nn.Module):
    def __init__(self):
        super(TransferMatrixMatchingLoss, self).__init__()

    def forward(self, total_response, target_response, target_phase_variants, seperate_loss=True):
        """
        Computes the MSE loss between total_phase and the closest value
        in the three versions of target_phase_shift: original, +2π, and -2π.
        
        Args:
            total_phase (torch.Tensor): Tensor of shape (N,) representing the computed phase.
            target_phase_shift (torch.Tensor): Tensor of shape (N,) representing the target phase shift.
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Compute absolute differences (3, N)
        # target_phase = torch.angle(target_response)
        # if total_response.shape != target_response.shape:
        #     total_response_real = F.interpolate(total_response.real.unsqueeze(0).unsqueeze(0), size=target_response.shape[-2:], mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        #     total_response_imag = F.interpolate(total_response.imag.unsqueeze(0).unsqueeze(0), size=target_response.shape[-2:], mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        #     total_response = total_response_real + 1j * total_response_imag
        W_transfer = total_response.shape[-1]
        W_target = target_response.shape[-1]
        if target_response.shape != total_response.shape:
            # interpolate the target to the same size as the transfer matrix
            target_response_real = F.interpolate(target_response.real.unsqueeze(0).unsqueeze(0), size=total_response.shape[-2:], mode="bilinear", align_corners=False).squeeze()
            target_response_imag = F.interpolate(target_response.imag.unsqueeze(0).unsqueeze(0), size=total_response.shape[-2:], mode="bilinear", align_corners=False).squeeze()
            target_response = target_response_real + 1j * target_response_imag
            ds_rate = W_transfer / W_target
            target_response = target_response / ds_rate

            target_phase_variants = F.interpolate(target_phase_variants.unsqueeze(0), size=total_response.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)

        target_mag = torch.abs(target_response)

        total_phase = torch.angle(total_response)
        total_mag = torch.abs(total_response)
        # begin calculate the phase loss
        abs_diffs = torch.abs(target_phase_variants - total_phase.unsqueeze(0))  # Broadcasting

        # Find the index of the closest match at each point
        closest_indices = torch.argmin(abs_diffs, dim=0)  # Shape (N,)

        # Gather the closest matching values
        # closest_values = target_phase_variants[closest_indices, torch.arange(total_phase.shape[0])]
        rows = torch.arange(closest_indices.shape[0]).unsqueeze(1).expand_as(closest_indices)
        cols = torch.arange(closest_indices.shape[1]).unsqueeze(0).expand_as(closest_indices)
        closest_values = target_phase_variants[closest_indices, rows, cols]


        if not seperate_loss:

            target_response = target_mag * torch.exp(1j * closest_values)

            response_error = target_response - total_response

            response_normalized_L2 = torch.norm(response_error) / (torch.norm(target_response) + 1e-12)

            return response_normalized_L2, round(response_normalized_L2.item(), 3)

        else:

            phase_error = closest_values - total_phase

            weight_map = (target_mag - torch.min(target_mag)) / (torch.max(target_mag) - torch.min(target_mag))

            phase_normalized_L2 = torch.norm(phase_error * weight_map) / (torch.norm(closest_values) + 1e-12)

            # begin calculate the magnitude loss

            mag_error = target_mag - total_mag
            mag_normalized_L2 = torch.norm(mag_error) / (torch.norm(target_mag) + 1e-12)

            print("this is the mag NL2norm: ", mag_normalized_L2.item(), "this is the phase NL2norm: ", phase_normalized_L2.item(), flush=True)
            return mag_normalized_L2 + phase_normalized_L2, round(mag_normalized_L2.item(), 3), round(phase_normalized_L2.item(), 3)


class ResponseMatchingLoss(nn.Module):
    def __init__(
            self, 
            probe_source_mode,
            num_modes,
            num_random_sources=1,
        ):
        super(ResponseMatchingLoss, self).__init__()
        self.probe_source_mode = probe_source_mode
        assert self.probe_source_mode in ["random", "fourier"], f"probe_source_mode should be one of ['random', 'fourier'], but got {self.probe_source_mode}"
        self.num_modes = num_modes
        self.num_random_sources = num_random_sources
        if self.probe_source_mode == "random":
            print(f"we are using random probe source with {self.num_random_sources} random sources", flush=True)

        with h5py.File("./core/fft_weights.h5", "r") as f:
            self.x1_freq_weight = torch.tensor(f["first_layer"], device="cuda").sqrt()
            self.x2_freq_weight = torch.tensor(f["second_layer"], device="cuda").sqrt()

    def forward(self, target, transfer_matrix, layer_number = 0):

        # interpolate the column of the transfer matrix to 480
        # pre_ds_rate = 480 / transfer_matrix.shape[-1]
        # transfer_matrix_real = F.interpolate(transfer_matrix.real.unsqueeze(0).unsqueeze(0), size=(transfer_matrix.shape[-2], 480), mode="bilinear", align_corners=False).squeeze()
        # transfer_matrix_imag = F.interpolate(transfer_matrix.imag.unsqueeze(0).unsqueeze(0), size=(transfer_matrix.shape[-2], 480), mode="bilinear", align_corners=False).squeeze()
        # transfer_matrix = transfer_matrix_real + 1j * transfer_matrix_imag
        # transfer_matrix = transfer_matrix / pre_ds_rate

        H_target = target.shape[-2]
        W_target = target.shape[-1]
        H_transfer = transfer_matrix.shape[-2]
        W_transfer = transfer_matrix.shape[-1]
        if target.shape != transfer_matrix.shape:
            # interpolate the target to the same size as the transfer matrix
            target_real = F.interpolate(target.real.unsqueeze(0).unsqueeze(0), size=transfer_matrix.shape[-2:], mode="bilinear", align_corners=False).squeeze()
            target_imag = F.interpolate(target.imag.unsqueeze(0).unsqueeze(0), size=transfer_matrix.shape[-2:], mode="bilinear", align_corners=False).squeeze()
            target = target_real + 1j * target_imag
            ds_rate = W_transfer / W_target
            target = target / ds_rate

        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # im0 = ax[0].imshow(
        #     target.detach().abs().cpu().numpy()
        # )
        # ax[0].set_title("Target Magnitude")
        # fig.colorbar(im0, ax=ax[0])
        # im1 = ax[1].imshow(
        #     transfer_matrix.detach().abs().cpu().numpy()
        # )
        # ax[1].set_title("Transfer Matrix Magnitude")
        # fig.colorbar(im1, ax=ax[1])
        # plt.savefig("./figs/cpr_itpl_tm_in_responsematching.png", dpi = 300)
        # quit()

        if self.probe_source_mode == "fourier":
            positive_freq = torch.eye(W_transfer, W_transfer, device=target.device)[:self.num_modes] # num_modes, L
            negative_freq = torch.eye(W_transfer, W_transfer, device=target.device)[-self.num_modes:]
            if layer_number == 1:
                # print("we are now use the first layer with the fourier basis", flush=True)
                # print("this is the weight that we will use for the fourier basis: ", self.x1_freq_weight[:self.num_modes], flush=True)
                # print("this is the weight that we will use for the fourier basis: ", self.x1_freq_weight[-self.num_modes:], flush=True)
                positive_freq = positive_freq * (self.x1_freq_weight[:self.num_modes].unsqueeze(1))
                negative_freq = negative_freq * (self.x1_freq_weight[-self.num_modes:].unsqueeze(1))
            elif layer_number == 2:
                # print("we are now use the second layer with the fourier basis", flush=True)
                # print("this is the weight that we will use for the fourier basis: ", self.x2_freq_weight[:self.num_modes], flush=True)
                # print("this is the weight that we will use for the fourier basis: ", self.x2_freq_weight[-self.num_modes:], flush=True)
                positive_freq = positive_freq * (self.x2_freq_weight[:self.num_modes].unsqueeze(1))
                negative_freq = negative_freq * (self.x2_freq_weight[-self.num_modes:].unsqueeze(1))
            elif layer_number == 0:
                # print("we are now not using any additonal weight for the fourier basis", flush=True)
                pass
            else:
                raise ValueError(f"layer_number should be one of [0, 1, 2], but got {layer_number}")
            fourier_basis_target = torch.cat(
                [positive_freq, negative_freq],
                dim=0
            )
            probe_source_target = torch.fft.ifft(fourier_basis_target, dim=-1).to(target.dtype)

            fourier_basis_transfer = torch.cat(
                [
                    torch.eye(W_transfer, W_transfer, device=transfer_matrix.device)[
                        :self.num_modes
                    ],
                    torch.eye(W_transfer, W_transfer, device=transfer_matrix.device)[
                        -self.num_modes:
                    ],
                ],
                dim=0
            )
            probe_source_transfer = torch.fft.ifft(fourier_basis_transfer, dim=-1).to(transfer_matrix.dtype)
        elif self.probe_source_mode == "random":
            raise NotImplementedError("random probe source is deprecated, please use fourier probe source")
            probe_source_target = torch.randn(self.num_random_sources, H_transfer, device=target.device).to(target.dtype)
            probe_source_transfer = torch.randn(self.num_random_sources, H_transfer, device=transfer_matrix.device).to(transfer_matrix.dtype)
        else:
            raise ValueError(f"probe_source_mode should be one of ['random', 'fourier'], but got {self.probe_source_mode}")

        response_target = torch.matmul(probe_source_target, target.T)
        response_transfer = torch.matmul(probe_source_transfer, transfer_matrix.T)

        # response_target_freq = torch.fft.fft(response_target, dim=-1)
        # response_transfer_freq = torch.fft.fft(response_transfer, dim=-1)

        # # Create frequency mask
        # mask = torch.zeros_like(response_target_freq)
        # mask[:, :self.num_modes] = 1
        # mask[:, -self.num_modes:] = 1

        # # Apply mask without in-place modification
        # response_target_freq = response_target_freq * mask
        # response_transfer_freq = response_transfer_freq * mask

        # response_target = torch.fft.ifft(response_target_freq, dim=-1)
        # response_transfer = torch.fft.ifft(response_transfer_freq, dim=-1)
        
        loss = torch.norm(response_transfer - response_target, 2) / torch.norm(response_target, 2)
        return loss
    
class CosSimLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, prediction):
        target_vec = target.flatten()
        pred_vec = prediction.flatten()

        # Inner product
        dot = torch.sum(target_vec * torch.conj(pred_vec))  # works for both real/complex

        # Norms
        norm_target = torch.linalg.norm(target_vec)
        norm_pred = torch.linalg.norm(pred_vec)

        # Cosine similarity
        cos_sim = torch.real(dot / (norm_target * norm_pred + 1e-8))  # add eps to avoid div/0

        return cos_sim
    
class HighFreqPenalty(torch.nn.Module):
    def __init__(self, mode_threshold) -> None:
        super().__init__()
        self.mode_threshold = mode_threshold

    def forward(self, inner_field):
        inner_field = inner_field.flatten(0, 1)
        spectrum = torch.fft.fft(inner_field, dim=-1)
        high_freq_percentage = spectrum[
            :,
            1 + self.mode_threshold : -self.mode_threshold,
        ].abs().square().sum(dim=-1) / spectrum.abs().square().sum(dim=-1)
        high_freq_percentage = high_freq_percentage.mean()

        return high_freq_percentage

class AdmmConsistencyLoss(nn.Module):
    def __init__(self, rho: float = 1.0):
        super(AdmmConsistencyLoss, self).__init__()
        self.rho = rho

    def forward(
        self, 
        x: torch.Tensor,               # shape: [N, H, W]
        admm_vars,
    ) -> torch.Tensor:
        """
        Computes ADMM consistency loss:
        (rho / 2) * || x - stack(z) + stack(u) ||^2

        Returns:
            torch.Tensor: scalar loss value
        """
        z = admm_vars["z_admm"]
        u = admm_vars["u_admm"]
        assert x.ndim == 3, f"x must have shape [N, H, W], got {x.shape}"
        assert len(z) == x.shape[0], f"z mismatch: len(z)={len(z)} vs x.shape[0]={x.shape[0]}"
        assert len(u) == x.shape[0], f"u mismatch: len(u)={len(u)} vs x.shape[0]={x.shape[0]}"

        z_tensor = torch.stack(z, dim=0)  # shape: [N, H, W]
        u_tensor = torch.stack(u, dim=0)  # shape: [N, H, W]
        diff = x - z_tensor + u_tensor    # shape: [N, H, W]
        loss = torch.norm(diff) ** 2      # scalar

        return self.rho / 2.0 * loss


class TransferMatrixSmoothError(nn.Module):
    def __init__(self, mode: str = "xy"):
        """
        mode can be one of the following:
          - "xy": diff along x and y
          - "diag": diff along diagonal
          - "both": diff along x, y, and diagonal
        """
        super(TransferMatrixSmoothError, self).__init__()
        self.mode = mode

    def forward(self, transfer_matrix: torch.Tensor) -> torch.Tensor:
        """
        # the transfer matrix is of shape bs, H, W
        # it is a complex matrix
        """
        # 1) calculate the difference along x, y, and diagonal
        # along x 
        dx = transfer_matrix[..., :, 1:] - transfer_matrix[..., :, :-1]  # shape (bs, H, W-1)
        # along y
        dy = transfer_matrix[..., 1:, :] - transfer_matrix[..., :-1, :]  # shape (bs, H-1, W)
        # along diagonal
        d_diag = transfer_matrix[..., 1:, 1:] - transfer_matrix[..., :-1, :-1]  # shape (bs, H-1, W-1)

        # 2) small function to calculate the amplitude square
        def amp_sqr(z: torch.Tensor) -> torch.Tensor:
            return z.real**2 + z.imag**2

        # 3) calculate the loss
        loss_x = amp_sqr(dx).mean()
        loss_y = amp_sqr(dy).mean()
        loss_diag = amp_sqr(d_diag).mean()

        # 4) return the loss
        if self.mode == "xy":
            return loss_x + loss_y
        elif self.mode == "diag":
            return loss_diag
        elif self.mode == "both":
            return loss_x + loss_y + loss_diag
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
    
    
class DownsampleRateScheduler(object):
    __mode_list__ = {"cosine", "quadratic", "milestone", "constant"}
    def __init__(
            self, 
            total_steps: int, 
            init_ds_rate: int, 
            final_ds_rate: int, 
            available_ds_rate: list, 
            mode:str="cosine", 
            milestone: list=None,
        ):
        super().__init__()
        self.init_ds_rate = init_ds_rate
        self.final_ds_rate = final_ds_rate
        self.available_ds_rate = available_ds_rate
        self.available_ds_rate.sort(reverse=True)
        self.milestone = milestone
        self.total_steps = total_steps
        self.current_step = 0
        self.current_ds = init_ds_rate
        self.mode = mode
        assert mode in self.__mode_list__, f"mode should be one of {self.__mode_list__}, but got {mode}"
    def _step_cosine(self):
        self.current_step += 1
        if self.current_step > self.total_steps:
            self.current_step = self.total_steps
        cos_inner = (math.pi * self.current_step) / self.total_steps
        cos_out = math.cos(cos_inner) + 1  # Change the sign to reverse the trend
        self.current_ds = (
            self.final_ds_rate + (self.init_ds_rate - self.final_ds_rate) * (cos_out / 2)**1
        )
        # find the closest available ds rate in the list
        self.current_ds = min(self.available_ds_rate, key=lambda x: abs(x - self.current_ds))
        return self.current_ds
    
    def _step_milestone(self):
        self.current_step += 1
        if self.current_step == 1:
            self.ds_idx = 0
        if self.ds_idx < len(self.milestone):
            if self.current_step > self.milestone[self.ds_idx]:
                self.ds_idx += 1
            self.current_ds = self.available_ds_rate[self.ds_idx]
            return self.current_ds
        else:
            self.current_ds = self.available_ds_rate[-1]
            return self.current_ds
        
    def _step_constant(self):
        self.current_ds = self.init_ds_rate
        return self.current_ds
    
    def _step_quadratic(self):
        raise NotImplementedError("Quadratic mode is not implemented yet")
        self.current_step += 1
        if self.current_step > self.total_steps:
            self.current_step = self.total_steps
        self.current_sharp = (
            self.initial_sharp + (self.final_sharp - self.initial_sharp) * (self.current_step / self.total_steps)**2
        )
        return self.current_sharp
    
    def step(self):
        if self.mode == "cosine":
            return self._step_cosine()
        elif self.mode == "quadratic":
            return self._step_quadratic()
        elif self.mode == "milestone":
            return self._step_milestone()
        elif self.mode == "constant":
            return self._step_constant()
        else:
            raise ValueError(f"mode should be one of {self.__mode_list__}, but got {self.mode}")

    def get_downsample_rate(self):
        return self.current_ds
    
class End2EndSharpnessScheduler(object):
    def __init__(
            self,
            mode,
            num_train_epochs,
            init_sharpness=10,
            final_sharpness=256,
        ) -> None:
        self.mode = mode.lower()
        assert self.mode == "cosine", f"mode should be cosine, but got {self.mode}"
        self.num_train_epochs = num_train_epochs
        self.init_sharpness = init_sharpness
        self.final_sharpness = final_sharpness
        self.current_sharpness = init_sharpness
        self.current_step = 0

    def step(self):
        """Take a scheduler step and return current sharpness value"""
        self.current_step += 1
        if self.current_step > self.num_train_epochs:
            self.current_step = self.num_train_epochs
            
        cos_inner = math.pi * self.current_step / self.num_train_epochs  # goes from 0 to pi
        factor = (1 - math.cos(cos_inner)) / 2  # cosine increasing from 0 to 1
        self.current_sharpness = self.init_sharpness + factor * (self.final_sharpness - self.init_sharpness)
        
        return self.current_sharpness

    def get_sharpness(self):
        """Get current sharpness value"""
        return self.current_sharpness

    
class InvDesSharpnessScheduler(object):
    def __init__(
            self, 
            mode,
            num_train_epochs,
            sharpness_peak_epoch,
            sharpness_span_per_epoch=128,
            init_sharpness=10, 
            final_sharpness=256,
        ) -> None:
        self.mode = mode
        assert self.mode in ["per_proj", "per_epoch", "per_training", "mixed"], f"mode should be one of [per_proj, per_epoch, per_training, mixed], but got {self.mode}"
        self.num_train_epochs = num_train_epochs
        self.sharpness_peak_epoch = sharpness_peak_epoch
        assert self.sharpness_peak_epoch < self.num_train_epochs, f"sharpness peak epoch should be less than num train epochs, but got {self.sharpness_peak_epoch} >= {self.num_train_epochs}"
        self.sharpness_span_per_epoch = sharpness_span_per_epoch
        self.init_sharpness = init_sharpness
        self.final_sharpness = final_sharpness
        self.current_sharpness = init_sharpness
        self.current_step = 0

        self.build_sharpness_schedule()

    def build_sharpness_schedule(self):
        if self.mode == "per_proj":
            self.sharpness_schedule = None # there is no schedule for per_proj
        elif self.mode == "per_epoch":
            self.sharpness_schedule = None # there is no schedule for per_epoch
        elif self.mode == "per_training":
            schedule = []
            for t in range(self.num_train_epochs + 1):
                cos_inner = math.pi * t / self.num_train_epochs  # goes from 0 to pi
                factor = (1 - math.cos(cos_inner)) / 2  # cosine increasing from 0 to 1
                sharpness = self.init_sharpness + factor * (self.final_sharpness - self.init_sharpness)
                schedule.append(sharpness)
            self.sharpness_schedule = schedule
        elif self.mode == "mixed":
            self.lower_sharpness = torch.linspace(
                self.init_sharpness,
                self.final_sharpness - self.sharpness_span_per_epoch,
                self.sharpness_peak_epoch,
            )
            self.upper_sharpness = torch.linspace(
                self.init_sharpness + self.sharpness_span_per_epoch,
                self.final_sharpness,
                self.sharpness_peak_epoch,
            )
        

    def step(self):
        if self.mode == "per_proj":
            return self._step_per_proj()
        elif self.mode == "per_epoch":
            return self._step_per_epoch()
        elif self.mode == "per_training":
            return self._step_per_training()
        elif self.mode == "mixed":
            return self._step_mixed()
        
    def _step_per_proj(self):
        self.current_sharpness = (
            self.init_sharpness,
            self.final_sharpness,
        )
        return self.current_sharpness
    
    def _step_per_epoch(self):
        self.current_sharpness = (
            self.init_sharpness,
            self.final_sharpness,
        )
        return self.current_sharpness
    
    def _step_per_training(self):
        self.current_sharpness = (
            self.sharpness_schedule[self.current_step],
            self.sharpness_schedule[self.current_step + 1],
        )
        self.current_step += 1
        return self.current_sharpness

    def _step_mixed(self):
        if self.current_step < self.sharpness_peak_epoch:
            self.current_sharpness = (
                self.lower_sharpness[self.current_step],
                self.upper_sharpness[self.current_step],
            )
        else:
            self.current_sharpness = (
                self.lower_sharpness[self.sharpness_peak_epoch - 1],
                self.upper_sharpness[self.sharpness_peak_epoch - 1],
            )
        self.current_step += 1
        return self.current_sharpness

    def get_current_sharpness(self):
        return self.current_sharpness

    
def probe_near2far_matrix(
        total_opt, 
        wl, # in um
        device,
        normalize=True,
    ):
    size_y = round(total_opt.device.aperture * total_opt.sim_cfg["resolution"])
    probe_source = torch.eye(size_y, size_y, device=device)
    probe_source = probe_source.to(torch.complex64)
    eps = torch.ones((1, size_y), device=device)
    tiny_fdfd = fdfd_hz(
        omega=2 * np.pi * C_0 / (wl * 1e-6),
        dL=1 / total_opt.sim_cfg["resolution"] * 1e-6,
        eps_r=eps,
        npml=[0, 0],
        power=1e-8,
        bloch_phases=None,
        neural_solver=None,
        numerical_solver="solve_direct",
        use_autodiff=False,
        sym_precond=True,
    )
    eps_vec = eps.flatten()
    entries_a, indices_a, eps_matrix, eps_vec_xx, eps_vec_yy = tiny_fdfd._make_A(eps_vec)

    Hz_vec = probe_source # bs (size_y) * size_y
    Ex_vec, Ey_vec = tiny_fdfd._Hz_to_Ex_Ey(Hz_vec, eps_vec_xx, eps_vec_yy)

    nearfield_response_fz = Hz_vec
    nearfield_response_fx = Ex_vec
    nearfield_response_fy = Ey_vec

    extended_farfield_slice_info = copy.deepcopy(total_opt.objective.port_slices_info["farfield_1"])
    xs = extended_farfield_slice_info["xs"]
    grid_step = 1 / total_opt.sim_cfg["resolution"]
    if not xs.shape:
        extended_farfield_slice_info["xs"] = np.array(
            [xs, xs + grid_step]
        )
    else:
        extended_farfield_slice_info["xs"] = np.concatenate(
            [xs, xs[-1:] + grid_step],
            axis=0,
        )
    farfield = get_farfields_GreenFunction(
        nearfield_slices=[
            total_opt.objective.port_slices["nearfield_1"]
        ],
        nearfield_slices_info=[
            total_opt.objective.port_slices_info["nearfield_1"]
        ],
        Fz=nearfield_response_fz.unsqueeze(-1),
        Fx=nearfield_response_fx.unsqueeze(-1),
        Fy=nearfield_response_fy.unsqueeze(-1),
        farfield_x=None,
        farfield_slice_info=total_opt.objective.port_slices_info["farfield_1"],
        freqs=torch.tensor([1 / wl], device=nearfield_response_fz.device),
        eps=total_opt.objective.eps_bg,
        mu=MU_0,
        dL=total_opt.objective.grid_step,
        component="Hz",
        decimation_factor=1,
        passing_slice=True,
    )
    farfield_fz = farfield["Hz"][..., 0]
    farfield_response = farfield_fz[:, 0, :] # torch.Size([1000, 480])
    farfield_response = torch.transpose(farfield_response, 0, 1)
    if normalize:
        farfield_response = farfield_response / torch.max(farfield_response.abs())
    # plt.figure()
    # plt.imshow(torch.abs(farfield_response).cpu().numpy())
    # plt.colorbar()
    # plt.savefig("./figs/near2far_tm.png")
    # quit()
    return farfield_response

def draw_diffraction_region(
        total_opt, 
        wl, # in um
        source,
        device,
    ):
    size_y = round(total_opt.device.aperture * total_opt.sim_cfg["resolution"])
    source = source.unsqueeze(0)
    eps = torch.ones((1, size_y), device=device)
    tiny_fdfd = fdfd_hz(
        omega=2 * np.pi * C_0 / (wl * 1e-6),
        dL=1 / total_opt.sim_cfg["resolution"] * 1e-6,
        eps_r=eps,
        npml=[0, 0],
        power=1e-8,
        bloch_phases=None,
        neural_solver=None,
        numerical_solver="solve_direct",
        use_autodiff=False,
        sym_precond=True,
    )
    eps_vec = eps.flatten()
    entries_a, indices_a, eps_matrix, eps_vec_xx, eps_vec_yy = tiny_fdfd._make_A(eps_vec)

    Hz_vec = source # bs (size_y) * size_y
    Ex_vec, Ey_vec = tiny_fdfd._Hz_to_Ex_Ey(Hz_vec, eps_vec_xx, eps_vec_yy)

    nearfield_response_fz = Hz_vec
    nearfield_response_fx = Ex_vec
    nearfield_response_fy = Ey_vec

    # extended_farfield_slice_info = copy.deepcopy(total_opt.objective.port_slices_info["farfield_1"])
    # xs = extended_farfield_slice_info["xs"]
    # grid_step = 1 / total_opt.sim_cfg["resolution"]
    # if not xs.shape:
    #     extended_farfield_slice_info["xs"] = np.array(
    #         [xs, xs + grid_step]
    #     )
    # else:
    #     extended_farfield_slice_info["xs"] = np.concatenate(
    #         [xs, xs[-1:] + grid_step],
    #         axis=0,
    #     )
    farfield = get_farfields_GreenFunction(
        nearfield_slices=[
            total_opt.objective.port_slices["nearfield_1"]
        ],
        nearfield_slices_info=[
            total_opt.objective.port_slices_info["nearfield_1"]
        ],
        Fz=nearfield_response_fz.unsqueeze(-1),
        Fx=nearfield_response_fx.unsqueeze(-1),
        Fy=nearfield_response_fy.unsqueeze(-1),
        farfield_x=None,
        farfield_slice_info=total_opt.objective.port_slices_info["farfield_1"],
        freqs=torch.tensor([1 / wl], device=nearfield_response_fz.device),
        eps=total_opt.objective.eps_bg,
        mu=MU_0,
        dL=total_opt.objective.grid_step,
        component="Hz",
        decimation_factor=1,
        passing_slice=True,
    )
    farfield_fz = farfield["Hz"][..., 0]
    return farfield_fz.squeeze().T
    # farfield_response = farfield_fz[:, 0, :] # torch.Size([1000, 480])
    # farfield_response = torch.transpose(farfield_response, 0, 1)
    # print(farfield_response.shape, flush=True)
    # return farfield_response

def get_mid_weight(l, w, period=0.3):
    return (w*l)/(period-w)

def get_terminal_weight(l, height, max_height = 0.75):
    return l * (height - max_height) / (height)

# def reset_optimizer_and_scheduler(model, lr_init, lr_final, num_epoch):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=num_epoch, eta_min=lr_final
#     )
#     return optimizer, scheduler

def reset_optimizer_and_scheduler(model, lr_init, lr_final, num_epoch):
    if isinstance(model, nn.Module):
        params = model.parameters()
    elif isinstance(model, list) and all(isinstance(m, nn.Module) for m in model):
        params = []
        for m in model:
            params += list(m.parameters())
    else:
        raise ValueError(f"model should be an nn.Module or a list of nn.Module, but got {type(model)}")

    optimizer = torch.optim.Adam(params, lr=lr_init)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_final)
    return optimizer, scheduler

def insert_zeros_after_every_N_except_last(x: torch.Tensor, N: int, M: int):
    """
    Insert M zeros (or Falses) after every N elements along the last dimension,
    except after the last group.
    
    x: shape (BS, 1, NL), where NL = N * L
    returns: shape (BS, 1, NL + (L - 1) * M)
    """
    x = torch.repeat_interleave(x, N, dim=-1)  # (BS, 1, N*NL)
    BS, C, NL = x.shape
    assert NL % N == 0, "Last dimension must be divisible by N"
    L = NL // N  # number of groups

    x_groups = x.view(BS, C, L, N)  # shape: (BS, 1, L, N)

    # Create fill values depending on dtype
    fill_value = 0
    if x.dtype == torch.bool:
        fill_value = False

    # Zeros or Falses to insert
    padding = torch.full((BS, C, L - 1, M), fill_value, dtype=x.dtype, device=x.device)

    # Interleave x_groups[:-1] with padding
    x_with_zeros = torch.cat([x_groups[:, :, :-1], padding], dim=-1)  # shape: (BS, 1, L-1, N+M)
    x_with_zeros = x_with_zeros.reshape(BS, C, -1)

    # Append last group without padding
    last_group = x_groups[:, :, -1, :]  # (BS, 1, N)
    x_final = torch.cat([x_with_zeros, last_group], dim=-1)

    return x_final