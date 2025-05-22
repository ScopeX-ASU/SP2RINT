from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Parameter, init
from torch.types import Device

from .utils import mzi_out_diff_to_phase, mzi_phase_to_out_diff, partition_chunks

__all__ = ["ONNBaseLayer"]


class ONNBaseLayer(nn.Module):
    def __init__(self, *args, device: Device = torch.device("cpu"), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # cuda or cpu, defaults to cpu
        self.device = device

    def build_parameters(self, mode: str = "weight") -> None:
        ## weight mode
        phase = torch.empty(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock,
            device=self.device,
        )
        weight = torch.empty(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock,
            device=self.device,
        )
        # TIA gain
        S_scale = torch.ones(
            size=list(weight.shape[:-2]) + [1], device=self.device, dtype=torch.float32
        )

        if mode == "weight":
            self.weight = Parameter(weight)
        elif mode == "phase":
            self.phase = Parameter(phase)
            self.S_scale = Parameter(S_scale)
        else:
            raise NotImplementedError

        for p_name, p in {
            "weight": weight,
            "phase": phase,
            "S_scale": S_scale,
        }.items():
            if not hasattr(self, p_name):
                self.register_buffer(p_name, p)

    def reset_parameters(self, mode=None) -> None:
        mode = mode or self.mode
        if mode in {"weight"}:
            # init.kaiming_normal_(self.weight.data)
            if hasattr(self, "kernel_size"):  # for conv2d
                weight = nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                    bias=False,
                ).weight.data
                weight = weight.flatten(1)
                in_channels_pad = self.in_channels_pad - weight.shape[1]
                out_channels_pad = self.out_channels_pad - weight.shape[0]
            elif hasattr(self, "in_features"):  # for linear
                weight = nn.Linear(
                    self.in_features, self.out_features, bias=False
                ).weight.data
                in_channels_pad = self.in_features_pad - weight.shape[1]
                out_channels_pad = self.out_features_pad - weight.shape[0]
            weight = torch.nn.functional.pad(
                weight,
                (0, in_channels_pad, 0, out_channels_pad),
                mode="constant",
                value=0,
            )
            self.weight.data.copy_(
                partition_chunks(weight, out_shape=self.weight.shape).to(
                    self.weight.device
                )
            )

        elif mode in {"phase"}:
            self.reset_parameters(mode="weight")
            scale = self.weight.data.abs().flatten(-2, -1).max(dim=-1, keepdim=True)[0]
            self.S_scale.data.copy_(scale)
            self.phase.data.copy_(
                mzi_out_diff_to_phase(self.weight.data.div(scale[..., None]))
            )
        else:
            raise NotImplementedError

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    @classmethod
    def from_layer(cls, layer: nn.Module, *args, **kwargs) -> nn.Module:
        raise NotImplementedError

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_fast_forward(self) -> None:
        self.fast_forward_flag = True

    def disable_fast_forward(self) -> None:
        self.fast_forward_flag = False

    def set_phase_variation(self, flag: bool = False) -> None:
        self._enable_phase_variation = flag

    def set_weight_noise(self, noise_std: float = 0.0) -> None:
        self.weight_noise_std = noise_std

    def set_output_noise(self, noise_std: float = 0.0) -> None:
        self.output_noise_std = noise_std

    def set_gamma_noise(
        self, noise_std: float, random_state: Optional[int] = None
    ) -> None:
        self.gamma_noise_std = noise_std

    # crosstalk changes
    def set_crosstalk_noise(self, flag: bool = False) -> None:
        self._enable_crosstalk = flag

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        self.phase_quantizer.set_bit(w_bit)
        self.weight_quantizer.set_bit(w_bit)

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        self.input_quantizer.set_bit(in_bit)

    def load_parameters(self, param_dict: Dict[str, Any]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        if self.mode == "phase":
            self.build_weight(update_list=param_dict)

    def switch_mode_to(self, mode: str) -> None:
        self.mode = mode

    def set_enable_ste(self, enable_ste: bool) -> None:
        self._enable_ste = enable_ste

    def set_noise_flag(self, noise_flag: bool) -> None:
        self._noise_flag = noise_flag

    def build_weight_from_phase(self, phases: Tensor) -> Tensor:
        ## inplace operation: not differentiable operation using copy_
        self.weight.data.copy_(
            mzi_phase_to_out_diff(phases).mul(self.S_scale.data[..., None])
        )
        return self.weight

    def build_phase_from_weight_(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        ## inplace operation: not differentiable operation using copy_
        phase, S_scale = self.build_phase_from_weight(weight)
        self.phase.data.copy_(phase)
        self.S_scale.data.copy_(S_scale)
        return self.phase, self.S_scale

    def build_phase_from_weight(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        ## inplace operation: not differentiable operation using copy_
        S_scale = (
            weight.data.abs().flatten(-2, -1).max(dim=-1, keepdim=True)[0]
        )  # block-wise abs_max as scale factor

        weight = torch.where(
            S_scale[..., None] > 1e-8,
            weight.data.div(S_scale[..., None]),
            torch.zeros_like(weight.data),
        )
        phase = mzi_out_diff_to_phase(weight)
        return phase, S_scale

    def sync_parameters(self, src: str = "weight") -> None:
        """
        description: synchronize all parameters from the source parameters
        """
        if src == "weight":
            self.build_phase_from_weight_(self.weight)
        elif src == "phase":
            self.build_weight_from_phase(self.phase)
        else:
            raise NotImplementedError

    def print_parameters(self):
        print(self.phase) if self.mode == "phase" else print(self.weight)

    def build_weight(
        self,
        weight=None,
        enable_noise: bool = True,
        enable_ste: bool = False,
    ) -> Tensor:
        if self.mode == "weight":
            weight = weight if weight is not None else self.weight
            if self.w_bit < 16:
                weight = self.weight_quantizer(weight)
        else:
            raise NotImplementedError
        if self.weight_noise_std > 1e-6:
            weight = weight * (1 + torch.randn_like(weight) * self.weight_noise_std)
        return weight

    def forward(self, x):
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ""
