import inspect
from typing import Callable, Dict, Optional, Union

import torch
from mmcv.cnn.bricks import build_activation_layer, build_conv_layer, build_norm_layer
from mmengine.registry import MODELS
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor, nn  # , set_deterministic
from torch.types import Device, _size
from torchonn.op.mrr_op import *  # noqa: F403

from core.models.layers import MetaParams

__all__ = [
    "LinearBlock",
    "ConvBlock",
    "ConvBlockPTC",
    "Meta_Base",
]

# MODELS.register_module(name="Linear", module=nn.Linear)


def build_linear_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    if cfg is None:
        cfg_ = dict(type="Linear")
    else:
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dict")
        if "type" not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if inspect.isclass(layer_type):
        return layer_type(*args, **kwargs, **cfg_)  # type: ignore
    # Switch registry to the target scope. If `linear_layer` cannot be found
    # in the registry, fallback to search `linear_layer` in the
    # mmengine.MODELS.
    with MODELS.switch_scope_and_registry(None) as registry:
        linear_layer = registry.get(layer_type)
    if linear_layer is None:
        raise KeyError(
            f"Cannot find {linear_layer} in registry under scope "
            f"name {registry.scope}"
        )
    layer = linear_layer(*args, **kwargs, **cfg_)

    return layer


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        linear_cfg: dict = dict(type="Linear"),
        norm_cfg: dict | None = None,
        act_cfg: dict | None = dict(type="ReLU", inplace=True),
        dropout: float = 0.0,
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False) if dropout > 0 else None
        if linear_cfg["type"] not in {"Linear", None}:
            linear_cfg.update({"device": device})
        self.linear = build_linear_layer(
            linear_cfg,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, out_features)[1]
        else:
            self.norm = None

        if act_cfg is not None:
            self.activation = build_activation_layer(act_cfg)
        else:
            self.activation = None

    def forward(self, x: Tensor) -> Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvBlockPTC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Union[int, _size] = 1,
        padding: Union[int, _size] = 0,
        dilation: Union[int, _size] = 1,
        groups: int = 1,
        bias: bool = False,
        conv_cfg: dict = dict(type="PTCBlockConv2d"),
        norm_cfg: dict | None = dict(type="BN", affine=True),
        act_cfg: dict | None = dict(type="ReLU", inplace=True),
        device: Device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ) -> None:
        super().__init__()
        conv_cfg = conv_cfg.copy()
        if conv_cfg["type"] not in {"Conv2d", None}:
            conv_cfg.update({"device": device})
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if norm_cfg is not None:
            _, self.norm = build_norm_layer(norm_cfg, out_channels)
        else:
            self.norm = None

        if act_cfg is not None:
            self.activation = build_activation_layer(act_cfg)
        else:
            self.activation = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Union[int, _size] = 1,
        padding: Union[int, _size] = 0,
        dilation: Union[int, _size] = 1,
        groups: int = 1,
        bias: bool = False,
        mid_channels: int = 0,
        conv_cfg: dict = dict(type="MetaConv2d"),
        norm_cfg: dict | None = dict(type="BN", affine=True),
        act_cfg: dict | None = dict(type="ReLU", inplace=True),
        device: Device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ) -> None:
        super().__init__()
        conv_cfg = conv_cfg.copy()
        if conv_cfg["type"] not in {"Conv2d", None}:
            conv_cfg.update({"device": device, "mid_channels": mid_channels})
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if norm_cfg is not None:
            _, self.norm = build_norm_layer(norm_cfg, out_channels)
        else:
            self.norm = None

        if act_cfg is not None:
            self.activation = build_activation_layer(act_cfg)
        else:
            self.activation = None

    
    def set_test_mode(self, test_mode: bool = True):
        self.conv.set_test_mode(test_mode)

    def set_near2far_matrix(self, near2far_matrix: Tensor):
        self.conv.set_near2far_matrix(near2far_matrix)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Meta_Base(nn.Module):
    def __init__(
        self,
        *args,
        conv_cfg=dict(type="MetaConv2d"),
        conv_cfg_local=None,
        linear_cfg=dict(type="Linear"),
        device: Device = torch.device("cuda:0"),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        with MODELS.switch_scope_and_registry(None) as registry:
            self._conv = (registry.get(conv_cfg["type"]),)
            if conv_cfg_local:
                self._conv_local = (registry.get(conv_cfg_local["type"]),)
            else:
                self._conv_local = None
            self._linear = (registry.get(linear_cfg["type"]),)
            self._conv_linear = self._conv + self._linear
            if self._conv_local:
                self._conv_linear += self._conv_local
        self.meta_params = MetaParams(
            path_depth=conv_cfg["path_depth"],
            delta_z_mode=conv_cfg["delta_z_mode"],
            pixel_size_mode=conv_cfg["pixel_size_mode"],
            path_multiplier=conv_cfg["path_multiplier"],
            lambda_mode=conv_cfg["lambda_mode"],
            rotate_mode=conv_cfg["rotate_mode"],
            gumbel_mode=conv_cfg["gumbel_mode"],
            enable_identity=conv_cfg["enable_identity"],
            gumbel_T=conv_cfg["gumbel_T"],
            swap_mode=conv_cfg["swap_mode"],
            ref_lambda=conv_cfg["ref_lambda"],
            ref_pixel_size=conv_cfg["ref_pixel_size"],
            pixel_size_res=conv_cfg["pixel_size_res"],  # nm
            delta_z_res=conv_cfg["delta_z_res"],  # nm
            lambda_res=conv_cfg["lambda_res"],  # nm
            delta_z_data=conv_cfg["delta_z_data"],  # nm
            pixel_size_data=conv_cfg["pixel_size_data"],  # nm
            lambda_data=conv_cfg["lambda_data"],
            lambda_train=conv_cfg["lambda_train"],
            delta_z_train=conv_cfg["delta_z_train"],
            pixel_size_train=conv_cfg["pixel_size_train"],
            device=device,
        )

    def reset_parameters(self, random_state: int = None) -> None:
        for name, m in self.named_modules():
            if isinstance(m, self._conv_linear + (MetaParams,)):
                if random_state is not None:
                    # deterministic seed, but different for different layer, and controllable by random_state
                    set_torch_deterministic(random_state + sum(map(ord, name)))
                m.reset_parameters()
                if isinstance(
                    m, self._conv
                ):  # all metaconv layers should share the same meta_params
                    m.set_meta_params(self.meta_params)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def reset_linear_parameters(self) -> None:
        for name, m in self.named_modules():
            if isinstance(m, self._linear):
                m.reset_parameters()

    def backup_phases(self) -> None:
        self.phase_backup = {}
        for layer_name, layer in self.fc_layers.items():
            self.phase_backup[layer_name] = {
                "weight": (
                    layer.weight.data.clone() if layer.weight is not None else None
                ),
            }

    def set_light_redist(self, flag: bool = True):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_light_redist"
            ):
                layer.set_light_redist(flag)

    def set_input_power_gating(self, flag: bool = True, ER: float = 6) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_input_power_gating"
            ):
                layer.set_input_power_gating(flag, ER)

    def set_output_power_gating(self, flag: bool = True) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_output_power_gating"
            ):
                layer.set_output_power_gating(flag)

    def restore_phases(self) -> None:
        for layer_name, layer in self.fc_layers.items():
            backup = self.phase_backup[layer_name]
            for param_name, param_src in backup.items():
                param_dst = getattr(layer, param_name)
                if param_src is not None and param_dst is not None:
                    param_dst.data.copy_(param_src.data)

    def set_phase_variation(self, flag: bool = True):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_phase_variation"
            ):
                layer.set_phase_variation(flag)

    def set_output_noise(self, noise_std: float = 0.0):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_output_noise"
            ):
                layer.set_output_noise(noise_std)

    def phase_rounding(self):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "phase_rounding"
            ):
                layer.phase_rounding()

    def update_lambda_pixel_size(self):
        self.meta_params.update_lambda_pixel_size()

    def set_gumbel_temperature(self, T: float = 5.0):
        self.meta_params.set_gumbel_temperature(T)

    def set_encode_mode(self, mode: str = "mag"):
        index = 0
        for layer in self.modules():
            if (
                isinstance(layer, self._conv_linear)
                and hasattr(layer, "set_encode_mode")
                and index == 0
            ):
                # print("This is Here!!!!!!!!")
                layer.set_encode_mode(mode)
                index += 1

    def set_gamma_noise(
        self, noise_std: float = 0.0, random_state: Optional[int] = None
    ) -> None:
        self.gamma_noise_std = noise_std
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_gamma_noise"
            ):
                layer.set_gamma_noise(noise_std, random_state=random_state)

    def set_crosstalk_noise(self, flag: bool = True) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_crosstalk_noise"
            ):
                layer.set_crosstalk_noise(flag)

    def set_weight_noise(self, noise_std: float = 0.0) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_weight_noise"
            ):
                layer.set_weight_noise(noise_std)

    def requires_grad_Meta(self, mode: bool = True):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "requires_grad_Meta"
            ):
                layer.requires_grad_Meta(mode)

    def set_delta_z_data(self, delta_z_data: float = 0.0) -> None:
        self.meta_params.set_delta_z_data(delta_z_data)

    def set_lambda_data(self, lambda_data: float = 0.0) -> None:
        self.meta_params.set_lambda_data(lambda_data)

    def set_pixel_size_data(self, pixel_size_data: float = 0.0) -> None:
        self.meta_params.set_pixel_size_data(pixel_size_data)

    def set_delta_z_mode(self, mode: bool = True):
        self.meta_params.set_delta_z_mode(mode)

    def set_weight_train(self, flag: bool = True):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_weight_train"
            ):
                layer.set_weight_train(flag)

    def set_skip_path(self, flag: bool = True):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(layer, "set_skip_path"):
                layer.set_skip_path(flag)

    def set_pixel_size_mode(self, mode: bool = True):
        self.meta_params.set_pixel_size_mode(mode)

    def requires_grad_beta(self, mode: bool = True):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "requires_grad_beta"
            ):
                layer.requires_grad_beta(mode)

    def set_skip_meta(self, flag: bool = False):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(layer, "set_skip_meta"):
                layer.set_skip_meta(flag)

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_weight_bitwidth"
            ):
                layer.set_weight_bitwidth(w_bit)

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_input_bitwidth"
            ):
                layer.set_input_bitwidth(in_bit)

    def sinkhorn_perm(self, n_step=10, t_min=0.1, noise_std=0.01):
        with torch.no_grad():
            w = self.meta_params.build_swap_permutation().data.abs()
            w = self.meta_params.unitary_projection(
                w, n_step=n_step, t=t_min, noise_std=noise_std
            )
            self.meta_params.swap_permutation.data.copy_(w)
    
    def get_smoothing_loss(self, lambda_smooth=1e-3):
        loss = 0
        # exit(0)
        for layer in self.modules():
            # print(layer._get_name())
            if isinstance(layer, self._conv_linear) and hasattr(layer, "get_smoothing_loss"):
                smooth = layer.get_smoothing_loss(lambda_smooth)
                # print(smooth)
                loss += smooth
                
        return loss
    
    def set_pixel_size_res(self, res: int) -> None:
        self.meta_params.set_pixel_size_res(res)

    def set_delta_z_res(self, res: int) -> None:
        self.meta_params.set_delta_z_res(res)

    def set_phase_res(self, res: int) -> None:
        self.phase_res = res
        for layer in self.modules():
            if isinstance(layer, self.phase_res) and hasattr(layer, "set_phase_res"):
                layer.set_phase_res(res)

    def load_parameters(self, param_dict: Dict[str, Dict[str, Tensor]]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        for layer_name, layer_param_dict in param_dict.items():
            self.layers[layer_name].load_parameters(layer_param_dict)

    def build_obj_fn(self, X: Tensor, y: Tensor, criterion: Callable) -> Callable:
        def obj_fn(X_cur=None, y_cur=None, param_dict=None):
            if param_dict is not None:
                self.load_parameters(param_dict)
            if X_cur is None or y_cur is None:
                data, target = X, y
            else:
                data, target = X_cur, y_cur
            pred = self.forward(data)
            return criterion(pred, target)

        return obj_fn

    def enable_fast_forward(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "enable_fast_forward"
            ):
                layer.enable_fast_forward()

    def disable_fast_forward(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "disable_fast_forward"
            ):
                layer.disable_fast_forward()

    def sync_parameters(self, src: str = "weight") -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "sync_parameters"
            ):
                layer.sync_parameters(src=src)

    def build_weight(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(layer, "build_weight"):
                layer.build_weight()

    def print_parameters(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "print_parameters"
            ):
                layer.print_parameters()

    def switch_mode_to(self, mode: str) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "switch_mode_to"
            ):
                layer.switch_mode_to(mode)

    def set_enable_remap(self, enable_remap: bool) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_enable_remap(enable_remap)

    def build_rotation_mask(self, mode=None, batch_size: int = 32):
        self.meta_params.build_rotate_mask(mode, batch_size)

    def get_swap_loss(self):
        loss = 0
        if hasattr(self.meta_params, "get_swap_loss"):
            loss = self.meta_params.get_swap_loss().detach().data.item()
        return loss

    def get_swap_alm_loss(self, rho: float = 0.1):
        loss = 0
        if hasattr(self.meta_params, "get_swap_alm_loss"):
            loss = loss + self.meta_params.get_swap_alm_loss(rho=rho)
        return loss

    def update_swap_alm_multiplier(
        self, rho: float = 0.1, max_lambda: Optional[float] = None
    ):
        if hasattr(self.meta_params, "update_swap_alm_multiplier"):
            self.meta_params.update_swap_alm_multiplier(rho=rho, max_lambda=max_lambda)

    def get_alm_multiplier(self):
        return self.meta_params.swap_alm_multiplier.data.mean().item()

    def check_perm(self):
        with torch.no_grad():
            return tuple(
                range(len(self.meta_params.build_swap_permutation()))
            ) == tuple(
                sorted(self.meta_params.build_swap_permutation().cpu().numpy().tolist())
            )

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
