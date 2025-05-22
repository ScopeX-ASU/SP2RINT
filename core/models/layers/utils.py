import os
import random
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import tqdm
from pyutils.general import logger
from pyutils.quant.lsq import get_default_kwargs_q, grad_scale, round_pass
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor, nn
from torch.nn import Parameter

# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))


__all__ = [
    "STE",
    "mzi_out_diff_to_phase",
    "mzi_phase_to_out_diff",
    "DeterministicCtx",
    "calculate_grad_hessian",
    "pad_quantize_fn",
    "hard_diff_round",
    "WeightQuantizer_LSQ",
    "DeviceQuantizer",
    "gradientmask",
    "merge_chunks",
    "partition_chunks",
]


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_noisy):
        ## use x_noisy as forward
        return x_noisy

    @staticmethod
    def backward(ctx, grad_output):
        ## gradient flow back to x, not x_noisy
        return grad_output.clone(), None


def mzi_out_diff_to_phase(x: Tensor) -> Tensor:
    """Y-branch-based 1x2 MZI.
    The power difference on two output ports, i.e., out_diff = out1 - out2, converted to the internal arm phase difference (delta_phi)
    delta_phi \in [-pi/2, pi/2], if delta_phi > 0, heat up upper arm phase shifter; if delta_phi < 0, heat up lower arm phase shifter
    out_diff \in [-1, 1] ideally with infinite extinction ratio.
    out1 = 0.5(1+sin(delta_phi))
    out2 = 0.5(1-sin(delta_phi))
    out_diff = out1-out2=sin(delta_phi)
    delta_phi = arcsin(out_diff), need to make sure delta_phi is in the range of [-pi/2, pi/2]

    Args:
        x (Tensor): output port power difference of the 1x2 MZI

    Returns:
        Tensor: delta phi
    """
    return torch.asin(
        x.clamp(-1, 1)
    )  # this clamp is for safety, as the input x may not be exactly in [-1, 1]


def mzi_phase_to_out_diff(x: Tensor) -> Tensor:
    """Y-branch-based 1x2 MZI.
    The internal arm phase difference (delta_phi) converted to the power difference on two output ports, i.e., out_diff = out1 - out2
    delta_phi \in [-pi/2, pi/2], if delta_phi > 0, heat up upper arm phase shifter; if delta_phi < 0, heat up lower arm phase shifter
    out_diff \in [-1, 1] ideally with infinite extinction ratio.
    out1 = 0.5(1+sin(delta_phi))
    out2 = 0.5(1-sin(delta_phi))
    out_diff = out1-out2=sin(delta_phi)

    Args:
        x (Tensor): delta phi

    Returns:
        Tensor: output port power difference of the 1x2 MZI
    """
    return torch.sin(x)


class DeterministicCtx:
    def __init__(self, random_state: Optional[int] = None) -> None:
        self.random_state = random_state

    def __enter__(self):
        self.random_state = random.getstate()
        self.numpy_random_state = np.random.get_state()
        self.torch_random_state = torch.random.get_rng_state()
        self.torch_cuda_random_state = torch.cuda.get_rng_state()
        set_torch_deterministic(self.random_state)
        return self

    def __exit__(self, *args):
        random.setstate(self.random_state)
        np.random.seed(self.numpy_random_state)
        np.random.set_state(self.numpy_random_state)
        torch.random.set_rng_state(self.torch_random_state)
        torch.cuda.set_rng_state(self.torch_cuda_random_state)


def calculate_grad_hessian(
    model, train_loader, criterion, num_samples=10, device="cuda:0"
):
    ## average gradients and second order gradients will be stored in weight._first_grad and weight._second_grad
    is_train = model.training
    model.train()
    ## freeze BN stat is important
    bn_state = None
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            bn_state = m.training
            m.eval()
    params = []
    for m in model.modules():
        if isinstance(m, model._conv_linear):
            # print(m)
            m.weight._first_grad = 0
            m.weight._second_grad = 0
            params.append(m.weight)
    generator = torch.Generator(params[0].device).manual_seed(0)

    for idx, (data, target) in enumerate(tqdm.tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward(create_graph=True)
        ## record the gradient
        grads = []
        for p in params:
            if p.grad is not None:
                ## accumulate gradients and average across all batches
                p._first_grad += p.grad.data / len(train_loader)
                grads.append(p.grad)

        # compute second order gradient
        for _ in range(num_samples):
            zs = [
                torch.randint(0, 2, p.size(), generator=generator, device=p.device)
                * 2.0
                - 1.0
                for p in params
            ]  # Rademacher distribution {-1.0, 1.0}
            h_zs = torch.autograd.grad(
                grads,
                params,
                grad_outputs=zs,
                only_inputs=True,
                retain_graph=num_samples - 1,
            )
            for h_z, z, p in zip(h_zs, zs, params):
                ## accumulate second order gradients
                p._second_grad += h_z * z / (num_samples * len(train_loader))
        model.zero_grad()
        if idx == 3:
            break
    # print(params[0]._first_grad, params[0]._first_grad.shape)
    # print(params[0]._second_grad, params[0]._second_grad.shape)
    # print(params[0].shape)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.train(bn_state)
    model.train(is_train)


def uniform_quantize(num_levels, gradient_clip=False):
    class qfn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # n = float(2 ** k - 1)
            n = (
                num_levels - 1
            )  # explicit assign number of quantization level,e.g., k=5 or 8
            out = torch.round(input * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            if gradient_clip:
                grad_input.clamp_(-1, 1)
            return grad_input

    return qfn().apply


class pad_quantize_fn(torch.nn.Module):
    def __init__(self, w_bit, quant_ratio: float = 1.0, v_max: float = 2.0):
        """Differentiable weight quantizer. Support different algorithms. Support Quant-Noise with partial quantization.

        Args:
            w_bit (int): quantization bitwidth
            quant_ratio (float, optional): Quantization ratio to support full-precision gradient flow. Defaults to 1.0.
            v_max (float, optional): Maxmimum voltage (exclusive).
        """
        super().__init__()

        self.w_bit = w_bit  # w_bit is the number of quantization level, not bitwidth !

        self.quant_ratio = quant_ratio
        assert 0 <= quant_ratio <= 1, logger.error(
            f"Wrong quant ratio. Must in [0,1], but got {quant_ratio}"
        )
        self.uniform_q = uniform_quantize(num_levels=w_bit, gradient_clip=True)
        self.v_max = v_max

    def set_quant_ratio(self, quant_ratio=None):
        if quant_ratio is None:
            ### get recommended value
            quant_ratio = [
                None,
                0.2,
                0.3,
                0.4,
                0.5,
                0.55,
                0.6,
                0.7,
                0.8,
                0.83,
                0.86,
                0.89,
                0.92,
                0.95,
                0.98,
                0.99,
                1,
            ][min(self.w_bit, 16)]
        assert 0 <= quant_ratio <= 1, logger.error(
            f"Wrong quant ratio. Must in [0,1], but got {quant_ratio}"
        )
        self.quant_ratio = quant_ratio

    def forward(self, x):
        if self.quant_ratio < 1 and self.training:
            ### implementation from fairseq
            ### must fully quantize during inference
            quant_noise_mask = torch.empty_like(x, dtype=torch.bool).bernoulli_(
                1 - self.quant_ratio
            )
        else:
            quant_noise_mask = None

        weight = torch.sigmoid(x)  # [0, 1]
        weight_q = self.uniform_q(weight)
        if quant_noise_mask is not None:
            noise = weight_q.data.sub_(weight.data).masked_fill_(quant_noise_mask, 0)
            ### unquantized weights have to follow reparameterization, i.e., tanh
            weight_q = weight + noise

        return weight_q


class HardRoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        # x_max, indices = x.max(dim=1, keepdim=True)
        # illegal_indices = [k for k, v in Counter(indices.view(-1).cpu().numpy().tolist()).items() if v > 1]
        # mask = x_max > 0.95
        # for i in illegal_indices:

        mask = (x.max(dim=-1, keepdim=True)[0] > 0.9).repeat(
            [1] * (x.dim() - 1) + [x.size(-1)]
        )
        ctx.mask = mask
        return torch.where(mask, x.round(), x)

    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output.clone().masked_fill_(ctx.mask, 0)


def hard_diff_round(x: Tensor) -> Tensor:
    """Project to the closest permutation matrix"""
    assert x.size(-1) == x.size(
        -2
    ), f"input x has to be a square matrix, but got {x.size()}"
    return HardRoundFunction.apply(x)


class WeightQuantizer_LSQ(nn.Module):
    def __init__(self, out_features: int, device="cuda:0", **kwargs_q):
        super().__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q)
        self.nbits = kwargs_q["nbits"]
        if self.nbits <= 0:  # no need to enable quantize
            self.register_parameter("alpha", None)
            return
        self.q_mode = kwargs_q["mode"]
        self.offset = kwargs_q["offset"]
        self.zero_point = None
        self.device = device
        if self.q_mode == "kernel_wise":
            self.alpha = Parameter(torch.empty(out_features, device=device))
            if self.offset:
                self.zero_point = Parameter(torch.empty(out_features, device=device))
                torch.nn.init.zeros_(self.zero_point)
        else:
            self.alpha = Parameter(torch.empty(1, device=device))
            if self.offset:
                self.zero_point = Parameter(torch.tensor([0.0], device=device))

        self.register_buffer("init_state", torch.zeros(1))
        self.register_buffer("signed", torch.tensor([kwargs_q["signed"]]))

    def update_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q["nbits"] = nbits
        self._compute_quant_range()

    def _compute_quant_range(self):
        if self.signed == 1:
            self.Qn = -(2 ** (self.nbits - 1))
            self.Qp = 2 ** (self.nbits - 1) - 1
        else:
            self.Qn = 0
            self.Qp = 2**self.nbits - 1

    def extra_repr(self):
        if self.alpha is None:
            return "fake"
        return "{}".format(self.kwargs_q)

    def _initialize_state(self, x):
        logger.info(
            f"LSQ Weight quantizer: (mode: {self.q_mode}): initialize weight scale for int{self.nbits} quantization with offset: {self.offset}"
        )
        if self.q_mode == "kernel_wise":
            logger.info(f"Scale dimension: {self.alpha.shape}")

        self._compute_quant_range()
        self.alpha.data.copy_(x.data.abs().mean().mul_(2 / self.Qp**0.5))
        if self.offset:
            self.zero_point.data.copy_(
                self.zero_point.data * 0.9
                + 0.1 * (x.data.min() - self.alpha.data * self.Qn)
            )
        self.init_state.fill_(1)

    def forward(self, x):
        if self.alpha is None:
            return x

        if self.training and self.init_state == 0:
            self._initialize_state(x)

        assert self.init_state == 1

        g = 1.0 / (x.data.numel() * self.Qp) ** 0.5

        self.alpha.data.clamp_(min=1e-4)

        alpha = grad_scale(self.alpha, g)  # scale alpha's gradient by g

        if len(x.shape) == 2:  # linear layer
            alpha = alpha[..., None]
        elif len(x.shape) == 3:
            alpha = alpha[..., None, None]
        elif len(x.shape) == 4:  # conv layer
            alpha = alpha[..., None, None, None]
        elif len(x.shape) == 6:
            alpha = alpha[..., None, None, None, None, None]
        else:
            raise NotImplementedError

        if self.offset:
            zero_point = round_pass(self.zero_point)
            zero_point = grad_scale(zero_point, g)
            # zero_point = (
            #     zero_point[..., None]
            #     if len(x.shape) == 2
            #     else zero_point[..., None, None, None]
            # )
            if len(x.shape) == 2:
                zero_point = zero_point[..., None]
            elif len(x.shape) == 3:
                zero_point = zero_point[..., None, None]
            elif len(x.shape) == 4:
                zero_point = zero_point[..., None, None, None]
            elif len(x.shape) == 6:
                zero_point = zero_point[..., None, None, None, None, None]
            x = round_pass((x / alpha + zero_point).clamp(self.Qn, self.Qp))
            x = (x - zero_point) * alpha
        else:
            x = round_pass((x / alpha).clamp(self.Qn, self.Qp)).mul(alpha)

        return x


class ActQuantizer_LSQ(nn.Module):
    def __init__(self, in_features: int, device="cuda:0", **kwargs_q):
        super().__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q)
        self.nbits = kwargs_q["nbits"]
        if self.nbits <= 0:  # no need to enable quantize
            self.register_parameter("alpha", None)
            return
        self.q_mode = kwargs_q["mode"]
        self.offset = kwargs_q["offset"]
        self.zero_point = None
        self.device = device
        if self.q_mode == "kernel_wise":
            self.alpha = Parameter(torch.empty(in_features, device=device))
            if self.offset:
                self.zero_point = Parameter(torch.empty(in_features, device=device))
                torch.nn.init.zeros_(self.zero_point)
        else:
            self.alpha = Parameter(torch.empty(1, device=device))
            if self.offset:
                self.zero_point = Parameter(torch.tensor([0.0], device=device))

        self.register_buffer("init_state", torch.zeros(1))
        self.register_buffer("signed", torch.zeros(1))

    def update_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q["nbits"] = self.nbits = nbits
        self._compute_quant_range()

    def _compute_quant_range(self):
        if self.signed == 1:
            self.Qn = -(2 ** (self.nbits - 1))
            self.Qp = 2 ** (self.nbits - 1) - 1
        else:
            self.Qn = 0
            self.Qp = 2**self.nbits - 1

    def extra_repr(self):
        if self.alpha is None:
            return "fake"
        return "{}".format(self.kwargs_q)

    def _initialize_state(self, x):
        logger.info(
            f"LSQ Act quantizer: (mode: {self.q_mode}): initialize weight scale for int{self.nbits} quantization with offset: {self.offset}"
        )
        if self.q_mode == "kernel_wise":
            logger.info(f"Scale dimension: {self.alpha.shape}")
        # choose implementation from https://github.com/YanjingLi0202/Q-ViT/blob/main/Quant.py
        if (
            x.data.min() < -1e-5
        ):  # there are significant negative values we will use signed representation
            self.signed.data.fill_(1)
        self._compute_quant_range()
        self.alpha.data.copy_(x.data.abs().mean().mul_(2 / self.Qp**0.5))
        if self.offset:
            self.zero_point.data.copy_(
                self.zero_point.data * 0.9
                + 0.1 * (x.data.min() - self.alpha.data * self.Qn)
            )
        self.init_state.fill_(1)

    def forward(self, x):
        if self.alpha is None:
            return x

        if self.training and self.init_state == 0:
            self._initialize_state(x)

        assert self.init_state == 1

        g = 1.0 / (x.data.numel() * self.Qp) ** 0.5

        self.alpha.data.clamp_(min=1e-4)

        alpha = grad_scale(self.alpha, g)  # scale alpha's gradient by g

        if len(x.shape) == 2:  # linear layer
            alpha = alpha.unsqueeze(0)
        elif len(x.shape) == 3:
            # 1D CNN => x.shape = [N, C, L]
            # We want alpha to broadcast over (N, C, L),
            # typically shape [1, 1, 1]
            alpha = alpha.unsqueeze(0).unsqueeze(2)     # shape [1, 1]
        elif len(x.shape) == 4:  # conv layer
            alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            raise NotImplementedError

        if self.offset:
            zero_point = (
                self.zero_point.round() - self.zero_point
            ).detach() + self.zero_point
            zero_point = grad_scale(zero_point, g)
            # zero_point = (
            #     zero_point.unsqueeze(0)
            #     if len(x.shape) == 2
            #     else zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            # )
            if len(x.shape) == 2:
                zero_point = zero_point.unsqueeze(0)
            elif len(x.shape) == 3:
                zero_point = zero_point.unsqueeze(0).unsqueeze(2)
            elif len(x.shape) == 4:
                zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)            
            x = round_pass((x / alpha + zero_point).clamp(self.Qn, self.Qp))
            x = (x - zero_point) * alpha
        else:
            x = round_pass((x / alpha).clamp(self.Qn, self.Qp)).mul(alpha)

        return x


class PixelLambdaConstraint(nn.Module):
    def __init__(
        self,
        ref_lambda_: float,
        origin_lambda: float,  # Pass in the one either 0.532 or the properly trained lambda
        ref_pixel_size: float,
        pixel_size_mode: str = "fixed",
        device="cuda:0",
    ):
        super().__init__()

        # Reference Lambda and pixel size
        # For init stage, 0.532 and 0.4 will be used
        # For adaptation stage, pixel size will be fixed, constrain lambda
        self.ref_lambda_ = ref_lambda_  # 0.532
        self.ref_pixel_size = ref_pixel_size  # 0.3
        self.origin_lambda = origin_lambda  # the properly trained lambda for adaptation stage or 0.532 for init stage
        self.pixel_size_mode = pixel_size_mode
        self.device = device

        self.device = device
        self.register_buffer("init_state", torch.zeros(1))

    def forward(self, pixel_size: Tensor, lambda_: Tensor):
        if self.training and self.init_state == 0:
            self._initialize_state(pixel_size)

        lambda_ = lambda_.clamp(
            min=(self.origin_lambda - 0.02), max=(self.origin_lambda + 0.02)
        )
        # Tuned lambda / reference lambda 0.532
        ratio = lambda_ / self.ref_lambda_  # This is the reference wavelength
        # Pixel size not less than reference pixel size * ratio + 0.02um(minimal gap between pixels)
        pixel_size_min = (
            self.ref_pixel_size * ratio + 0.02
        )  # This is the minimum pixel size determined from SiO2 metasurface
        # Clamp to proper range, clamp function has forward and backward function, differentiable
        pixel_size = pixel_size.clamp(min=pixel_size_min)
        return pixel_size, lambda_

    def _initialize_state(self, x):
        logger.info("Initializing pixel-size and lambda constraint")
        self.init_state.fill_(1)


class DeviceQuantizer(nn.Module):
    def __init__(self, resolution: int = 1, mode: str = "nm", device="cuda:0"):
        super().__init__()
        self.device = device
        self.resolution = resolution
        self.mode = mode

        self.set_resolution(self.resolution)
        self.register_buffer("init_state", torch.zeros(1))

    def set_resolution(self, resolution):
        self.resolution = resolution
        if self.mode == "nm":
            self.scale_factor = 1000 / self.resolution
        elif self.mode == "degree":
            self.scale_factor = self.resolution
        elif self.mode == "dep":
            self.scale_factor = 10**self.resolution
        elif self.mode == "mag":
            self.act_fn = torch.nn.Tanh()
            self.scale_factor = None
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.training and self.init_state == 0:
            self._initialize_state(x)

        if self.mode == "degree":
            phase_degree = x * (180 / torch.pi)
            quantized_degrees = (
                round_pass(phase_degree / self.scale_factor) * self.scale_factor
            )
            quantized_x = quantized_degrees * (torch.pi / 180)
        elif self.mode == "mag":
            quantized_x = (self.act_fn(x) + 1) / 2 # must be in [0, 1]
        else:
            scaled_x = x * self.scale_factor
            rounded_x = round_pass(scaled_x)
            quantized_x = rounded_x / self.scale_factor

        return quantized_x

    def extra_repr(self):
        if self.resolution is not None:
            s = "resolution={resolution}"
        else:
            s = "resolution=None"
            print
        if self.mode is not None:
            s += ", mode={mode}"
        if self.scale_factor is not None:
            s += ", scale_factor={scale_factor}"
        return s.format(**self.__dict__)

    def _initialize_state(self, x):
        logger.info(
            f"Initializing device quantization with resolution: {self.resolution} under {self.mode} mode"
        )
        self.init_state.fill_(1)


class GradientMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(mask)
        return x

    @staticmethod
    def backward(ctx, grad_outputs):
        (mask,) = ctx.saved_tensors
        grad_inputs = grad_outputs * mask

        return grad_inputs, None


gradientmask = GradientMask.apply


def merge_chunks(x: Tensor) -> Tensor:
    # x = [h1, w1, h2, w2, ...., hk, wk]
    # out: [h1*h2*...*hk, w1*w2*...*wk]

    dim = x.dim()
    x = x.permute(
        list(range(0, dim, 2)) + list(range(1, dim + 1, 2))
    )  # x = [h, bs, w, bs]
    x = x.reshape(np.prod([x.shape[i] for i in range(dim // 2)]), -1)

    return x


def partition_chunks(x: Tensor, out_shape: int | Tuple[int, ...]) -> Tensor:
    ### x: [h1*h2*...*hk, w1*w2*...*wk]
    ### out_shape: (h1, w1, ...)
    ### out: [h1, w1, h2, w2, ...., hk, wk]
    in_shape = list(out_shape[::2]) + list(out_shape[1::2])
    x = x.reshape(in_shape)  # [h1, h2, ..., hk, w1, w2, ..., wk]
    x = x.permute(
        torch.arange(len(out_shape)).view(2, -1).t().flatten().tolist()
    )  # [h1, w1, h2, w2, ...., hk, wk]
    return x


def polynomial(x: Tensor, coeff: Tensor) -> Tensor:
    ## coeff: from high to low order coefficient, last one is constant
    ## e.g., [p5, p4, p3, p2, p1, p0] -> p5*x^5 + p4*x^4 + p3*x^3 + p2*x^2 + p1*x + p0
    # print(x.shape)
    out = 0
    for i in range(coeff.size(0) - 1, 0, -1):
        out = out + x.pow(i).mul_(coeff[coeff.size(0) - i - 1])
    out.add_(coeff[-1])
    return out


def polynomial2(
    x: Tensor | float, y: Tensor | float, coeff: Tensor | List[float]
) -> Tensor | float:
    ## coeff [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3]
    if len(coeff) == 3:
        # coeff [1, x, y]
        return coeff[0] + coeff[1] * x + coeff[2] * y
    elif len(coeff) == 6:
        # coeff [1, x, y, x^2, xy, y^2]
        return (
            coeff[0]
            + coeff[1] * x
            + coeff[2] * y
            + coeff[3] * x**2
            + coeff[4] * x * y
            + coeff[5] * y**2
        )
    elif len(coeff) == 10:
        # coeff [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3]
        x_2, y_2 = x**2, y**2
        return (
            coeff[0]
            + coeff[1] * x
            + coeff[2] * y
            + coeff[3] * x_2
            + coeff[4] * y * x
            + coeff[5] * y_2
            + coeff[6] * x_2 * x
            + coeff[7] * y * x_2
            + coeff[8] * y_2 * x
            + coeff[9] * y_2 * y
        )
    else:
        raise NotImplementedError
