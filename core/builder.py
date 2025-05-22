from typing import Optional, Tuple

import torch
import torch.nn as nn
from pyutils.config import configs
from pyutils.datasets import get_dataset
from pyutils.loss import AdaptiveLossSoft, CrossEntropyLossSmooth, DKDLoss, KDLoss
from pyutils.lr_scheduler.warmup_cosine_restart import CosineAnnealingWarmupRestarts
from pyutils.optimizer.dadapt_adam import DAdaptAdam
from pyutils.optimizer.dadapt_sgd import DAdaptSGD
from pyutils.optimizer.prodigy import Prodigy
from pyutils.optimizer.sam import SAM
from pyutils.typing import DataLoader, Optimizer, Scheduler
from torch.types import Device
from .utils import (
    EnergyConservationLoss, 
    NL2norm, 
    DownsampleRateScheduler, 
    InvDesSharpnessScheduler,
    TransferMatrixMatchingLoss, 
    ResponseMatchingLoss, 
    TransferMatrixSmoothError,
    CosSimLoss,
    AdmmConsistencyLoss,
    HighFreqPenalty,
    End2EndSharpnessScheduler,
)

from core.datasets import (
    CIFAR10Dataset,
    CIFAR100Dataset,
    DarcyDataset,
    FashionMNISTDataset,
    MNISTDataset,
    NavierStokesDataset,
    QuickDrawDataset,
    HillenbrandDataset,
    SVHNDataset,
    VowelDataset,
)
from core.models import *  # noqa: F403

__all__ = [
    "make_dataloader",
    "make_model",
    "make_optimizer",
    "make_scheduler",
    "make_criterion",
]


def make_dataloader(
    cfg: dict = None, splits=["train", "valid", "test"]
) -> Tuple[DataLoader, DataLoader]:
    cfg = cfg or configs.dataset
    name = cfg.name.lower()
    if name == "mnist":
        train_dataset, validation_dataset, test_dataset = (
            MNISTDataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                binarize_threshold=0.273,
                digits_of_interest=list(range(10)),
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "fmnist":
        train_dataset, validation_dataset, test_dataset = (
            FashionMNISTDataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "svhn":
        train_dataset, validation_dataset, test_dataset = (
            SVHNDataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                binarize_threshold=0.1307,
                grayscale=False,
                digits_of_interest=list(range(10)),
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
                augment=False,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "connectionist":
        train_dataset, validation_dataset, test_dataset = (
            VowelDataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                n_input_features=cfg.n_input_features,
                seed=42,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "hillenbrand":
        train_dataset, validation_dataset, test_dataset = (
            HillenbrandDataset(
                root=cfg.root,
                split=split,
                n_valid_speakers=cfg.n_valid_speakers,
                n_test_speakers=cfg.n_test_speakers,
                feature_type="padded_signal",  # "mfcc" or "padded_signal"
                random_seed=42,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "cifar10":
        train_dataset, validation_dataset, test_dataset = (
            CIFAR10Dataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                grayscale=False,
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
                augment=cfg.transform != "basic",
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "cifar100":
        train_dataset, validation_dataset, test_dataset = (
            CIFAR100Dataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                grayscale=False,
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
                augment=cfg.transform != "basic",
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "darcy":
        train_dataset, validation_dataset, test_dataset = (
            DarcyDataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                img_height=cfg.img_height,
                img_width=cfg.img_width,
                positional_encoding=cfg.positional_encoding,
                encode_input=cfg.encode_input,
                encode_output=cfg.encode_output,
                processed_dir=cfg.processed_dir,
                train_noise_cfg=(getattr(configs, "train_noise", {})),
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "quickdraw":
        train_dataset, validation_dataset, test_dataset = (
            QuickDrawDataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                classes_of_interest=cfg.classes_of_interest,
                num_classes=cfg.num_classes,
                n_samples_per_class=cfg.n_samples_per_class,
                n_bytes_per_class=cfg.n_bytes_per_class,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "navierstokes":
        train_dataset, validation_dataset, test_dataset = (
            NavierStokesDataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                test_ratio=cfg.test_ratio,
                img_height=cfg.img_height,
                img_width=cfg.img_width,
                processed_dir=cfg.processed_dir,
                train_noise_cfg=(getattr(configs, "train_noise", {})),
                # train_noise_cfg=cfg.train_noise_cfg,
            )
            for split in ["train", "valid", "test"]
        )
    else:
        train_dataset, test_dataset = get_dataset(
            name,
            cfg.img_height,
            cfg.img_width,
            dataset_dir=cfg.root,
            transform=cfg.transform,
        )
        validation_dataset = None

    train_loader = (
        torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=configs.run.batch_size,
            shuffle=int(cfg.shuffle),
            pin_memory=True,
            num_workers=cfg.num_workers,
        )
        if train_dataset is not None
        else None
    )

    validation_loader = (
        torch.utils.data.DataLoader(
            dataset=validation_dataset,
            batch_size=configs.run.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=cfg.num_workers,
        )
        if validation_dataset is not None
        else None
    )

    test_loader = (
        torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=configs.run.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=cfg.num_workers,
        )
        if test_dataset is not None
        else None
    )

    return train_loader, validation_loader, test_loader


def make_model(
    device: Device, model_cfg: Optional[str] = None, random_state: int = None, **kwargs
) -> nn.Module:
    model_cfg = model_cfg or configs.model
    name = model_cfg.name
    if name == "Meta_CNN":
        model = eval(model_cfg.name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            kernel_list=model_cfg.kernel_list,
            mid_channel_list=model_cfg.mid_channel_list,
            kernel_size_list=model_cfg.kernel_size_list,
            pool_out_size=model_cfg.pool_out_size,
            stride_list=model_cfg.stride_list,
            padding_list=model_cfg.padding_list,
            groups_list=model_cfg.groups_list,
            dilation_list=model_cfg.dilation_list,
            hidden_list=model_cfg.hidden_list,
            conv_cfg=model_cfg.conv_cfg,
            linear_cfg=model_cfg.linear_cfg,
            norm_cfg=model_cfg.norm_cfg,
            act_cfg=model_cfg.act_cfg,
            device=device,
        ).to(device)
        model.set_encode_mode(model_cfg.encode_mode)
        model.reset_parameters(random_state)
    elif name == "FullOptMetalens":
        model = eval(model_cfg.name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            kernel_list=model_cfg.kernel_list,
            mid_channel_list=model_cfg.mid_channel_list,
            kernel_size_list=model_cfg.kernel_size_list,
            stride_list=model_cfg.stride_list,
            padding_list=model_cfg.padding_list,
            groups_list=model_cfg.groups_list,
            dilation_list=model_cfg.dilation_list,
            conv_cfg=model_cfg.conv_cfg,
            linear_cfg=model_cfg.linear_cfg,
            device=device,
        ).to(device)
        model.set_encode_mode(model_cfg.encode_mode)
        model.reset_parameters(random_state)
    elif name == "Meta_CNNETE":
        model = eval(model_cfg.name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            input_wg_width=model_cfg.input_wg_width,
            input_wg_interval=model_cfg.input_wg_interval,
            feature_dim=model_cfg.feature_dim,
            num_classes=getattr(configs.dataset, "num_classes", None), #configs.dataset.num_classes,
            kernel_list=model_cfg.kernel_list,
            mid_channel_list=model_cfg.mid_channel_list,
            kernel_size_list=model_cfg.kernel_size_list,
            pool_out_size=model_cfg.pool_out_size,
            stride_list=model_cfg.stride_list,
            padding_list=model_cfg.padding_list,
            groups_list=model_cfg.groups_list,
            dilation_list=model_cfg.dilation_list,
            hidden_list=model_cfg.hidden_list,
            conv_cfg=model_cfg.conv_cfg,
            linear_cfg=model_cfg.linear_cfg,
            digital_norm_cfg=model_cfg.digital_norm_cfg,
            digital_act_cfg=model_cfg.digital_act_cfg,
            optical_norm_cfg=model_cfg.optical_norm_cfg,
            optical_act_cfg=model_cfg.optical_act_cfg,
            device=device,
            feature_extractor_type=model_cfg.feature_extractor_type,
            fft_mode_1=model_cfg.fft_mode_1,
            fft_mode_2=model_cfg.fft_mode_2,
            window_size=model_cfg.window_size,
            darcy=getattr(model_cfg, "darcy", False),
        ).to(device)
        model.set_encode_mode(model_cfg.encode_mode)
        model.reset_parameters(random_state)
    elif name == "Meta_CNNETE_1D":
        model = eval(model_cfg.name)(
            sequence_length=configs.dataset.n_input_features,
            input_wg_width=model_cfg.input_wg_width,
            input_wg_interval=model_cfg.input_wg_interval,
            in_channels=configs.dataset.in_channels,
            feature_dim=model_cfg.feature_dim,
            num_classes=configs.dataset.num_classes,
            kernel_list=model_cfg.kernel_list,
            mid_channel_list=model_cfg.mid_channel_list,
            kernel_size_list=model_cfg.kernel_size_list,
            pool_out_size=model_cfg.pool_out_size,
            stride_list=model_cfg.stride_list,
            padding_list=model_cfg.padding_list,
            groups_list=model_cfg.groups_list,
            dilation_list=model_cfg.dilation_list,
            hidden_list=model_cfg.hidden_list,
            conv_cfg=model_cfg.conv_cfg,
            linear_cfg=model_cfg.linear_cfg,
            digital_norm_cfg=model_cfg.digital_norm_cfg,
            digital_act_cfg=model_cfg.digital_act_cfg,
            optical_norm_cfg=model_cfg.optical_norm_cfg,
            optical_act_cfg=model_cfg.optical_act_cfg,
            device=device,
            feature_extractor_type=model_cfg.feature_extractor_type,
            fft_mode_1=model_cfg.fft_mode_1,
            fft_mode_2=model_cfg.fft_mode_2,
            window_size=model_cfg.window_size,
        ).to(device)
        model.set_encode_mode(model_cfg.encode_mode)
        model.reset_parameters(random_state)
    elif "meta_cnn_dno" in name.lower():
        model = eval(name)(
            in_channels=configs.dataset.in_channels,
            out_channels=configs.dataset.out_channels,
            kernel_list=model_cfg.kernel_list,
            mid_channel_list=model_cfg.mid_channel_list,
            kernel_size_list=model_cfg.kernel_size_list,
            pool_out_size=model_cfg.pool_out_size,
            stride_list=model_cfg.stride_list,
            padding_list=model_cfg.padding_list,
            groups_list=model_cfg.groups_list,
            dilation_list=model_cfg.dilation_list,
            hidden_list=model_cfg.hidden_list,
            conv_cfg=model_cfg.conv_cfg,
            linear_cfg=model_cfg.linear_cfg,
            norm_cfg=model_cfg.norm_cfg,
            act_cfg=model_cfg.act_cfg,
            prediction_kernel_list=model_cfg.prediction_kernel_list,
            prediction_kernel_size_list=model_cfg.prediction_kernel_size_list,
            prediction_stride_list=model_cfg.prediction_stride_list,
            prediction_padding_list=model_cfg.prediction_padding_list,
            prediction_dilation_list=model_cfg.prediction_dilation_list,
            prediction_groups_list=model_cfg.prediction_groups_list,
            prediction_conv_cfg=model_cfg.prediction_conv_cfg,
            prediction_norm_cfg=model_cfg.prediction_norm_cfg,
            prediction_act_cfg=model_cfg.prediction_act_cfg,
            device=device,
        ).to(device)
        model.set_encode_mode(model_cfg.encode_mode)
        model.reset_parameters(random_state)
    else:
        model = None
        raise NotImplementedError(f"Not supported model name: {model_cfg.name}")
    return model


def make_optimizer(params, name: str = None, opt_configs=None) -> Optimizer:
    opt_cfg = opt_configs or configs.optimizer
    if name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=opt_cfg.lr,
            momentum=opt_cfg.momentum,
            weight_decay=opt_cfg.weight_decay,
            nesterov=True,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=getattr(opt_cfg, "betas", (0.9, 0.999)),
        )
    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
        )
    elif name == "sam_sgd":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(opt_cfg, "rho", 0.5),
            adaptive=getattr(opt_cfg, "adaptive", True),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            momenum=0.9,
        )
    elif name == "sam_adam":
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(opt_cfg, "rho", 0.001),
            adaptive=getattr(opt_cfg, "adaptive", True),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
        )
    elif name == "dadapt_adam":
        optimizer = DAdaptAdam(
            params,
            lr=opt_cfg.lr,
            betas=getattr(opt_cfg, "betas", (0.9, 0.999)),
            weight_decay=opt_cfg.weight_decay,
        )
    elif name == "dadapt_sgd":
        optimizer = DAdaptSGD(
            params,
            lr=opt_cfg.lr,
            momentum=opt_cfg.momentum,
            weight_decay=opt_cfg.weight_decay,
        )
    elif name == "prodigy":
        optimizer = Prodigy(
            params,
            lr=opt_cfg.lr,
            betas=getattr(opt_cfg, "betas", (0.9, 0.999)),
            weight_decay=opt_cfg.weight_decay,
        )
    else:
        raise NotImplementedError(name)

    return optimizer


def make_scheduler(optimizer: Optimizer, name: str = None, cfgs=None) -> Scheduler:
    cfgs = cfgs or configs
    name = (name or cfgs.scheduler.name).lower()
    if name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1
        )
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfgs.run.n_epochs),
            eta_min=float(cfgs.scheduler.lr_min),
        )
    elif name == "cosine_warmup":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=cfgs.run.n_epochs,
            max_lr=cfgs.optimizer.lr,
            min_lr=cfgs.scheduler.lr_min,
            warmup_steps=int(cfgs.scheduler.warmup_steps),
        )
    elif name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=cfgs.scheduler.lr_gamma
        )
    elif name == "downsample_rate":
        scheduler = DownsampleRateScheduler(
            total_steps=cfgs.n_epochs, 
            init_ds_rate=cfgs.init_ds_rate, 
            final_ds_rate=cfgs.final_ds_rate, 
            available_ds_rate=cfgs.available_ds_rate, 
            mode=cfgs.mode, 
            milestone=cfgs.milestone,
        )
    elif name == "invdes_sharpness":
        scheduler = InvDesSharpnessScheduler(
            mode=cfgs.mode,
            num_train_epochs=cfgs.num_train_epochs,
            sharpness_peak_epoch=cfgs.sharpness_peak_epoch,
            sharpness_span_per_epoch=cfgs.sharpness_span_per_epoch,
            init_sharpness=cfgs.init_sharpness, 
            final_sharpness=cfgs.final_sharpness,
        )
    elif name == "end2end_sharpness":
        scheduler = End2EndSharpnessScheduler(
            mode="cosine",
            num_train_epochs=cfgs.num_train_epochs,
            init_sharpness=cfgs.init_sharpness,
            final_sharpness=cfgs.final_sharpness,
        )
    else:
        raise NotImplementedError(name)

    return scheduler


def make_criterion(name: str = None, cfg=None) -> nn.Module:
    name = (name or configs.criterion.name).lower()
    cfg = cfg or configs.criterion
    if name == "nll":
        criterion = nn.NLLLoss()
    elif name == "mse":
        criterion = nn.MSELoss()
    elif name == "mae":
        criterion = nn.L1Loss()
    elif name == "ce":
        criterion = nn.CrossEntropyLoss()
    elif name == "ce_smooth":
        criterion = CrossEntropyLossSmooth(
            label_smoothing=getattr(cfg, "label_smoothing", 0.1)
        )
    elif name == "adaptive":
        criterion = AdaptiveLossSoft(alpha_min=-1.0, alpha_max=1.0)
    elif name == "kd":
        criterion = KDLoss(
            T=getattr(cfg, "T", 2),
            ce_weight=getattr(cfg, "ce_weight", 0),
            kd_weight=getattr(cfg, "kd_weight", 0.9),
            logit_stand=getattr(cfg, "logit_stand", False),
        )
    elif name == "dkd":
        criterion = DKDLoss(
            T=getattr(cfg, "T", 2),
            ce_weight=getattr(cfg, "ce_weight", 0),
            kd_alpha=getattr(cfg, "kd_alpha", 1),
            kd_beta=getattr(cfg, "kd_beta", 1),
            logit_stand=getattr(cfg, "logit_stand", False),
        )
    elif name == "energy_conservation":
        criterion = EnergyConservationLoss(
            loss_coeff=getattr(cfg, "loss_coeff", 0.9),
        )
    elif name == "nl2norm":
        criterion = NL2norm()
    elif name == "distance_constraint":
        raise NotImplementedError("Distance constraint loss is not implemented yet.")
        criterion = NL2norm()
    elif name == "tmmatching":
        criterion = TransferMatrixMatchingLoss()
    elif name == "responsematching":
        num_modes = getattr(cfg, "num_modes", 15)
        criterion = ResponseMatchingLoss(
            probe_source_mode=getattr(cfg, "probe_source_mode", "fourier"),
            num_modes=num_modes,
            num_random_sources = getattr(cfg, "num_random_sources", num_modes * 2),
        )
    elif name == "smooth_penalty":
        criterion = TransferMatrixSmoothError(
            mode=getattr(cfg, "mode", "diag"),
        )
    elif name == "activation_smooth":
        criterion = HighFreqPenalty(
            mode_threshold=getattr(cfg, "mode_threshold", 8),
        )
    elif name == "cosine_similarity":
        criterion = CosSimLoss()
    elif name == "admm_consistency":
        criterion = AdmmConsistencyLoss(
            rho=getattr(cfg, "rho_admm", 1),
        )
    else:
        raise NotImplementedError(name)
    return criterion
