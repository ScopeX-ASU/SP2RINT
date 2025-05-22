#!/usr/bin/env python
# coding=UTF-8
import argparse
import os
from typing import Callable, Dict, Iterable

import mlflow
import torch
import torch.cuda.amp as amp
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from pyutils.config import configs
from pyutils.general import AverageMeter
from pyutils.general import logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler

from core import builder
from core.datasets.mixup import MixupAll
from core.utils import get_parameter_group

# from core.utils import CurlLoss, PoyntingLoss, plot_compare
# from core.losses import DarcyLoss


def legalize_perm(model, permutation_params):
    """Stochastic permutation legalization (SPL)

    Args:
        model (_type_): _description_
        area_loss_func (Callable): _description_
    """

    optimizer = builder.make_optimizer(
        permutation_params,
        name=configs.optimizer.name,
        opt_configs=configs.permutation.optimizer,
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=2e-4)
    scheduler = builder.make_scheduler(optimizer)
    lg.info("Force to legalize permutation")
    for step in range(configs.permutation.n_epochs):
        # # Build rotation mask
        # model.build_rotation_mask(mode=configs.arch_optimizer.rot_mask_)
        optimizer.zero_grad()
        alm_perm_loss = model.get_swap_alm_loss(rho=1e-7)
        loss = alm_perm_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.update_swap_alm_multiplier(rho=1e-7)
        with torch.no_grad():
            if step % 200 == 0:
                legal = model.check_perm()
                perm_loss = model.get_swap_loss()
                lg.info(f"Step: {step}, Perm Loss: {perm_loss}, Perm legality: {legal}")
    legal = model.check_perm()
    lg.info("Legalize permutation...")
    model.sinkhorn_perm(n_step=200, t_min=0.005, noise_std=0.01)
    legal = model.check_perm()
    lg.info(f"Final perm legality: {legal}...")
    if legal:
        lg.info("All permutations are legal")
    else:
        lg.info("Not all permutations are legal!")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    aux_criterions: Dict,
    mixup_fn: Callable = None,
    device: torch.device = torch.device("cuda:0"),
    plot: bool = False,
    grad_scaler=None,
) -> None:
    model.train()
    step = epoch * len(train_loader)

    mse_meter = AverageMeter("mse")
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}
    aux_output_weight = getattr(configs.criterion, "aux_output_weight", 0)

    # poynting_loss = PoyntingLoss(configs.model.grid_step, wavelength=1.55)
    data_counter = 0
    total_data = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        data_counter += data.shape[0]

        target = target.to(device, non_blocking=True)
        if mixup_fn is not None:
            data, target = mixup_fn(data, target)

        with amp.autocast(enabled=grad_scaler._enabled):
            output = model(data)

            if type(output) is tuple:
                output, aux_output = output
            else:
                aux_output = None
            regression_loss = criterion(output, target)
            mse_meter.update(regression_loss.item())
            loss = regression_loss
            for name, config in aux_criterions.items():
                aux_criterion, weight = config
                if name == "curl_loss":
                    fields = torch.cat([target[:, 0:1]], output, target[:, 2:3], dim=1)
                    aux_loss = weight * aux_criterion(fields, data[:, 0:1])
                elif name == "tv_loss":
                    aux_loss = weight * aux_criterion(output, target)
                elif name == "poynting_loss":
                    aux_loss = weight * aux_criterion(output, target)
                elif name == "darcy_loss":
                    loss = 0
                    aux_loss = weight * aux_criterion(
                        output, data[:, 0:1]
                    )  # remove pos encoding if any
                loss = loss + aux_loss
                aux_meters[name].update(aux_loss.item())
            # TODO aux output loss
            if aux_output is not None and aux_output_weight > 0:
                aux_output_loss = aux_output_weight * F.mse_loss(
                    aux_output, target.abs()
                )  # field magnitude learning
                loss = loss + aux_output_loss
            else:
                aux_output_loss = None

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        step += 1

        if batch_idx % int(configs.run.log_interval) == 0:
            log = "Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4e} Regression Loss: {:.4e}".format(
                epoch,
                data_counter,
                total_data,
                100.0 * data_counter / total_data,
                loss.data.item(),
                regression_loss.data.item(),
            )
            for name, aux_meter in aux_meters.items():
                log += f" {name}: {aux_meter.val:.4e}"
            if aux_output_loss is not None:
                log += f" aux_output_loss: {aux_output_loss.item()}"
            lg.info(log)

            mlflow.log_metrics({"train_loss": loss.item()}, step=step)
        # break
    scheduler.step()
    avg_regression_loss = mse_meter.avg
    lg.info(f"Train Regression Loss: {avg_regression_loss:.4e}")
    mlflow.log_metrics(
        {"train_regression": avg_regression_loss, "lr": get_learning_rate(optimizer)},
        step=epoch,
    )


def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = True,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    with torch.no_grad():
        for i, (data, target) in enumerate(validation_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if mixup_fn is not None:
                data, target = mixup_fn(data, target, random_state=i, vflip=False)

            output = model(data)

            val_loss = criterion(output, target)
            mse_meter.update(val_loss.item())

    loss_vector.append(mse_meter.avg)

    lg.info("\nValidation set: Average loss: {:.4e}\n".format(mse_meter.avg))
    mlflow.log_metrics({"val_loss": mse_meter.avg}, step=epoch)


def test(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = False,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if mixup_fn is not None:
                data, target = mixup_fn(
                    data, target, random_state=i + 10000, vflip=False
                )

            output = model(data)
            # print(output.shape)

            val_loss = criterion(output, target)
            mse_meter.update(val_loss.item())

    loss_vector.append(mse_meter.avg)

    lg.info("\nTest set: Average loss: {:.4e}\n".format(mse_meter.avg))
    mlflow.log_metrics({"test_loss": mse_meter.avg}, step=epoch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if bool(configs.run.deterministic):
        set_torch_deterministic(int(configs.run.random_state))

    train_loader, validation_loader, test_loader = builder.make_dataloader(
        splits=["train", "valid", "test"]
    )

    if (
        configs.run.do_distill
        and configs.teacher is not None
        and os.path.exists(configs.teacher.checkpoint)
    ):
        teacher = builder.make_model(device, model_cfg=configs.teacher)
        load_model(teacher, path=configs.teacher.checkpoint)
        teacher.eval()
        lg.info(f"Load teacher model from {configs.teacher.checkpoint}")
    else:
        teacher = None

    model = builder.make_model(
        device,
        model_cfg=configs.model,
        random_state=(
            int(configs.run.random_state) if int(configs.run.deterministic) else None
        ),
    )
    lg.info(model)

    model(next(iter(test_loader))[0].to(device))

    optimizer = builder.make_optimizer(
        get_parameter_group(model, weight_decay=float(configs.optimizer.weight_decay)),
        name=configs.optimizer.name,
        opt_configs=configs.optimizer,
    )
    scheduler = builder.make_scheduler(optimizer)
    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(
        device
    )
    if configs.aux_criterion is not None:
        aux_criterions = {
            name: [builder.make_criterion(name, cfg=config), float(config.weight)]
            for name, config in configs.aux_criterion.items()
            if float(config.weight) > 0
        }
    else:
        aux_criterions = {}
    print(aux_criterions)
    mixup_config = configs.dataset.augment
    if mixup_config is not None:
        mixup_fn = MixupAll(**mixup_config)
        test_mixup_fn = MixupAll(**configs.dataset.test_augment)
    else:
        mixup_fn = test_mixup_fn = None
    saver = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=False,
        truncate=4,
        metric_name="err",
        format="{:.4f}",
    )

    grad_scaler = amp.GradScaler(enabled=getattr(configs.run, "fp16", False))
    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    mlflow.set_experiment(configs.run.experiment)
    experiment = mlflow.get_experiment_by_name(configs.run.experiment)

    # run_id_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mlflow.start_run(run_name=model_name)
    mlflow.log_params(
        {
            "exp_name": configs.run.experiment,
            "exp_id": experiment.experiment_id,
            "run_id": mlflow.active_run().info.run_id,
            "init_lr": configs.optimizer.lr,
            "checkpoint": checkpoint,
            "restore_checkpoint": configs.checkpoint.restore_checkpoint,
            "pid": os.getpid(),
        }
    )

    lossv, accv = [0], [0]
    epoch = 0
    try:
        lg.info(
            f"Experiment {configs.run.experiment} ({experiment.experiment_id}) starts. Run ID: ({mlflow.active_run().info.run_id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})"
        )
        lg.info(configs)
        if (
            int(configs.checkpoint.resume)
            and len(configs.checkpoint.restore_checkpoint) > 0
        ):
            load_model(
                model,
                configs.checkpoint.restore_checkpoint,
                ignore_size_mismatch=int(configs.checkpoint.no_linear),
            )

            lg.info("Validate resumed model...")
            test(model, validation_loader, 0, criterion, lossv, accv, False, device)

        for epoch in range(1, int(configs.run.n_epochs) + 1):
            train(
                model,
                train_loader,
                optimizer,
                scheduler,
                epoch,
                criterion,
                aux_criterions,
                mixup_fn,
                device,
                # plot=configs.plot.train,
                grad_scaler=grad_scaler,
            )

            if validation_loader is not None:
                validate(
                    model,
                    validation_loader,
                    epoch,
                    criterion,
                    lossv,
                    accv,
                    device,
                    mixup_fn=test_mixup_fn,
                    # plot=configs.plot.valid,
                )
            if epoch > int(configs.run.n_epochs) - 21:
                test(
                    model,
                    test_loader,
                    epoch,
                    criterion,
                    [],
                    [],
                    device,
                    mixup_fn=test_mixup_fn,
                    # plot=configs.plot.test,
                )
                saver.save_model(
                    model,
                    lossv[-1],
                    epoch=epoch,
                    path=checkpoint,
                    save_model=False,
                    print_msg=True,
                )
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
