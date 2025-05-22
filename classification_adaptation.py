#!/usr/bin/env python
# coding=UTF-8
import argparse
import os
from typing import Callable, Dict, Iterable, List, Optional

import mlflow
import torch
import torch.cuda.amp as amp
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
from core.utils import get_parameter_group, register_hidden_hooks


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
    grad_scaler: Optional[Callable] = None,
    teacher: Optional[nn.Module] = None,
    permutation_params: Optional[List] = None,
    model_params: Optional[List] = None,
) -> None:
    model.train()
    step = epoch * len(train_loader)
    init_T = float(getattr(configs.model.conv_cfg, "gumbel_T", 5))
    gamma_T = float(getattr(configs.model.conv_cfg, "gumbel_decay_rate", 0.956))
    force_perm_legal_epoch = int(
        getattr(configs.permutation, "force_perm_legal_epoch", 60)
    )
    perm_loss_rho = float(getattr(configs.permutation, "perm_loss_rho", 0))
    perm_loss_rho_gamma = float(getattr(configs.permutation, "perm_loss_rho_gamma", 1))
    warm_up_epoch = int(getattr(configs.permutation, "warm_up_epoch", 10))
    max_lambda = float(getattr(configs.permutation, "max_lambda", 1))
    class_meter = AverageMeter("ce")
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}

    if epoch >= warm_up_epoch:
        perm_loss_rho = perm_loss_rho * perm_loss_rho_gamma ** (epoch - warm_up_epoch)
        lg.info(f"Permutation ALM Rho: {perm_loss_rho}")

    T = init_T * gamma_T ** (epoch - 1)
    model.set_gumbel_temperature(T)
    lg.info(f"Gumbel temperature: {T:.4f}/{init_T}")

    if epoch == force_perm_legal_epoch and configs.model.conv_cfg.swap_mode != "fixed":
        param_status_change(permutation_params, True)
        legalize_perm(model, permutation_params)

    data_counter = 0
    correct = 0
    total_data = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        data_counter += data.shape[0]

        target = target.to(device, non_blocking=True)
        if mixup_fn is not None:
            data, target = mixup_fn(data, target)

        with amp.autocast(enabled=grad_scaler._enabled):
            output = model(data)
            class_loss = criterion(output, target)
            class_meter.update(class_loss.item())
            loss = class_loss

            for name, config in aux_criterions.items():
                aux_criterion, weight = config
                aux_loss = 0
                if name in {"kd", "dkd"} and teacher is not None:
                    with torch.no_grad():
                        teacher_scores = teacher(data).data.detach()
                    aux_loss = weight * aux_criterion(output, teacher_scores, target)
                elif name == "mse_distill" and teacher is not None:
                    with torch.no_grad():
                        teacher(data).data.detach()
                    teacher_hiddens = [
                        m._recorded_hidden
                        for m in teacher.modules()
                        if hasattr(m, "_recorded_hidden")
                    ]
                    student_hiddens = [
                        m._recorded_hidden
                        for m in model.modules()
                        if hasattr(m, "_recorded_hidden")
                    ]

                    aux_loss = weight * sum(
                        F.mse_loss(h1, h2)
                        for h1, h2 in zip(teacher_hiddens, student_hiddens)
                    )
                loss = loss + aux_loss
                aux_meters[name].update(aux_loss)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).sum().item()

        param_status_change(permutation_params, True)
        if (
            epoch >= warm_up_epoch and perm_loss_rho > 0
        ):  # only train permutation in permutation phase; no constraints in warmup
            alm_perm_loss = model.get_swap_alm_loss(rho=perm_loss_rho)
            loss = loss + alm_perm_loss
        with torch.no_grad():
            perm_loss = model.get_swap_loss()

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        if configs.run.grad_clip:
            torch.nn.utils.clip_grad_value_(
                [p for p in model.parameters() if p.requires_grad],
                float(configs.run.max_grad_value),
            )
        grad_scaler.step(optimizer)
        grad_scaler.update()
        step += 1

        model.phase_rounding()
        model.update_lambda_pixel_size()

        if epoch >= warm_up_epoch and perm_loss_rho > 0:
            model.update_swap_alm_multiplier(perm_loss_rho * 0.1, max_lambda=max_lambda)

        if batch_idx % int(configs.run.log_interval) == 0:
            log = "Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4e} perm loss: {} alm multiplier: {} class Loss: {:.4e}".format(
                epoch,
                data_counter,
                total_data,
                100.0 * data_counter / total_data,
                loss.data.item(),
                perm_loss,
                model.get_alm_multiplier(),
                class_loss.data.item(),
            )
            for name, aux_meter in aux_meters.items():
                log += f" {name}: {aux_meter.val:.4e}"
            lg.info(log)

            mlflow.log_metrics({"train_loss": loss.item()}, step=step)

    scheduler.step()
    lg.info(model.meta_params.build_swap_permutation())

    avg_class_loss = class_meter.avg
    accuracy = 100.0 * correct / total_data
    lg.info(
        f"Train class Loss: {avg_class_loss:.4e}, Accuracy: {correct}/{total_data} ({accuracy:.2f}%)"
    )
    mlflow.log_metrics(
        {
            "train_class": avg_class_loss,
            "train_acc": accuracy,
            "lr": get_learning_rate(optimizer),
        },
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
    fp16: bool = False,
) -> None:
    model.eval()
    val_loss = 0
    correct = 0
    class_meter = AverageMeter("ce")
    with amp.autocast(enabled=fp16):
        with torch.no_grad():
            for i, (data, target) in enumerate(validation_loader):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if mixup_fn is not None:
                    data, target = mixup_fn(data, target, random_state=i, vflip=False)

                output = model(data)

                val_loss = criterion(output, target)
                class_meter.update(val_loss.item())
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

    loss_vector.append(class_meter.avg)
    accuracy = 100.0 * correct / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    lg.info(
        f"\nValidation set: Average loss: {class_meter.avg:.4e}, Accuracy: {correct}/{len(validation_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    mlflow.log_metrics({"val_loss": class_meter.avg, "val_acc": accuracy}, step=epoch)


def test(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    fp16: bool = False,
) -> None:
    model.eval()
    val_loss = 0
    correct = 0
    class_meter = AverageMeter("ce")
    with amp.autocast(enabled=fp16):
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if mixup_fn is not None:
                    data, target = mixup_fn(
                        data, target, random_state=i + 10000, vflip=False
                    )

                output = model(data)

                val_loss = criterion(output, target)
                class_meter.update(val_loss.item())
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

    loss_vector.append(class_meter.avg)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    accuracy_vector.append(accuracy)

    lg.info(
        f"\nTest set: Average loss: {class_meter.avg:.4e}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )

    mlflow.log_metrics({"test_loss": class_meter.avg, "test_acc": accuracy}, step=epoch)


def param_status_change(parameter_groups, status: bool):
    for group in parameter_groups:
        group_params = group["params"]
        for param in group_params:
            if hasattr(param, "requires_grad"):
                param.requires_grad = status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    lg.info(configs)

    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if bool(configs.run.deterministic):
        set_torch_deterministic()

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

    # print(next(iter(test_loader))[0].shape)
    ## dummy forward to initialize quantizer
    model(next(iter(test_loader))[0].to(device))

    # model.set_weight_train(configs.model.conv_cfg.weight_train)

    if configs.model.conv_cfg.swap_mode != "fixed":
        parameter_groups, permutation_params = get_parameter_group(
            model, weight_decay=float(configs.optimizer.weight_decay)
        )
    else:
        parameter_groups = get_parameter_group(
            model, weight_decay=float(configs.optimizer.weight_decay)
        )
        permutation_params = []

    # print(special_params)
    # # print(parameter_groups)

    # # new_parameter_groups, special_params = separate_special_params(parameter_groups)

    # # print(special_params)

    # exit(0)
    optimizer = builder.make_optimizer(
        parameter_groups,
        name=configs.optimizer.name,
        opt_configs=configs.optimizer,
    )

    # exit(0)
    scheduler = builder.make_scheduler(optimizer)
    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(
        device
    )

    aux_criterions = dict()
    if configs.aux_criterion is not None:
        for name, config in configs.aux_criterion.items():
            if float(config.weight) > 0:
                try:
                    fn = builder.make_criterion(name, cfg=config)
                except NotImplementedError:
                    fn = name
                aux_criterions[name] = [fn, float(config.weight)]
    print(aux_criterions)
    if "mse_distill" in aux_criterions and teacher is not None:
        ## register hooks for teacher and student
        register_hidden_hooks(teacher)
        register_hidden_hooks(model)
        print(len([m for m in teacher.modules() if hasattr(m, "_recorded_hidden")]))
        print(len([m for m in teacher.modules() if hasattr(m, "_recorded_hidden")]))
        print("Register hidden state hooks for teacher and students")

    mixup_config = configs.dataset.augment
    mixup_fn = MixupAll(**mixup_config) if mixup_config is not None else None
    test_mixup_fn = (
        MixupAll(**configs.dataset.test_augment) if mixup_config is not None else None
    )
    saver = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=True,
        truncate=2,
        metric_name="acc",
        format="{:.2f}",
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
            test(
                model,
                validation_loader,
                0,
                criterion,
                lossv,
                accv,
                device,
                fp16=grad_scaler._enabled,
            )
            model.reset_linear_parameters()
            lg.info("Linear parameters reset")
        if teacher is not None:
            test(
                teacher,
                validation_loader,
                0,
                criterion,
                [],
                [],
                device,
                fp16=grad_scaler._enabled,
            )
            lg.info("Map teacher to student...")
            if hasattr(model, "load_from_teacher"):
                with amp.autocast(grad_scaler._enabled):
                    model.load_from_teacher(teacher)

        ## compile models
        if getattr(configs.run, "compile", False):
            model = torch.compile(model)
            if teacher is not None:
                teacher = torch.compile(teacher)

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
                grad_scaler=grad_scaler,
                teacher=teacher,
                permutation_params=permutation_params,
                model_params=parameter_groups,
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
                    fp16=grad_scaler._enabled,
                )
            test(
                model,
                test_loader,
                epoch,
                criterion,
                lossv if validation_loader is None else [],
                accv if validation_loader is None else [],
                device,
                mixup_fn=test_mixup_fn,
                fp16=grad_scaler._enabled,
            )
            saver.save_model(
                getattr(model, "_orig_mod", model),  # remove compiled wrapper
                accv[-1],
                epoch=epoch,
                path=checkpoint,
                save_model=False,
                print_msg=True,
            )
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
