#!/usr/bin/env python
# coding=UTF-8
import argparse
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)

from typing import Callable, Dict, Iterable, Optional

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
from thirdparty.MAPS_local.unitest.test_patch_metalens_match_TF import PatchMetalens, response_matching_loss
import csv
from thirdparty.MAPS_local.core.utils import SharpnessScheduler
from pyutils.general import ensure_dir
from matplotlib import pyplot as plt
import h5py
sys.path.pop(0)
def project_to_implementable_subspace(model_trained, model_test, patched_metalens, out_epoch, device):
    '''
    there is not reparameterization of the transfer matrix of the metasurface during the DONN training,
    so we cannot ensure that the transfer matrix from DONN is implementable in the real world,
    for example, transfer matrix is not unitary, which means that it is not energy-conserved.

    in this function, we use the inverse design to project the transfer matrix of the metasurface trained in DONN to the implementable subspace
    we can actually get the real-world width of the metaatom array that corresponds to the required DONN transfer matrix"
    '''
    for i in range(configs.model_trained.conv_cfg.path_depth):
        # first, we need to read the transfer matrix of the metasurface trained in DONN
        A = model_trained.state_dict()[f"features.conv1.conv._conv_pos.metalens.{i}_1.W_buffer"]
        print("this is the shape of the transfer matrix", A.shape)

        target_response = A
        target_response_normalizer = torch.max(target_response.abs())
        target_response = target_response / target_response_normalizer
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        im0 = ax[0].imshow(
            target_response.abs().cpu().numpy()
        )
        ax[0].set_title("Target Magnitude")
        fig.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(
            torch.angle(target_response).cpu().numpy()
        )
        ax[1].set_title("Target Phase")
        fig.colorbar(im1, ax=ax[1])
        plt.savefig(f"./figs/TM_trained_{i}.png")
        plt.close()
        # -----------------------------------------------
        target_phase_response = torch.angle(target_response)
        diag_target_phase = torch.diag(target_phase_response)

        # second, we use inverse design to match the transfer matrix of the metasurface trained in DONN to the implementable subspace
        # maybe we need to reduce the epochs of the inverse design to save time
        patched_metalens.set_target_phase_response(target_phase_response)
        patched_metalens.rebuild_param()
        
        sources = torch.eye(480, device=device)
        with torch.no_grad():
            _ = patched_metalens.forward(256)
            full_wave_response = torch.zeros((480, 480), device=device, dtype=torch.complex128)
            for idx in range(480):
                source_i = sources[idx].repeat_interleave(int(1))
                source_zero_padding = torch.zeros(int(0.5 * 50), device=device)
                source_i = torch.cat([source_zero_padding, source_i, source_zero_padding])
                custom_source = dict(
                    source=source_i,
                    slice_name="in_slice_1",
                    mode="Hz1",
                    wl=0.85,
                    direction="x+",
                )
                _ = patched_metalens.total_opt(
                    sharpness=256, 
                    ls_knots={"design_region_0": patched_metalens.level_set_knots.unsqueeze(0)},
                    custom_source=custom_source
                )
                if idx == 0:
                    patched_metalens.total_opt.plot(
                        plot_filename=f"total_metalens_conv_{i}.png",
                        eps_map=patched_metalens.total_opt._eps_map,
                        obj=None,
                        field_key=("in_slice_1", 0.85, "Hz1", 300),
                        field_component="Hz",
                        in_slice_name="in_slice_1",
                        exclude_slice_names=[],
                    )
                    quit()
                response = patched_metalens.total_opt.objective.response[('in_slice_1', 'nearfield_1', 0.85, "Hz1", 300)]["fz"].squeeze()
                # response = response[int(0.3 * 50) // 2 :: int(0.3 * 50)]
                # assert len(response) == 32, f"{32}!={len(response)}"
                full_wave_response[idx] = response
            full_wave_response = full_wave_response.transpose(0, 1)
            full_wave_response = full_wave_response / torch.max(full_wave_response.abs()) # * target_response_normalizer
            full_wave_response_phase = torch.angle(full_wave_response)
            diag_fw_phase = torch.diag(full_wave_response_phase)[15//2::15]
            assert len(diag_fw_phase) == 32, f"{32}!={len(diag_fw_phase)}"
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            im0 = ax[0].imshow(
                full_wave_response.abs().cpu().numpy()
            )
            ax[0].set_title("Full Magnitude")
            fig.colorbar(im0, ax=ax[0])
            im1 = ax[1].imshow(
                full_wave_response_phase.cpu().numpy()
            )
            ax[1].set_title("Full Phase")
            fig.colorbar(im1, ax=ax[1])
            plt.savefig(f"./figs/TM_full_{i}.png")
            plt.close()

            plt.figure()
            plt.plot(diag_target_phase.detach().cpu().numpy(), label="target")
            plt.plot(diag_fw_phase.detach().cpu().numpy(), label="probed")
            plt.legend()
            plt.savefig(f"./figs/phase_comparison_{i}.png")
            plt.close()
        # the shape of the full wave response should be (480, 480)
        # -----------------------------------------------
        model_test.features.conv1.conv._conv_pos.metalens[f"{i}_1"].set_param_from_target_matrix(full_wave_response)
        # model_test.features.conv1.conv._conv_pos.metalens[f"{i}_1"].set_param_from_target_matrix(target_response * target_response_normalizer)

        # figure, ax = plt.subplots(1, 4, figsize=(20, 5))
        # im0 = ax[0].imshow(
        #     target_response.abs().cpu().numpy()
        # )
        # ax[0].set_title("Target Magnitude")
        # figure.colorbar(im0, ax=ax[0])
        # im1 = ax[1].imshow(
        #     stitched_response.detach().abs().cpu().numpy()
        # )
        # ax[1].set_title("Stitched Magnitude")
        # figure.colorbar(im1, ax=ax[1])
        # im2 = ax[2].imshow(
        #     full_wave_response.abs().cpu().numpy()
        # )
        # ax[2].set_title("Full Magnitude")
        # figure.colorbar(im2, ax=ax[2])
        # im3 = ax[3].imshow(
        #     (full_wave_response - target_response).abs().cpu().numpy()
        # )
        # ax[3].set_title("Difference Magnitude")
        # figure.colorbar(im3, ax=ax[3])
        # plt.savefig(configs.plot.plot_root + f"epoch-{out_epoch}_convid-{i}.png")
        # plt.close()
    return None


def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    aux_criterions: Dict,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    fp16: bool = False,
    stitched_response: torch.Tensor = None,
) -> None:
    model.eval()
    val_loss = 0
    correct = 0
    class_meter = AverageMeter("ce")
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}
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

                for name, config in aux_criterions.items():
                    aux_criterion, weight = config
                    aux_loss = 0
                    if name == "distance_constraint": # this prevent the wegiths from being too faraway from the initial weights
                        assert stitched_response is not None
                        aux_loss = weight * aux_criterion(stitched_response, model.features.conv1.conv._conv_pos.equivalent_W)
                    else:
                        raise NotImplementedError
                    val_loss = val_loss + aux_loss
                    aux_meters[name].update(aux_loss)

    loss_vector.append(class_meter.avg)
    accuracy = 100.0 * correct / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    log = f"\nValidation set: Average loss: {class_meter.avg:.4e}, Accuracy: {correct}/{len(validation_loader.dataset)} ({accuracy:.2f}%)\n"
    for name, aux_meter in aux_meters.items():
        log += f" {name}: {aux_meter.avg:.4e}"
    lg.info(
        log
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
    stitched_response: torch.Tensor = None,
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
    # initialize the stitched_response
    with h5py.File("/home/pingchua/projects/MAPS/figs/metalens_TF_uniform_wl-0.85_p-0.3_mat-Si/transfer_matrix.h5", "r") as f:
        transfer_matrix = f["transfer_matrix"][:]
        transfer_matrix = torch.tensor(transfer_matrix, dtype=torch.complex64)
        transfer_matrix = transfer_matrix / torch.max(transfer_matrix.abs())
    stitched_response = [transfer_matrix, transfer_matrix]
    stitched_response = torch.stack(stitched_response, dim=0).to(device)

    model_trained = builder.make_model(
        device,
        model_cfg=configs.model_trained,
        random_state=(
            int(configs.run.random_state) if int(configs.run.deterministic) else None
        ),
    )

    model_test = builder.make_model(
        device,
        model_cfg=configs.model_test,
        random_state=(
            int(configs.run.random_state) if int(configs.run.deterministic) else None
        ),
    )
    lg.info(model_trained)

    # print(next(iter(test_loader))[0].shape)
    ## dummy forward to initialize quantizer
    model_test.set_test_mode()
    model_trained(next(iter(test_loader))[0].to(device))
    model_test(next(iter(test_loader))[0].to(device))
    # -----------------------------------------------
    # build the patch metalens for inverse projection
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
    patch_metalens = PatchMetalens(
        atom_period=0.3,
        patch_size=17,
        num_atom=32,
        probing_region_size=17,
        target_phase_response=None,
        LUT=LUT,
        device=device,
        target_dx=0.3,
        plot_root=configs.plot.plot_root,
    )
    ensure_dir(configs.plot.plot_root)
    # -----------------------------------------------

    # model.set_weight_train(configs.model.conv_cfg.weight_train)

    optimizer = builder.make_optimizer(
        get_parameter_group(model_trained, weight_decay=float(configs.optimizer.weight_decay)),
        name=configs.optimizer.name,
        opt_configs=configs.optimizer,
    )
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
        raise NotImplementedError
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
    lg.info(f"Number of parameters: {count_parameters(model_trained)}")

    model_name = f"{configs.model_trained.name}"
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
                model_trained,
                configs.checkpoint.restore_checkpoint,
                # ignore_size_mismatch=int(configs.checkpoint.no_linear),
                ignore_size_mismatch=False,
            )
            lg.info("Done loading the trained model")
            load_model(
                model_test,
                configs.checkpoint.restore_checkpoint,
                # ignore_size_mismatch=int(configs.checkpoint.no_linear),
                ignore_size_mismatch=True,
            ) # the transfer matrix in this model will cuase mismatch, so we need to measure the transfer matrix and write it to the model manually
            lg.info("Done loading weight for the test mode")
            lg.info("Validate resumed model...")
            # before we start the training, we need to validate the model
            test(
                model_trained,
                test_loader,
                0,
                criterion,
                lossv,
                accv,
                device,
                fp16=grad_scaler._enabled,
            )

        stitched_response = project_to_implementable_subspace(model_trained, model_test, patch_metalens, epoch, device)

        test(
            model_test,
            test_loader,
            0,
            criterion,
            lossv,
            accv,
            device,
            fp16=grad_scaler._enabled,
        )

        quit()

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

            # if validation_loader is not None:
            #     validate(
            #         model,
            #         validation_loader,
            #         epoch,
            #         criterion,
            #         aux_criterions,
            #         [],
            #         [],
            #         device,
            #         mixup_fn=test_mixup_fn,
            #         fp16=grad_scaler._enabled,
            #         stitched_response=stitched_response,
            #     )
            # test(
            #     model,
            #     test_loader,
            #     epoch,
            #     criterion,
            #     [],
            #     [],
            #     device,
            #     mixup_fn=test_mixup_fn,
            #     fp16=grad_scaler._enabled,
            #     stitched_response=stitched_response
            # )
            # stitched_response = project_to_implementable_subspace(model, patch_metalens, epoch, device)
            if validation_loader is not None:
                validate(
                    model,
                    validation_loader,
                    epoch,
                    criterion,
                    aux_criterions,
                    lossv,
                    accv,
                    device,
                    mixup_fn=test_mixup_fn,
                    fp16=grad_scaler._enabled,
                    stitched_response=stitched_response,
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
                stitched_response=stitched_response
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
