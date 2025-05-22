#!/usr/bin/env python
# coding=UTF-8
import argparse
import os
import sys
import psutil
from typing import Callable, Dict, Iterable, Optional, List
import re
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
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core import builder
from core.datasets.mixup import MixupAll
from core.utils import (
    get_parameter_group, 
    register_hidden_hooks, 
    probe_near2far_matrix, 
    CosSimLoss,
    reset_optimizer_and_scheduler,
    DeterministicCtx,
    insert_zeros_after_every_N_except_last,
)
from core.models.patch_metalens import PatchMetalens
import csv
from thirdparty.MAPS_old.core.utils import SharpnessScheduler
from thirdparty.MAPS_old.core.fdfd.pardiso_solver import pardisoSolver
sys.path.pop()
from pyutils.general import ensure_dir
from matplotlib import pyplot as plt
import h5py
import copy
import numpy as np
import datetime
import psutil
import torch
import random
import wandb


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
    plot: bool = False,
    test_train_loader: bool = False,
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
                if isinstance(output, tuple):
                    output, inner_fields = output
                val_loss = criterion(output, target)
                # print("this is the criterion: ", criterion, flush=True)
                # print("this is the output: ", output[0], flush=True)
                # print("this is the target: ", target[0], flush=True)
                # print("this is the loss: ", val_loss, flush=True)
                # quit()
                class_meter.update(val_loss.item())
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

    loss_vector.append(class_meter.avg)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    accuracy_vector.append(accuracy)

    if not test_train_loader:
        lg.info(
            f"\nTest set: Average loss: {class_meter.avg:.4e}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
        )
        wandb.log(
            {
                "test_loss": class_meter.avg, 
                "test_acc": accuracy,
                "epoch": epoch,
            }
        )
    else:
        lg.info(
            f"\nFeasible train set: Average loss: {class_meter.avg:.4e}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
        )
        wandb.log(
            {
                "train_loss": class_meter.avg,
                "train_acc": accuracy,
                "epoch": epoch,
            }
        )

    if plot:
        for i in range(configs.model.conv_cfg.path_depth):
            # first, we need to read the transfer matrix of the metasurface trained in DONN
            if model.state_dict().get(f"features.conv1.conv._conv_pos.metalens.{i}_17.W_buffer") is not None:
                A = model.state_dict()[f"features.conv1.conv._conv_pos.metalens.{i}_17.W_buffer"]
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                im0 = ax[0].imshow(
                    A.abs().cpu().numpy()
                )
                ax[0].set_title("Magnitude")
                fig.colorbar(im0, ax=ax[0])
                im1 = ax[1].imshow(
                    torch.angle(A).cpu().numpy()
                )
                ax[1].set_title("Phase")
                fig.colorbar(im1, ax=ax[1])
                plt.savefig(configs.plot.plot_root + f"epoch-{epoch}_convid-{i}_test.png")
                plt.close()

    if test_train_loader:
        return class_meter.avg, accuracy



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if configs.model.linear_system:
        configs.model.optical_norm_cfg = None
        configs.model.optical_act_cfg = None

    if configs.model_test.linear_system:
        configs.model_test.optical_norm_cfg = None
        configs.model_test.optical_act_cfg = None

    if configs.model.hidden_channel_1 != 0 or configs.model.hidden_channel_2 != 0 or configs.model.hidden_channel_3 != 0:
        hidden_list = []
        if configs.model.hidden_channel_1 != 0:
            hidden_list.append(configs.model.hidden_channel_1)
        if configs.model.hidden_channel_2 != 0:
            hidden_list.append(configs.model.hidden_channel_2)
        if configs.model.hidden_channel_3 != 0:
            hidden_list.append(configs.model.hidden_channel_3)
        configs.model.hidden_list = hidden_list

    if configs.model_test.hidden_channel_1 != 0 or configs.model_test.hidden_channel_2 != 0 or configs.model_test.hidden_channel_3 != 0:
        hidden_list = []
        if configs.model_test.hidden_channel_1 != 0:
            hidden_list.append(configs.model_test.hidden_channel_1)
        if configs.model_test.hidden_channel_2 != 0:
            hidden_list.append(configs.model_test.hidden_channel_2)
        if configs.model_test.hidden_channel_3 != 0:
            hidden_list.append(configs.model_test.hidden_channel_3)
        configs.model_test.hidden_list = hidden_list
    assert configs.invdes.field_norm_condition == "wo_lens", "Only wo_lens is supported for now"
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

    model_test = builder.make_model(
        device,
        model_cfg=configs.model_test,
        random_state=(
            int(configs.run.random_state) if int(configs.run.deterministic) else None
        ),
    )
    lg.info(model_test)

    # initialize the stitched_response
    with h5py.File(f"/home/pingchua/projects/MAPS/figs/metalens_TF_uniform_numA-{configs.invdes.num_atom}_wl-0.85_p-0.3_mat-Si/transfer_matrix.h5", "r") as f:
        transfer_matrix = f["transfer_matrix"][:]
        transfer_matrix = torch.tensor(transfer_matrix, dtype=torch.complex64)
        transfer_matrix = transfer_matrix.to(device)
        if configs.model.conv_cfg.max_tm_norm:
            transfer_matrix = transfer_matrix / torch.max(transfer_matrix.abs())
    stitched_response = []
    for i in range(configs.model.conv_cfg.path_depth):
        stitched_response.append(transfer_matrix)
    stitched_response = torch.stack(stitched_response, dim=0).to(device)

    # -----------------------------------------------
    # build the patch metalens for inverse projection
    csv_file = f"core/metaatom_response_fsdx-0.3.csv"
    LUT = {}
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if float(row[0]) > 0.14:
                break
            LUT[float(row[0])] = float(row[1])

    LUT_height = {}
    csv_file_height = f"core/metaatom_response_fsdx-0.3_height.csv"
    with open(csv_file_height, mode='r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if float(row[0]) <= 0.189:
                continue
            LUT_height[float(row[0])] = float(row[2])

    patch_metalens = PatchMetalens(
        atom_period=0.3,
        patch_size=configs.invdes.patch_size,
        num_atom=configs.invdes.num_atom,
        probing_region_size=configs.invdes.patch_size,
        target_phase_response=None,
        LUT=LUT if configs.invdes.design_var_type == "width" else LUT_height,
        device=device,
        target_dx=0.3,
        plot_root=configs.plot.plot_root,
        downsample_mode=configs.invdes.downsample_mode,
        downsample_method=configs.invdes.downsample_method,
        dz=configs.model.conv_cfg.delta_z_data,
        param_method=configs.invdes.param_method,
        tm_norm=configs.invdes.tm_norm,
        field_norm_condition=configs.invdes.field_norm_condition,
        design_var_type=configs.invdes.design_var_type, # width or height
        atom_width=configs.invdes.atom_width,
    )
    if configs.invdes.finetune_entire:
        patch_metalens_list = [
            PatchMetalens(
                atom_period=0.3,
                patch_size=configs.invdes.patch_size,
                num_atom=configs.invdes.num_atom,
                probing_region_size=configs.invdes.patch_size,
                target_phase_response=None,
                LUT=LUT if configs.invdes.design_var_type == "width" else LUT_height,
                device=device,
                target_dx=0.3,
                plot_root=configs.plot.plot_root,
                downsample_mode=configs.invdes.downsample_mode,
                downsample_method=configs.invdes.downsample_method,
                dz=configs.model.conv_cfg.delta_z_data,
                param_method=configs.invdes.param_method,
                tm_norm=configs.invdes.tm_norm,
                field_norm_condition=configs.invdes.field_norm_condition,
                design_var_type=configs.invdes.design_var_type, # width or height
                atom_width=configs.invdes.atom_width,
            ) for _ in range(configs.model.conv_cfg.path_depth)
        ]
    else:
        patch_metalens_list = None
        
    ensure_dir(configs.plot.plot_root)

    if configs.model.conv_cfg.near2far_method == "green_fn":
        near2far_matrix = probe_near2far_matrix(
            patch_metalens[0].total_opt if isinstance(patch_metalens, list) else patch_metalens.total_opt,
            configs.model.conv_cfg.lambda_data,
            device,
        )
        model_test.set_near2far_matrix(near2far_matrix)
        in_downsample_rate = configs.model.conv_cfg.in_downsample_rate
        out_downsample_rate = configs.model.conv_cfg.out_downsample_rate
        ds_near2far_matrix = near2far_matrix[
            out_downsample_rate//2::out_downsample_rate, 
            :
        ]
        ds_near2far_matrix = ds_near2far_matrix.reshape(ds_near2far_matrix.shape[0], -1, in_downsample_rate).sum(dim=-1)
        if configs.model.conv_cfg.calculate_in_hr:
            model.set_near2far_matrix(near2far_matrix)
        else:
            model.set_near2far_matrix(ds_near2far_matrix)

        # plt.figure()
        # plt.imshow(near2far_matrix.abs().cpu().numpy())
        # plt.colorbar()
        # plt.savefig(f"./figs/near2far_matrix.png")
        # plt.close()

        # plt.figure()
        # plt.imshow(ds_near2far_matrix.abs().cpu().numpy())
        # plt.colorbar()
        # plt.savefig(f"./figs/near2far_matrix_ds.png")
        # plt.close()
        # quit()

    # print(next(iter(test_loader))[0].shape)
    ## dummy forward to initialize quantizer
    model_test.set_test_mode()
    model(next(iter(test_loader))[0].to(device))
    model_test(next(iter(test_loader))[0].to(device))

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
    grad_scaler = amp.GradScaler(enabled=getattr(configs.run, "fp16", False))
    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    # mlflow.set_experiment(configs.run.experiment)
    # experiment = mlflow.get_experiment_by_name(configs.run.experiment)

    # run_id_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # mlflow.start_run(run_name=model_name)
    # mlflow.log_params(
    #     {
    #         "exp_name": configs.run.experiment,
    #         "exp_id": experiment.experiment_id,
    #         "run_id": mlflow.active_run().info.run_id,
    #         "init_lr": configs.optimizer.lr,
    #         "checkpoint": checkpoint,
    #         "restore_checkpoint": configs.checkpoint.restore_checkpoint,
    #         "pid": os.getpid(),
    #     }
    # )
    wandb.login()
    tag = wandb.util.generate_id()
    group = f"{datetime.date.today()}"
    name = f"{configs.run.wandb.name}-{datetime.datetime.now().hour:02d}{datetime.datetime.now().minute:02d}{datetime.datetime.now().second:02d}-{tag}"
    configs.run.pid = os.getpid()
    run = wandb.init(
        project=configs.run.wandb.project,
        group=group,
        name=name,
        id=tag,
        config=configs,
    )

    lossv, accv = [0], [0]
    epoch = 0
    if configs.invdes.admm:
        # init the variables needed for ADMM
        admm_vars = {}
        admm_vars["rho_admm"] = configs.aux_criterion.admm_consistency.rho_admm
        admm_vars["z_admm"] = [
            stitched_response[i].clone().detach() for i in range(configs.model.conv_cfg.path_depth)
        ]
        admm_vars["u_admm"] = [
            torch.zeros_like(admm_vars["z_admm"][i]) for i in range(configs.model.conv_cfg.path_depth)
        ]
    else:
        admm_vars = None
    assert configs.invdes.reset_frequency in {"epoch", "proj"}, f"Unknown reset_frequency: {configs.invdes.reset_frequency}"
    
    batch_size = 200
    # random pick batch_size samples from the train_loader
    with DeterministicCtx(seed=41):
        samples = [test_loader.dataset[i] for i in random.sample(range(len(test_loader.dataset)), batch_size)]
    imgs = [sample[0] for sample in samples]
    imgs = torch.stack(imgs, dim=0).to(device)
    ds_rate = 1
    log_file_path = f"log/fmnist/meta_cnn/Benchmark/run-33_meta_cnn_fmnist_lr-0.002_pd-2_enc-phase_lam- 0.850_dz- 4.000_ps- 0.300_c-Exp6_draw_act_stats_out_ds_{ds_rate}_TMMat_tall_tgt.log"
    pattern = re.compile(r'Model saved to (\./checkpoint/.*?\.pt)')
    
    # List to store extracted paths
    saved_paths = []
    with open(log_file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                saved_paths.append(match.group(1))

    # Print all extracted paths
    for path in saved_paths:
        print("Extracted model path:", path)
    model_path = saved_paths[::2]
    model_test_path = saved_paths[1::2]
    test_model_activation = []
    for i in range(len(model_path)):

        load_model(
            model,
            model_path[i],
            ignore_size_mismatch=False,
        )
        load_model(
            model_test,
            model_test_path[i],
            ignore_size_mismatch=False,
        )

        # lg.info("Validate resumed model...")
        # test(
        #     model,
        #     validation_loader,
        #     0,
        #     criterion,
        #     lossv,
        #     accv,
        #     device,
        #     fp16=grad_scaler._enabled,
        # )
        # test(
        #     model_test,
        #     validation_loader,
        #     0,
        #     criterion,
        #     lossv,
        #     accv,
        #     device,
        #     fp16=grad_scaler._enabled,
        # )
        with torch.no_grad():
            model_output = model(imgs)
            model_test_output = model_test(imgs)

            activation = model_output[1].detach()
            activation_test = model_test_output[1].detach()[:, 1, :].unsqueeze(1)

            print("activation shape: ", activation.shape)
            print("activation_test shape: ", activation_test.shape)

            activation_fft = torch.fft.fft(activation, dim=-1).abs().square()
            activation_test_fft = torch.fft.fft(activation_test, dim=-1).abs().square()
            
            activation_fft = activation_test_fft.flatten(0, 1)
            low_freq_activation_fft = activation_fft[:, :16]
            low_freq_activation_fft[:, 1:] = low_freq_activation_fft[:, 1:] + activation_fft[:, -15:].flip(-1)
            low_freq_activation_fft = low_freq_activation_fft.mean(dim=0)
            # normalize the low_freq_activation_fft
            low_freq_activation_fft = low_freq_activation_fft / torch.sum(low_freq_activation_fft)
            test_model_activation.append(low_freq_activation_fft)
    test_model_activation = torch.stack(test_model_activation, dim=0).mean(dim=0)
    low_freq_activation_fft = test_model_activation / torch.sum(test_model_activation)
    with open(f"./unitest/activation_fft_{ds_rate}.csv", mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(["frequency", "energy"])
        for i in range(len(low_freq_activation_fft)):
            writer.writerow([i, low_freq_activation_fft[i].item()])


if __name__ == "__main__":
    main()