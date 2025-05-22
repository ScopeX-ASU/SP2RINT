import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.config import configs
from pyutils.general import ensure_dir, logger

dataset = "navierstokes"
model = "meta_cnn_dno"
root = f"log/{dataset}/{model}/Benchmark"
script = "regression_benchmark.py"
config_file = f"configs/{dataset}/{model}/train/train.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    (
        lr,
        w_bit,
        in_bit,
        pd,
        pm,
        swap,
        rotate,
        weight_t,
        xy_pol,
        iden,
        group,
        alpha,
        alpha_train,
        beta,
        beta_train,
        en_mode,
        delta_z_data,
        pixel_size_data,
        lambda_data,
        skip_meta,
        lambda_mode,
        pixel_size_mode,
        delta_z_mode,
        lambda_train,
        pixel_size_train,
        delta_z_train,
        skp,
        gpu_id,
        id,
    ) = args
    pres = [f"export CUDA_VISIBLE_DEVICES={gpu_id};", "python3", script, config_file]

    with open(
        os.path.join(
            root,
            f"{model}_{dataset}_lr-{lr:.3f}_wb-{w_bit}_ib-{in_bit}_pd-{pd}_pm-{pm}_swap-{swap}_rot-{rotate}_iden-{int(iden)}_a-[{','.join([str(i) for i in alpha])}]_b-{int(beta)}_en_code-{en_mode}_skip_meta-{skip_meta}_lambda_data-{lambda_data: .3f}_delta_z_data-{delta_z_data: .3f}_pixel_size_data-{pixel_size_data: .3f}_skip_path-{skp}_run-{id}.log",
        ),
        "w",
    ) as wfid:
        exp = [
            f"--optimizer.lr={lr}",
            f"--run.random_state={41+id}",
            f"--model.conv_cfg.w_bit={w_bit}",
            f"--model.conv_cfg.in_bit={in_bit}",
            f"--model.conv_cfg.path_depth={pd}",
            f"--model.conv_cfg.path_multiplier={pm}",
            f"--model.conv_cfg.swap_mode={swap}",
            f"--model.conv_cfg.rotate_mode={rotate}",
            f"--model.conv_cfg.weight_train={weight_t}",
            f"--model.encode_mode={en_mode}",
            f"--model.conv_cfg.delta_z_mode={delta_z_mode}",
            f"--model.conv_cfg.pixel_size_mode={pixel_size_mode}",
            f"--model.conv_cfg.lambda_mode={lambda_mode}",
            f"--model.conv_cfg.enable_xy_pol={xy_pol}",
            f"--model.conv_cfg.enable_identity={iden}",
            f"--model.conv_cfg.beta_train={beta_train}",
            f"--model.conv_cfg.delta_z_train={delta_z_train}",
            f"--model.conv_cfg.lambda_train={lambda_train}",
            f"--model.conv_cfg.pixel_size_train={pixel_size_train}",
            f"--model.conv_cfg.skip_meta={skip_meta}",
            f"--model.conv_cfg.skip_path={skp}",
            f"--model.groups_list=[{','.join([str(i) for i in group])}]",
            f"--model.conv_cfg.delta_z_data={delta_z_data}",
            f"--model.conv_cfg.lambda_data={lambda_data}",
            f"--model.conv_cfg.pixel_size_data={pixel_size_data}",
            f"--model.conv_cfg.enable_alpha=[{','.join([str(i) for i in alpha])}]",
            f"--model.conv_cfg.alpha_train=[{','.join([str(i) for i in alpha_train])}]",
            f"--model.conv_cfg.enable_beta={beta}",
            f"--checkpoint.no_linear={1}",
            f"--checkpoint.model_comment=init_train_lr-{lr:.4f}_wb-{w_bit}_ib-{in_bit}_rotm-{rotate}_run-{id}",
        ]
        cmd = " ".join(pres + exp)
        logger.info(f"running command:\n\t{cmd}")
        subprocess.call(cmd, stderr=wfid, stdout=wfid, shell=True)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = []
    index = 0

    tasks = [
        (
            0.01,
            8,
            8,
            4,
            4,
            "fixed",
            "fixed",
            True,
            True,
            False,
            [32, 32, 32, 32],
            [True, True],
            [True, True],
            True,
            True,
            "phase",
            8.42,
            0.4,
            0.532,
            False,
            "train_share",
            "train_share",
            "train_share",
            True,
            True,
            True,
            True,
            1,
            1,
        ),
    ]
    with Pool(4) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
