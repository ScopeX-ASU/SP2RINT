import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.config import configs
from pyutils.general import ensure_dir, logger

dataset = "fmnist"
model = "meta_cnn"
root = f"log/{dataset}/{model}/FullOpt"
# script = "projected_classification.py"
script = "classification_benchmark.py"
config_file = f"configs/{dataset}/{model}/train/train_fullopt.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    (
        lr,
        pac,
        w_bit,
        in_bit,
        pd,
        pm,
        swap,
        rotate,
        weight_t,
        xy_pol,
        iden,
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
        kernel_size,
        pac_kernel_size_list,
        comment,
        mode,
        distance_constraint_weight,
    ) = args
    pres = [f"export CUDA_VISIBLE_DEVICES={gpu_id};", "python3", script, config_file]

    with open(
        os.path.join(
            root,
            f"run-{id}_{model}_{dataset}_lr-{lr:.3f}_wb-{w_bit}_ib-{in_bit}_pd-{pd}_pm-{pm}_swap-{swap}_rot-{rotate}_iden-{int(iden)}_a-[{','.join([str(int(i)) for i in alpha])}]_b-{int(beta)}_enc-{en_mode}_skip_meta-{skip_meta}_lam-{lambda_data: .3f}_dz-{delta_z_data: .3f}_ps-{pixel_size_data: .3f}_skip_p-{skp}_c-{comment}.log",
        ),
        "w",
    ) as wfid:
        exp = [
            f"--optimizer.lr={lr}",
            f"--run.n_epochs={50}",
            f"--run.random_state={41+18}",
            f"--aux_criterion.distance_constraint.weight={distance_constraint_weight}",
            f"--plot.plot_root={f'./figs/{dataset}/{model}/id-{id}_c-{comment}/'}",
            f"--model.conv_cfg.w_bit={w_bit}",
            f"--model.conv_cfg.in_bit={in_bit}",
            f"--model.conv_cfg.path_depth={pd}",
            f"--model.conv_cfg.path_multiplier={pm}",
            f"--model.conv_cfg.swap_mode={swap}",
            f"--model.conv_cfg.rotate_mode={rotate}",
            f"--model.conv_cfg.weight_train={weight_t}",
            # f"--model.encode_mode={en_mode}",
            f"--model.conv_cfg.encode_mode={en_mode}",
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
            f"--model.conv_cfg.delta_z_data={delta_z_data}",
            f"--model.conv_cfg.lambda_data={lambda_data}",
            f"--model.conv_cfg.pixel_size_data={pixel_size_data}",
            f"--model.conv_cfg.enable_alpha=[{','.join([str(i) for i in alpha])}]",
            f"--model.conv_cfg.alpha_train=[{','.join([str(i) for i in alpha_train])}]",
            f"--model.conv_cfg.enable_beta={beta}",
            f"--model.conv_cfg.mode={mode}",
            f"--model.conv_cfg.pac={pac}",
            f"--model.conv_cfg.kernel_size_list=[{','.join([str(i) for i in pac_kernel_size_list])}]",
            f"--model.conv_cfg.length={500}",
            f"--model.kernel_list={[1]}",
            f"--model.kernel_size_list={[kernel_size]}",
            f"--model.mid_channel_list={[1]}",
            f"--model.conv_cfg.metalens_init_file_path={{}}",
            f"--model.feature_dim={kernel_size}",

            f"--model.norm_cfg={None}",
            f"--model.act_cfg={None}",

            f"--checkpoint.no_linear={1}",
            f"--checkpoint.model_comment=init_train_lr-{lr:.4f}_wb-{w_bit}_ib-{in_bit}_rotm-{rotate}_c-{comment}",
        ]
        cmd = " ".join(pres + exp)
        logger.info(f"running command:\n\t{cmd}")
        subprocess.call(cmd, stderr=wfid, stdout=wfid, shell=True)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        # (0.001, 16, 16, 2, 1, "fixed", "fixed", True, False, False, [False, False], [False, False], False, False, "phase", 4, 0.02, 0.850, False, "fixed", "fixed", "fixed", False, False, False, False, 2, 120, 480, "px_wise_no_smooth"),
        # (0.001, 16, 16, 2, 1, "fixed", "fixed", True, False, False, [False, False], [False, False], False, False, "phase", 4, 0.06, 0.850, False, "fixed", "fixed", "fixed", False, False, False, False, 3, 120, 160, "ds_px_wise_no_smooth"),
        # (0.001, 16, 16, 2, 1, "fixed", "fixed", True, False, False, [False, False], [False, False], False, False, "phase", 4, 0.3, 0.850, False, "fixed", "fixed", "fixed", False, False, False, False, 1, 120, 32, "atom_wise_no_smooth"),
        # (0.001, 16, 16, 2, 1, "fixed", "fixed", True, False, False, [False, False], [False, False], False, False, "phase", 4, 0.6, 0.850, False, "fixed", "fixed", "fixed", False, False, False, False, 1, 120, 16, "two_atom_wise_no_smooth"),
        # (0.001, 16, 16, 2, 1, "fixed", "fixed", True, False, False, [False, False], [False, False], False, False, "mag", 4, 0.6, 0.850, False, "fixed", "fixed", "fixed", False, False, False, False, 1, 120, 16, "fft_two_atom_wise_no_smooth", "fft"),
        # (0.001, 16, 16, 2, 1, "fixed", "fixed", True, False, False, [False, False], [False, False], False, False, "mag", 4, 0.6, 0.850, False, "fixed", "fixed", "fixed", False, False, False, False, 1, 120, 16, "two_atom_wise_no_smooth_cweight", "regular", "phase_mag"),
        # (0.001, True, 16, 16, 2, 1, "fixed", "fixed", True, False, False, [False, False], [False, False], False, False, "complex", 4, 0.6, 0.850, False, "fixed", "fixed", "fixed", False, False, False, False, 1, 120, 32, "pac_test", "regular", "phase_mag"),
        # (0.002, True, 16, 16, 2, 1, "fixed", "fixed", True, False, False, [False, False], [False, False], False, False, "complex", 4, 0.6, 0.850, False, "fixed", "fixed", "fixed", False, False, False, False, 0, 120, 32, [1, 3, 5, 7, 13], "pac_test_2", "regular", "phase_mag"),
        # (0.002, True, 16, 16, 2, 1, "fixed", "fixed", True, False, False, [False, False], [False, False], False, False, "complex", 4, 0.6, 0.850, False, "fixed", "fixed", "fixed", False, False, False, False, 0, 120, 32, [1, 3, 5, 7, 13], "pac_test_3", "regular", "phase_mag"),
        # (0.002, True, 16, 16, 2, 1, "fixed", "fixed", True, False, False, [False, False], [False, False], False, False, "complex", 4, 0.6, 0.850, False, "fixed", "fixed", "fixed", False, False, False, False, 0, 120, 32, [1, 3, 5, 7, 13], "pac_test_4", "regular", "phase_mag"),
        # (0.002, True, 16, 16, 2, 1, "fixed", "fixed", True, False, False, [False, False], [False, False], False, False, "complex", 4, 0.6, 0.850, False, "fixed", "fixed", "fixed", False, False, False, False, 1, 120, 32, [17], "pac_test_6_unitary_proj", "regular", "phase_mag"),
        # (0.002, True, 16, 16, 2, 1, "fixed", "fixed", True, False, False, [False, False], [False, False], False, False, "complex", 4, 0.6, 0.850, False, "fixed", "fixed", "fixed", False, False, False, False, 0, 120, 32, [17], "pac_test_6_distance_constraints_1", "regular", "phase_mag", 1),
        # (0.002, True, 16, 16, 2, 1, "fixed", "fixed", True, False, False, [False, False], [False, False], False, False, "complex", 4, 0.6, 0.850, False, "fixed", "fixed", "fixed", False, False, False, False, 1, 120, 32, [17], "pac_test_6_distance_constraints_2", "regular", "phase_mag", 2),
        # (0.002, True, 16, 16, 2, 1, "fixed", "fixed", True, False, False, [False, False], [False, False], False, False, "complex", 4, 0.6, 0.850, False, "fixed", "fixed", "fixed", False, False, False, False, 2, 120, 32, [17], "pac_test_6_distance_constraints_5", "regular", "phase_mag", 5),
        # (0.002, True, 16, 16, 2, 1, "fixed", "fixed", True, False, False, [False, False], [False, False], False, False, "complex", 4, 0.6, 0.850, False, "fixed", "fixed", "fixed", False, False, False, False, 3, 120, 32, [17], "pac_test_6_distance_constraints_10", "regular", "phase_mag", 10),

        (0.02, True, 16, 16, 5, 1, "fixed", "fixed", True, False, False, [False, False], [False, False], False, False, "mag_phase", 4, 0.3, 0.850, False, "fixed", "fixed", "fixed", False, False, False, False, 0, 0, 500, [1], "test_baseline_wo_norm_act", "phase_mag", 1),
    ]
    with Pool(4) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
