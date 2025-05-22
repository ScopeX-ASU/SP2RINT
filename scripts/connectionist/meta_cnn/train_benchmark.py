import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.config import configs, Config
from pyutils.general import ensure_dir, logger

dataset = "connectionist"
model = "meta_cnn"
root = f"log/{dataset}/{model}/Benchmark"
script = "projected_classification.py"
config_file = f"configs/{dataset}/{model}/train/train.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    (
        lr,
        lr_min,
        TM_model_method,
        calculate_in_hr,
        pac,
        pd,
        en_mode,
        delta_z_data,
        pixel_size_data,
        lambda_data,
        gpu_id,
        id,
        kernel_size,
        feature_dim,
        pac_kernel_size_list,
        comment,
        feature_extractor_type,
        fft_mode_1,
        fft_mode_2,
        mode,
        distance_constraint_weight,
        project_GD,
        project_init,
        layer_wise_matching,
        finetune_entire,
        adaptive_finetune_lr,
        seperate_loss,
        linear_system,
        downsample_mode,
        ds_sche_mode,
        in_downsample_rate_init,
        in_downsample_rate_final,
        out_downsample_rate_init,
        out_downsample_rate_final,
        milestone,
        in_downsample_rate,
        out_downsample_rate,
        test_in_downsample_rate,
        test_out_downsample_rate,
        near2far_method,
        n_epochs,
        pool_out_size,
        ckpt,
        test_ckpt,
        invdes_criterion,
        weighted_response,
        probe_source_mode,
        invdes_res_match_modes,
        num_random_sources,
        smooth_weight,
        smooth_mode,
        inv_param_method,
        patch_ds_method,
        tm_norm,
        field_norm_condition,
        invdes_lr,
        adaptive_invdes_lr,
        invdes_num_epoch,
        epoch_per_proj,
        reset_frequency,
        hidden_channel_1,
        hidden_channel_2,
        hidden_channel_3,
        window_size,
        input_wg_width,
        input_wg_interval,
        admm,
        rho_admm,
        invdes_sharp_scheduler_mode,
        invdes_sharpness_peak_epoch,
        invdes_sharpness_span_per_epoch,
        activation_smooth,
        uniform_metasurface,
        design_var_type,
    ) = args
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    pres = [
        # f"export CUDA_VISIBLE_DEVICES={gpu_id};", 
        "python3", 
        script, 
        config_file
    ]

    with open(
        os.path.join(
            root,
            f"run-{id}_{model}_{dataset}_lr-{lr:.3f}_pd-{pd}_enc-{en_mode}_lam-{lambda_data: .3f}_dz-{delta_z_data: .3f}_ps-{pixel_size_data: .3f}_c-{comment}.log",
        ),
        "w",
    ) as wfid:
        if "conv_lpa" in comment.lower():
            conv_LPA = True
            smoothen_transfer_matrix = False
            LPA = False
        elif "lpa" in comment.lower():
            LPA = True
            conv_LPA = False
            smoothen_transfer_matrix = False
        elif "smoothen_transfer_matrix" in comment.lower():
            smoothen_transfer_matrix = True
            conv_LPA = False
            LPA = False
        else:
            LPA = False
            conv_LPA = False
            smoothen_transfer_matrix = False
        exp = [
            f"--optimizer.lr={lr}",

            f"--scheduler.lr_min={lr_min}",

            f"--run.n_epochs={n_epochs}",
            f"--run.gpu_id={gpu_id}",
            f"--run.random_state={41+18}",
            f"--run.project_GD={project_GD}",
            f"--run.projection_once={smoothen_transfer_matrix}",
            f"--run.conv_LPA={conv_LPA}",
            f"--run.LPA={LPA}",
            f"--run.uniform_metasurface={uniform_metasurface}",

            f"--in_downsample_rate_scheduler.n_epochs={n_epochs}",
            f"--in_downsample_rate_scheduler.init_ds_rate={in_downsample_rate_init}",
            f"--in_downsample_rate_scheduler.final_ds_rate={in_downsample_rate_final}",
            f"--in_downsample_rate_scheduler.milestone={[milestone]}",
            f"--in_downsample_rate_scheduler.mode={ds_sche_mode}",

            f"--out_downsample_rate_scheduler.n_epochs={n_epochs}",
            f"--out_downsample_rate_scheduler.init_ds_rate={out_downsample_rate_init}",
            f"--out_downsample_rate_scheduler.final_ds_rate={out_downsample_rate_final}",
            f"--out_downsample_rate_scheduler.milestone={[milestone]}",
            f"--out_downsample_rate_scheduler.mode={ds_sche_mode}",

            f"--invdes.project_init={project_init}",
            f"--invdes.seperate_loss={seperate_loss}",
            f"--invdes.downsample_mode={downsample_mode}",
            f"--invdes.patch_size={17}",
            f"--invdes.num_atom={kernel_size}",
            f"--invdes.param_method={inv_param_method}",
            f"--invdes.criterion.name={invdes_criterion}",
            f"--invdes.criterion.weighted_response={weighted_response}",
            f"--invdes.criterion.probe_source_mode={probe_source_mode}",
            f"--invdes.criterion.num_modes={invdes_res_match_modes}",
            f"--invdes.criterion.num_random_sources={num_random_sources}",
            f"--invdes.num_epoch={invdes_num_epoch}",
            f"--invdes.lr={invdes_lr}",
            f"--invdes.tm_norm={tm_norm}",
            f"--invdes.downsample_method={patch_ds_method}",
            f"--invdes.epoch_per_proj={epoch_per_proj}",
            f"--invdes.reset_frequency={reset_frequency}",
            f"--invdes.finetune_entire={finetune_entire}",
            f"--invdes.admm={admm}",
            f"--invdes.field_norm_condition={field_norm_condition}",
            f"--invdes.adaptive_invdes_lr={adaptive_invdes_lr}",
            f"--invdes.adaptive_finetune_lr={adaptive_finetune_lr}",
            f"--invdes.finetune_lr_init={5.5e-4}",
            f"--invdes.finetune_lr_final={5e-5}",
            f"--invdes.layer_wise_matching={layer_wise_matching}",
            f"--invdes.design_var_type={design_var_type}",
            f"--invdes.atom_width={0.15}",

            f"--invdes_sharpness_scheduler.mode={invdes_sharp_scheduler_mode}",
            f"--invdes_sharpness_scheduler.init_sharpness={10}",
            f"--invdes_sharpness_scheduler.final_sharpness={256}",
            f"--invdes_sharpness_scheduler.sharpness_peak_epoch={invdes_sharpness_peak_epoch}",
            f"--invdes_sharpness_scheduler.sharpness_span_per_epoch={invdes_sharpness_span_per_epoch}",
            f"--invdes_sharpness_scheduler.num_train_epochs={n_epochs}",


            f"--aux_criterion.distance_constraint.weight={distance_constraint_weight}",
            f"--aux_criterion.smooth_penalty.weight={smooth_weight}",
            f"--aux_criterion.smooth_penalty.mode={smooth_mode}",
            f"--aux_criterion.admm_consistency.weight={1.0 if admm else 0}",
            f"--aux_criterion.admm_consistency.rho_admm={rho_admm}",
            f"--aux_criterion.activation_smooth.weight={activation_smooth}",
            f"--aux_criterion.activation_smooth.mode_threshold={8}",

            f"--plot.plot_root={f'./figs/{dataset}/{model}/id-{id}_c-{comment}/'}",

            f"--model.linear_system={linear_system}",
            f"--model.encode_mode={en_mode}", 
            f"--model.feature_dim={feature_dim}",
            f"--model.kernel_list={[1]}",
            f"--model.kernel_size_list={[kernel_size]}",
            f"--model.mid_channel_list={[1]}",
            f"--model.feature_extractor_type={feature_extractor_type}",
            f"--model.pool_out_size={pool_out_size}",
            f"--model.fft_mode_1={fft_mode_1}",
            f"--model.fft_mode_2={fft_mode_2}",
            f"--model.hidden_channel_1={hidden_channel_1}",
            f"--model.hidden_channel_2={hidden_channel_2}",
            f"--model.hidden_channel_3={hidden_channel_3}",
            f"--model.window_size={window_size}",
            f"--model.input_wg_width={input_wg_width}",
            f"--model.input_wg_interval={input_wg_interval}",
            f"--model.lambda_cen={lambda_data}",

            f"--model.conv_cfg.encode_mode={en_mode}", # this is the mode for the x encoder for the metalens (phase or mag or mag_phase)
            f"--model.conv_cfg.path_depth={pd}",
            f"--model.conv_cfg.delta_z_data={delta_z_data}",
            f"--model.conv_cfg.lambda_data={lambda_data}",
            f"--model.conv_cfg.pixel_size_data={pixel_size_data}",
            f"--model.conv_cfg.mode={mode}", # this is the mode for the transfer function of the metalens
            f"--model.conv_cfg.pac={pac}",
            f"--model.conv_cfg.kernel_size_list=[{','.join([str(i) for i in pac_kernel_size_list])}]",
            f"--model.conv_cfg.length={kernel_size}",
            f"--model.conv_cfg.metalens_init_file_path={{}}",
            f"--model.conv_cfg.in_downsample_rate={in_downsample_rate}",
            f"--model.conv_cfg.out_downsample_rate={out_downsample_rate}",
            f"--model.conv_cfg.near2far_method={near2far_method}",
            f"--model.conv_cfg.resolution={50}",
            f"--model.conv_cfg.max_tm_norm={True if tm_norm == 'max' else False}",
            f"--model.conv_cfg.calculate_in_hr={calculate_in_hr}",
            f"--model.conv_cfg.TM_model_method={TM_model_method}",


            f"--model_test.linear_system={linear_system}",
            f"--model_test.encode_mode={en_mode}",
            f"--model_test.pool_out_size={pool_out_size}",
            f"--model_test.feature_extractor_type={feature_extractor_type}",
            f"--model_test.fft_mode_1={fft_mode_1}",
            f"--model_test.fft_mode_2={fft_mode_2}",
            f"--model_test.kernel_size_list={[kernel_size]}",
            f"--model_test.feature_dim={feature_dim}",
            f"--model_test.hidden_channel_1={hidden_channel_1}",
            f"--model_test.hidden_channel_2={hidden_channel_2}",
            f"--model_test.hidden_channel_3={hidden_channel_3}",
            f"--model_test.window_size={window_size}",
            f"--model_test.input_wg_width={input_wg_width}",
            f"--model_test.input_wg_interval={input_wg_interval}",
            f"--model_test.lambda_cen={lambda_data}",

            f"--model_test.conv_cfg.encode_mode={en_mode}",
            f"--model_test.conv_cfg.in_downsample_rate={test_in_downsample_rate}",
            f"--model_test.conv_cfg.out_downsample_rate={test_out_downsample_rate}",
            f"--model_test.conv_cfg.near2far_method={near2far_method}",
            f"--model_test.conv_cfg.resolution={50}",
            f"--model_test.conv_cfg.length={kernel_size}",
            f"--model_test.conv_cfg.path_depth={pd}",
            f"--model_test.conv_cfg.delta_z_data={delta_z_data}",
            f"--model_test.conv_cfg.lambda_data={lambda_data}",
            f"--model_test.conv_cfg.pixel_size_data={pixel_size_data}",
            f"--model_test.conv_cfg.mode={mode}", # this is the mode for the transfer function of the metalens
            f"--model_test.conv_cfg.pac={pac}",
            f"--model_test.conv_cfg.kernel_size_list=[{','.join([str(i) for i in pac_kernel_size_list])}]",
            f"--model_test.conv_cfg.metalens_init_file_path={{}}",
            f"--model_test.conv_cfg.max_tm_norm={True if tm_norm == 'max' else False}",
            f"--model_test.conv_cfg.calculate_in_hr={calculate_in_hr}",
            f"--model_test.conv_cfg.TM_model_method={'default'}",


            f"--checkpoint.no_linear={1}",
            f"--checkpoint.model_comment=init_train_lr-{lr:.4f}_c-{comment}",
            f"--checkpoint.resume={1 if ckpt is not None else 0}",
            f"--checkpoint.restore_checkpoint={ckpt}",
            f"--checkpoint.restore_test_checkpoint={test_ckpt}",
            f"--checkpoint.save_best_model_k={1}",
        ]
        cmd = " ".join(pres + exp)
        logger.info(f"running command:\n\t{cmd}")
        subprocess.call(cmd, stderr=wfid, stdout=wfid, shell=True)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 1, 0, 32, 4, [17], "Ours_from_scratch", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 3, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0.5),
        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 2, 0, 32, 4, [17], "Ours_from_scratch_double_budget", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 3, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0.5),
        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 3, 0, 32, 4, [17], "Ours_transfer_from_FMNIST", "none", 3, 3, "phase_mag", 0, False, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 3, 1, 1, "green_fn", 50, 50, "./checkpoint/fmnist/meta_cnn/train/Meta_CNNETE_init_train_lr-0.0020_c-Exp6_out_ds_3_TMMat_tall_tgt_acc-89.93_epoch-44.pt", "./checkpoint/fmnist/meta_cnn/train/Meta_CNNETE_init_train_lr-0.0020_c-Exp6_out_ds_3_TMMat_tall_tgt_test_acc-89.93_epoch-44.pt", "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0.5),
        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 3, 0, 32, 4, [17], "Ours_uniform_metasurface", "none", 3, 3, "phase_mag", 0, False, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 3, 1, 1, "green_fn", 50, 50, None, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0.5, True),
        # (0.0005, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 0, 1, 32, 4, [1], "baseline_LPA", "none", 3, 3, "phase", 0, False, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0, False, "width"),
        (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 1, 1, 32, 11, [17], "baseline_smoothen_transfer_matrix", "none", 3, 3, "phase_mag", 0, False, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, None, "TMMatching", False, "fourier", 13, 20, 0.5, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 10, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0, False, "width"),
        # (0.0002, 0, "conv", True, True, 2, "phase", 4, 0.3, 0.850, 2, 1, 32, 4, [17], "baseline_conv_LPA", "none", 3, 3, "phase_mag", 0, False, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0, False, "width"),
        
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 45, 32, 4, [17], "baseline_baodi_TM_matching_ds_15_end2end_only", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.0002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 3, 45, 32, 4, [1], "baseline_phase_mask", "none", 3, 3, "phase_mag", 0, False, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 45, 32, 4, [17], "baseline_smoothen_transfer_matrix", "none", 3, 3, "phase_mag", 0, False, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0.5, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 45, 32, 4, [17], "baseline_project_smoothen_transfer_matrix", "none", 3, 3, "phase_mag", 0, False, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
    ]
    with Pool(8) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")