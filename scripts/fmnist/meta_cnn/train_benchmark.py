import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.config import configs, Config
from pyutils.general import ensure_dir, logger

dataset = "fmnist"
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
            f"--run.log_interval={1 if TM_model_method == 'end2end' else 200}",

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

            f"--end2end_sharpness_scheduler.mode={'cosine'}",
            f"--end2end_sharpness_scheduler.init_sharpness={10}",
            f"--end2end_sharpness_scheduler.final_sharpness={256}",
            f"--end2end_sharpness_scheduler.num_train_epochs={invdes_num_epoch * 6}", # three times of the budget

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
            # f"--model_test.conv_cfg.TM_model_method={'default' if TM_model_method != 'end2end' else 'end2end'}",
            f"--model_test.conv_cfg.TM_model_method={'default'}",


            f"--checkpoint.no_linear={1}",
            f"--checkpoint.model_comment=init_train_lr-{lr:.4f}_c-{comment}",
            f"--checkpoint.resume={1 if ckpt is not None else 0}",
            f"--checkpoint.restore_checkpoint={ckpt}",
            f"--checkpoint.save_best_model_k={1}",
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

        # (0.002, True, 16, 16, 2, 1, "fixed", "fixed", True, False, False, [False, False], [False, False], False, False, "complex", 4, 0.3, 0.850, False, "fixed", "fixed", "fixed", False, False, False, False, 0, 0, 32, [1], "test_baseline_wo_norm_act", "regular", "phase_mag", 1),

        # (0.002, True, 2, "phase", 4, 0.3, 0.850, 0, 1, 32, [1], "test", "regular", "phase", 0, False),
        # (0.002, True, 2, "phase", 4, 0.3, 0.850, 0, 2, 32, [17], "proj", "regular", "phase_mag", 0.5, True, 15),
        # (0.002, True, 2, "phase", 4, 0.3, 0.850, 1, 2, 32, [17], "proj_1", "regular", "phase_mag", 1, True, 15),
        # (0.002, True, 2, "phase", 4, 0.3, 0.850, 2, 2, 32, [17], "proj_5", "regular", "phase_mag", 5, True, 15),
        # (0.002, True, 2, "phase", 4, 0.3, 0.850, 3, 2, 32, [17], "proj_10", "regular", "phase_mag", 10, True, 15),
        # (0.002, True, 2, "phase", 4, 0.3, 0.850, 0, 2, 32, [17], "proj_last_time_init", "regular", "phase_mag", 0.5, True, 15, "last_time"),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 0, 2, 32, [17], "proj_reproduce", "regular", "phase_mag", 0.5, True, 15, "LPA", True, False),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 0, 2, 32, [17], "proj_reproduce_correct_redo", "regular", "phase_mag", 0.5, True, "LPA", True, False, 15, 15, 15, 1, 1, "RS"),
        # (0.002, True, 2, "phase", 4, 0.3, 0.850, 1, 2, 32, [17], "proj_reproduce_phase_encode_redo", "regular", "phase_mag", 0.5, True, "LPA", True, False, 15, 15, 15, 1, 1, "RS"),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 2, 2, 32, [17], "proj_reproduce_comb_invdes_obj_redo", "regular", "phase_mag", 0.5, True, "LPA", False, False, 15, 15, 15, 1, 1, "RS"),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 3, 2, 32, [17], "proj_reproduce_linear_sys_redo", "regular", "phase_mag", 0.5, True, "LPA", True, True, 15, 15, 15, 1, 1, "RS"),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 0, 3, 32, [17], "downsample_5", "regular", "phase_mag", 0.5, True, "LPA", True, False, 5, 1, 1, 1, 1, "RS"),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 1, 3, 32, [17], "downsample_15", "regular", "phase_mag", 0.5, True, "LPA", True, False, 15, 1, 1, 1, 1, "RS"),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 2, 3, 32, [17], "downsample_3", "regular", "phase_mag", 0.5, True, "LPA", True, False, 3, 1, 1, 1, 1, "RS"),

        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 1, 3, 32, [17], "downsample_15_last_time", "regular", "phase_mag", 0.5, True, "last_time", True, False, 15, 1, 1, 1, 1, "RS"),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 0, 3, 32, [17], "downsample_5_last_time", "regular", "phase_mag", 0.5, True, "last_time", True, False, 5, 1, 1, 1, 1, "RS"),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 1, 4, 32, [17], "new_downsample_15_wo_roll_last_time", "regular", "phase_mag", 0.5, True, "last_time", True, False, 15, 1, 1, 1, 1, "RS"),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 1, 4, 32, [17], "new_downsample_15_last_time", "regular", "phase_mag", 0.5, True, "last_time", True, False, "input_only", 15, 5, 50, 1, 1, 1, 1, "RS", 50),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 0, 5, 32, [17], "vary_downsample_rate_last_time", "regular", "phase_mag", 0.5, True, "last_time", True, False, "both", 15, 5, 45, 1, 1, 1, 1, "RS", 50),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 0, 6, 32, [17], "opt_receiver_no_proj_fft", "fft", (3, 3), "phase_mag", 0, False, "last_time", True, True, "both", 15, 5, 50, 1, 1, 1, 1, "RS", 50, 50),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 0, 6, 32, [17], "small_smooth", "fft", 3, 3, "phase_mag", 0, False, "last_time", "layer_wise", True, True, "both", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 10, "diag"),
        # (0.002, True, 3, "complex", 2, 0.3, 0.850, 1, 6, 64, [17], "full_opt_test", "fft", 3, 3, "phase_mag", 0, False, "last_time", True, True, "both", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 0, 7, 32, [17], "opt_receiver_less_const_fft", "fft", "phase_mag", 0.1, True, "last_time", True, True, "both", 15, 5, 50, 15, 15, 1, 1, "RS", 50, 50),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 1, 7, 32, [17], "opt_receiver_fft", "fft", "phase_mag", 0.5, True, "last_time", True, True, "both", 15, 5, 50, 15, 15, 1, 1, "RS", 50, 50),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 1, 7, 54, [17], "opt_receiver_larger_lens", "regular", "phase_mag", 0.5, True, "last_time", True, True, "both", 15, 5, 50, 1, 1, 1, 1, "RS", 50, 50),
        # (0.002, True, 3, "complex", 2, 0.3, 0.850, 1, 8, 64, [17], "layerwise_green_gn", "fft", 3, 3, "phase_mag", 0.5, True, "last_time", "layer_wise", True, True, "both", 15, 5, 50, 15, 15, 1, 1, "green_fn", 20, 50, "./checkpoint/fmnist/meta_cnn/train/Meta_CNNETE_init_train_lr-0.0020_c-full_opt_test_acc-86.10_epoch-49.pt"),
        # (0.002, True, 3, "complex", 2, 0.3, 0.850, 1, 8, 64, [17], "entire_green_gn", "fft", 3, 3, "phase_mag", 0.5, True, "last_time", "entire", False, True, "both", 15, 5, 50, 15, 15, 1, 1, "green_fn", 20, 50, "./checkpoint/fmnist/meta_cnn/train/Meta_CNNETE_init_train_lr-0.0020_c-full_opt_test_acc-86.10_epoch-49.pt"),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 1, 8, 32, [17], "small_smooth_matchentireTM", "fft", 3, 3, "phase_mag", 0.5, True, "last_time", "entire", False, True, "both", 15, 5, 50, 15, 15, 1, 1, "green_fn", 20, 50, "./checkpoint/fmnist/meta_cnn/train/Meta_CNNETE_init_train_lr-0.0020_c-small_wo_smooth_acc-76.92_epoch-48.pt", "TMMatching", 10, 10, "diag"),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 1, 8, 32, [17], "layerwise_wo_smooth_grating_width", "fft", 3, 3, "phase_mag", 2, True, "last_time", "layer_wise", False, True, "both", 15, 5, 50, 15, 15, 1, 1, "green_fn", 20, 50, "./checkpoint/fmnist/meta_cnn/train/Meta_CNNETE_init_train_lr-0.0020_c-small_wo_smooth_acc-76.92_epoch-48.pt", "ResponseMatching", 10, 0, "diag", "grating_width"),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 0, 9, 32, 160, [17], "hybrid_train", "fft", 3, 3, "phase_mag", 0, False, "last_time", "layer_wise", False, True, "both", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", 10, 0, "diag", "grating_width", 72, 54, 36),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 0, 10, 32, 160, [17], "hybrid_start_point", "fft", 3, 3, "phase_mag", 0.5, True, "last_time", "layer_wise", False, True, "both", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, "./checkpoint/fmnist/meta_cnn/train/Meta_CNNETE_init_train_lr-0.0020_c-hybrid_train_acc-84.38_epoch-43.pt", "TMMatching", 10, 0, "diag", "level_set", "avg", "max", 5e-3, 20, 72, 54, 36),
        # (0.002, True, 2, "complex", 4, 0.3, 0.850, 0, 10, 32, 160, [17], "hybrid_response_match", "fft", 3, 3, "phase_mag", 0.5, True, "last_time", "layer_wise", False, True, "both", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, "./checkpoint/fmnist/meta_cnn/train/Meta_CNNETE_init_train_lr-0.0020_c-hybrid_train_acc-84.38_epoch-43.pt", "ResponseMatching", 10, 0, "diag", "level_set", "avg", "max", 5e-3, 20, 72, 54, 36),
        # (0.002, True, 2, "phase", 4, 0.3, 0.850, 0, 11, 32, 3, [17], "hybrid_start_point", "none", 3, 3, "phase_mag", 0, False, "last_time", "layer_wise", False, True, "both", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "max", 5e-3, 20, 72, 54, 36, 2),
        # (0.002, True, 2, "phase", 4, 0.3, 0.850, 2, 12, 32, 3, [17], "hybrid_TM_match", "none", 3, 3, "phase_mag", 0.5, True, "last_time", "layer_wise", False, True, "both", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, "./checkpoint/fmnist/meta_cnn/train/Meta_CNNETE_init_train_lr-0.0020_c-hybrid_start_point_acc-90.22_epoch-23.pt", "TMMatching", 10, 0, "diag", "level_set", "avg", "max", 5e-3, 20, 72, 54, 36, 2),
        # (0.002, True, 2, "phase", 4, 0.3, 0.850, 1, 12, 32, 3, [17], "hybrid_TM_match_field_norm", "none", 3, 3, "phase_mag", 0.5, True, "last_time", "layer_wise", False, True, "both", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, "./checkpoint/fmnist/meta_cnn/train/Meta_CNNETE_init_train_lr-0.0020_c-hybrid_start_point_acc-90.22_epoch-23.pt", "TMMatching", 10, 0, "diag", "level_set", "avg", "field", 5e-3, 20, 72, 54, 36, 2),
        # (0.002, True, 2, "phase", 4, 0.3, 0.850, 1, 13, 32, 3, [17], "hybrid_projfreq_epoch_wo_distill", "none", 3, 3, "phase_mag", 0.5, True, "last_time", "layer_wise", False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", 5e-3, 20, 5, "epoch", 72, 54, 36, 2), # "./checkpoint/fmnist/meta_cnn/train/Meta_CNNETE_init_train_lr-0.0020_c-hybrid_start_point_acc-90.22_epoch-23.pt"
        # (0.002, True, 2, "phase", 4, 0.3, 0.850, 1, 13, 32, 3, [17], "hybrid_projfreq_proj_wo_distill", "none", 3, 3, "phase_mag", 0.5, True, "last_time", "layer_wise", False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", 5e-3, 20, 5, "proj", 72, 54, 36, 2), # "./checkpoint/fmnist/meta_cnn/train/Meta_CNNETE_init_train_lr-0.0020_c-hybrid_start_point_acc-90.22_epoch-23.pt"
        # (0.002, True, 2, "phase", 4, 0.3, 0.850, 2, 14, 32, 3, [17], "hybrid_resp_match", "none", 3, 3, "phase_mag", 0.5, True, "last_time", "layer_wise", False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", 10, 0, "diag", "level_set", "avg", "field", 5e-3, 20, 5, "epoch", 72, 54, 36, 2), # "./checkpoint/fmnist/meta_cnn/train/Meta_CNNETE_init_train_lr-0.0020_c-hybrid_start_point_acc-90.22_epoch-23.pt"
        # (0.002, 0, True, 2, "phase", 4, 0.3, 0.850, 2, 14, 32, 3, [17], "hybrid_resp_match_proj", "none", 3, 3, "phase_mag", 0.5, True, "last_time", "layer_wise", False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", 10, 0, "diag", "level_set", "avg", "field", 5e-3, 20, 5, "proj", 72, 54, 36, 2), # "./checkpoint/fmnist/meta_cnn/train/Meta_CNNETE_init_train_lr-0.0020_c-hybrid_start_point_acc-90.22_epoch-23.pt"
        # (0.0013, 0.0005, True, 2, "phase", 4, 0.3, 0.850, 1, 15, 32, 3, [17], "hybrid_projfreq_proj_tune_lr", "none", 3, 3, "phase_mag", 0.5, True, "last_time", "layer_wise", False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", 5e-3, 20, 5, "proj", 72, 54, 36, 2),
        # (0.002, 0, False, True, 2, "phase", 4, 0.3, 0.850, 1, 16, 32, 3, [17], "hybrid_projfreq_proj_denser_calLR", "none", 3, 3, "phase_mag", 0.5, True, "last_time", "layer_wise", False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", 5e-3, 20, 2, "proj", 72, 54, 36, 2),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 16, 32, 3, [17], "hybrid_projfreq_proj_denser_calHR", "none", 3, 3, "phase_mag", 0.5, True, "last_time", "layer_wise", False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", 5e-3, 20, 2, "proj", 72, 54, 36, 2),
        # (0.002, 0, False, True, 2, "phase", 4, 0.3, 0.850, 1, 16, 32, 3, [17], "hybrid_projfreq_epoch_denser_calLR", "none", 3, 3, "phase_mag", 0.5, True, "last_time", "layer_wise", False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", 5e-3, 20, 2, "epoch", 72, 54, 36, 2),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 16, 32, 3, [17], "hybrid_projfreq_epoch_denser_calHR", "none", 3, 3, "phase_mag", 0.5, True, "last_time", "layer_wise", False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", 5e-3, 20, 2, "epoch", 72, 54, 36, 2),
        # (0.002, 0, False, True, 2, "phase", 4, 0.3, 0.850, 1, 17, 32, 3, [17], "hybrid_projfreq_epoch_finetune_entire", "none", 3, 3, "phase_mag", 0.5, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", 5e-3, 20, 2, "epoch", 72, 54, 36, 2),
        # (0.002, 0, False, True, 2, "phase", 4, 0.3, 0.850, 1, 19, 32, 3, [17], "hybrid_response_matching", "none", 3, 3, "phase_mag", 0.5, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", 10, 0, "diag", "level_set", "avg", "field", 5e-3, 20, 2, "epoch", 72, 54, 36, 2, False, 1),
        # (0.002, 0, False, True, 2, "phase", 4, 0.3, 0.850, 1, 20, 32, 3, [17], "hybrid_no_ditance_penalty", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", 5e-3, 20, 2, "epoch", 72, 54, 36, 2, False, 1),
        # (0.002, 0, False, True, 2, "phase", 4, 0.3, 0.850, 1, 18, 32, 3, [17], "hybrid_admm", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", 5e-3, 20, 2, "epoch", 72, 54, 36, 2, True, 1),
        # (0.002, 0, False, True, 2, "phase", 4, 0.3, 0.850, 1, 21, 32, 3, [17], "hybrid_norm_wo_lens", "none", 3, 3, "phase_mag", 0.5, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 2, False, 1),
        # (0.002, 0, False, True, 2, "phase", 4, 0.3, 0.850, 1, 22, 32, 3, [17], "hybrid_norm_wo_lens_less_lr", "none", 3, 3, "phase_mag", 0.5, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 1e-3, 20, 2, "epoch", 72, 54, 36, 2, False, 1),
        # (0.002, 0, False, True, 2, "phase", 4, 0.3, 0.850, 1, 23, 32, 3, [17], "expid22_bk4verification", "none", 3, 3, "phase_mag", 0.5, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 1e-3, 20, 2, "epoch", 72, 54, 36, 2, False, 1),
        # (0.002, 0, False, True, 2, "phase", 4, 0.3, 0.850, 1, 24, 32, 3, [17], "migarate_verification_1", "none", 3, 3, "phase_mag", 0.5, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 1e-4, 20, 2, "epoch", 72, 54, 36, 2, False, 1),
        # (0.002, 0, False, True, 2, "phase", 4, 0.3, 0.850, 2, 24, 32, 3, [17], "origion_solver", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 2, 0.3, 1, False, 1),
        # (0.002, 0, False, True, 2, "phase", 4, 0.3, 0.850, 2, 24, 32, 3, [17], "hyper_solver", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 2, 0.3, 1, False, 1),
        # (0.002, 0, False, True, 2, "phase", 4, 0.3, 0.850, 1, 25, 32, 3, [17], "MAPS_old_timing", "none", 3, 3, "phase_mag", 0.5, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 2, False, 1),
        # (0.002, 0, False, True, 2, "phase", 4, 0.3, 0.850, 1, 25, 32, 3, [17], "MAPS_local_timing", "none", 3, 3, "phase_mag", 0.5, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 2, False, 1),
        # (0.002, 0, False, True, 2, "phase", 4, 0.3, 0.850, 1, 25, 32, 3, [17], "MAPS_local_timing_vef_share_factor", "none", 3, 3, "phase_mag", 0.5, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 2, False, 1),
        # (0.002, 0, False, True, 2, "phase", 4, 0.3, 0.850, 0, 26, 32, 3, [17], "corrected_matmul", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 2, False, 1),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 27, 32, 3, [17], "Exp0_sweep_ws_2", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 2, 0.1, 0.4, False, 1),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 27, 32, 3, [17], "Exp0_sweep_ws_3", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 3, 0.1, 0.4, False, 1),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 27, 32, 3, [17], "Exp0_sweep_ws_4", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 4, 0.1, 0.4, False, 1),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 27, 32, 4, [17], "Exp0_sweep_outC_4", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 3, 0.1, 0.4, False, 1),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 27, 32, 5, [17], "Exp0_sweep_outC_5", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 3, 0.1, 0.4, False, 1),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 27, 32, 6, [17], "Exp0_sweep_outC_6", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 3, 0.1, 0.4, False, 1),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 27, 32, 10, [17], "Exp0_sweep_outC_10", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 3, 0.1, 0.4, False, 1),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 27, 32, 8, [17], "Exp0_sweep_outC_8", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 3, 0.1, 0.4, False, 1),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 27, 32, 4, [17], "Exp0_sweep_gap_0p5", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 3, 0.1, 0.5, False, 1),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 27, 32, 4, [17], "Exp0_sweep_gap_0p6", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 3, 0.1, 0.6, False, 1),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 27, 32, 4, [17], "Exp0_sweep_gap_0p7", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 3, 0.1, 0.7, False, 1),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 27, 32, 4, [17], "Exp0_sweep_gap_0p8", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 3, 0.1, 0.8, False, 1),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 27, 32, 4, [17], "Exp0_sweep_width_0p2", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 27, 32, 4, [17], "Exp0_sweep_width_0p3", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 3, 0.3, 0.4, False, 1),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 27, 32, 4, [17], "Exp0_sweep_width_0p4", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 3, 0.4, 0.4, False, 1),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 27, 32, 4, [17], "Exp0_sweep_width_0p5", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 3, 0.5, 0.4, False, 1),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 28, 32, 3, [17], "cal_in_HR", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, 20, 2, "epoch", 72, 54, 36, 2, 0.1, 0.4, False, 1),

        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 29, 32, 4, [17], "Exp2_per_proj", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "proj", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_proj", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 29, 32, 4, [17], "Exp2_per_epoch", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 29, 32, 4, [17], "Exp2_per_train", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_training", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 29, 32, 4, [17], "Exp2_mixed", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "mixed", 30, 128),
        
        
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 29, 32, 4, [17], "Exp2_per_proj_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "proj", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_proj", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 29, 32, 4, [17], "Exp2_per_epoch_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 29, 32, 4, [17], "Exp2_per_train_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_training", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 29, 32, 4, [17], "Exp2_mixed_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "mixed", 30, 128),
    
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 30, 32, 4, [17], "Exp3_20_20", "none", 3, 3, "phase_mag", 0, True, "last_time", False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 20, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 30, 32, 4, [17], "Exp3_20_10", "none", 3, 3, "phase_mag", 0, True, "last_time", False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 10, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 30, 32, 4, [17], "Exp3_20_5", "none", 3, 3, "phase_mag", 0, True, "last_time", False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 5, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 30, 32, 4, [17], "Exp3_20_4", "none", 3, 3, "phase_mag", 0, True, "last_time", False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 4, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 30, 32, 4, [17], "Exp3_20_1", "none", 3, 3, "phase_mag", 0, True, "last_time", False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 1, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 30, 32, 4, [17], "Exp3_10_2", "none", 3, 3, "phase_mag", 0, True, "last_time", False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 30, 32, 4, [17], "Exp3_10_1", "none", 3, 3, "phase_mag", 0, True, "last_time", False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 1, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 30, 32, 4, [17], "Exp3_40_4", "none", 3, 3, "phase_mag", 0, True, "last_time", False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 40, 4, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),

        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 30, 32, 4, [17], "Exp3_20_20_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 20, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 30, 32, 4, [17], "Exp3_20_10_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 10, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 30, 32, 4, [17], "Exp3_20_5_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 5, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 30, 32, 4, [17], "Exp3_20_4_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 4, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 30, 32, 4, [17], "Exp3_20_1_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 1, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 30, 32, 4, [17], "Exp3_10_2_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 30, 32, 4, [17], "Exp3_10_1_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 1, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 30, 32, 4, [17], "Exp3_40_4_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 40, 4, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 30, 32, 4, [17], "Exp3_5_1_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 5, 1, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 30, 32, 4, [17], "Exp3_4_1_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 4, 1, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 30, 32, 4, [17], "Exp3_3_1_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 3, 1, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 30, 32, 4, [17], "Exp3_2_1_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 2, 1, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),



        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 31, 32, 4, [17], "Exp4_layerwise_end2end", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 31, 32, 4, [17], "Exp4_end2end", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 31, 32, 4, [17], "Exp4_layerwise_end2end_adaptive_ft_lr", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, True, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 31, 32, 4, [17], "Exp4_end2end_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 31, 32, 4, [17], "Exp4_layerwise_end2end_adaptive_ft_lr_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, True, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 31, 32, 4, [17], "Exp4_layerwise_end2end_fix_ft_lr_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 31, 32, 4, [17], "Exp4_layerwise_only", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),

        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 31, 32, 4, [17], "Exp4_end2end_final", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 31, 32, 4, [17], "Exp4_layerwise_end2end_adaptive_ft_lr_final", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, True, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 31, 32, 4, [17], "Exp4_layerwise_end2end_fix_ft_lr_final", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),

        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 31, 32, 4, [17], "Exp4_HR_end2end_final", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 5, 5, 50, 15, 5, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 31, 32, 4, [17], "Exp4_HR_layerwise_end2end_adaptive_ft_lr_final", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, True, False, True, "both", "constant", 15, 15, 5, 5, 50, 15, 5, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 31, 32, 4, [17], "Exp4_HR_layerwise_end2end_fix_ft_lr_final", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 15, 5, 5, 50, 15, 5, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),

        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 0, 31, 32, 4, [17], "Exp4_redo_finetune_entire", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 3, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0, False, "width"),
        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 1, 31, 32, 4, [17], "Exp4_redo_layerwise_finetune_adaptive_lr", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, True, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 3, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0, False, "width"),
        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 2, 31, 32, 4, [17], "Exp4_redo_layerwise_finetune_fix_lr", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 3, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0, False, "width"),

        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 32, 32, 4, [17], "Exp5_response_matching_3", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "fourier", 3, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 32, 32, 4, [17], "Exp5_response_matching_4", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "fourier", 4, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 32, 32, 4, [17], "Exp5_response_matching_5", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "fourier", 5, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 32, 32, 4, [17], "Exp5_response_matching_6", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "fourier", 6, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 32, 32, 4, [17], "Exp5_response_matching_7", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "fourier", 7, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 32, 32, 4, [17], "Exp5_response_matching_8", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "fourier", 8, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 32, 32, 4, [17], "Exp5_response_matching_9", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 32, 32, 4, [17], "Exp5_response_matching_10", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "fourier", 10, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 32, 32, 4, [17], "Exp5_response_matching_random_5", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "random", 3, 5, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 32, 32, 4, [17], "Exp5_response_matching_random_10", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "random", 4, 10, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 32, 32, 4, [17], "Exp5_response_matching_random_18", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "random", 4, 18, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
    
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 33, 32, 4, [17], "Exp6_out_ds_5_rerun_match_tm", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 5, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 33, 32, 4, [17], "Exp6_out_ds_15_ResMat", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 33, 32, 4, [17], "Exp6_out_ds_5_ResMat", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 5, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 33, 32, 4, [17], "Exp6_out_ds_5_ResMat_tall_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 5, 5, 50, 15, 5, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 33, 32, 4, [17], "Exp6_out_ds_3_ResMat", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 33, 32, 4, [17], "Exp6_out_ds_3_ResMat_tall_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 3, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 33, 32, 4, [17], "Exp6_out_ds_1_ResMat", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 1, 1, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 33, 32, 4, [17], "Exp6_out_ds_1_ResMat_tall_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 1, 1, 50, 15, 1, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 33, 32, 4, [17], "Exp6_out_ds_5_rerun_match_tm_taller_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 33, 32, 4, [17], "Exp6_out_ds_3_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 33, 32, 4, [17], "Exp6_out_ds_1_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 15, 1, 1, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 33, 32, 4, [17], "Exp6_out_ds_15_TMMat_tall_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 33, 32, 4, [17], "Exp6_out_ds_5_TMMat_tall_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 5, 5, 50, 15, 5, 1, 1, "green_fn", 50, 50, None, "TMMatching", "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 33, 32, 4, [17], "Exp6_out_ds_3_TMMat_tall_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 3, 1, 1, "green_fn", 50, 50, None, "TMMatching", "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 33, 32, 4, [17], "Exp6_out_ds_1_TMMat_tall_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 1, 1, 50, 15, 1, 1, 1, "green_fn", 50, 50, None, "TMMatching", "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),

        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 34, 32, 4, [17], "Exp7_out_ds_15_ResMat_tall_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", True, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 34, 32, 4, [17], "Exp7_out_ds_5_ResMat_tall_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 5, 5, 50, 15, 5, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", True, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 34, 32, 4, [17], "Exp7_out_ds_3_ResMat_tall_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 3, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", True, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 34, 32, 4, [17], "Exp7_out_ds_1_ResMat_tall_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 1, 1, 50, 15, 1, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", True, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 34, 32, 4, [17], "Exp7_out_ds_15_ResMat_tall_tgt_in_itpl_480", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", True, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 34, 32, 4, [17], "Exp7_out_ds_5_ResMat_tall_tgt_in_itpl_480", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 5, 5, 50, 15, 5, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", True, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 34, 32, 4, [17], "Exp7_out_ds_3_ResMat_tall_tgt_in_itpl_480", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 3, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", True, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 34, 32, 4, [17], "Exp7_out_ds_1_ResMat_tall_tgt_in_itpl_480", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 1, 1, 50, 15, 1, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", True, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 34, 32, 4, [17], "cpr_with_baodi_ResMat_ds_15_end2end_only_in_itpl_480", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", True, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        
        
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 35, 32, 4, [17], "Exp4_final_layerwise_only", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 35, 32, 4, [17], "Exp4_final_end2end_only", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 35, 32, 4, [17], "Exp4_final_mix_fix_invdes_lr", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 35, 32, 4, [17], "Exp4_final_mix_adaptive_invdes_lr", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, True, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        

        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 36, 32, 4, [17], "Exp7_ds_5", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 5, 5, 50, 15, 5, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 36, 32, 4, [17], "Exp7_ds_3", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 3, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 36, 32, 4, [17], "Exp7_ds_1", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 1, 1, 50, 15, 1, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),

        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 36, 32, 4, [17], "Exp7_ds_5_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 5, 5, 50, 15, 5, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 1, 36, 32, 4, [17], "Exp7_ds_3_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 3, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 36, 32, 4, [17], "Exp7_ds_1_rerun", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 1, 1, 50, 15, 1, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),

        # (0.002, 0, "fourier_basis", True, True, 2, "phase", 4, 0.3, 0.850, 0, 36, 32, 4, [17], "Exp7_ds_5_fourier_basis_small_lr", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 5, 5, 50, 15, 5, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 2e-3, True, 10, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, "fourier_basis", True, True, 2, "phase", 4, 0.3, 0.850, 0, 36, 32, 4, [17], "Exp7_ds_15_fourier_basis", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, "fourier_basis", True, True, 2, "phase", 4, 0.3, 0.850, 1, 36, 32, 4, [17], "Exp7_ds_5_fourier_basis", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 5, 5, 50, 15, 5, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, "fourier_basis", True, True, 2, "phase", 4, 0.3, 0.850, 2, 36, 32, 4, [17], "Exp7_ds_3_fourier_basis", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 3, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, "fourier_basis", True, True, 2, "phase", 4, 0.3, 0.850, 3, 36, 32, 4, [17], "Exp7_ds_1_fourier_basis", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 1, 1, 50, 15, 1, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 10, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 0, 33, 32, 4, [17], "Exp6_out_ds_2_TMMat_tall_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 2, 2, 50, 15, 2, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0, False, "width"),
        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 1, 33, 32, 4, [17], "Exp6_out_ds_4_TMMat_tall_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 4, 4, 50, 15, 4, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0, False, "width"),
        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 2, 33, 32, 4, [17], "Exp6_out_ds_6_TMMat_tall_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 6, 6, 50, 15, 6, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0, False, "width"),
        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 3, 33, 32, 4, [17], "Exp6_out_ds_8_TMMat_tall_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 8, 8, 50, 15, 8, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0, False, "width"),
        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 0, 33, 32, 4, [17], "Exp6_out_ds_10_TMMat_tall_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 10, 10, 50, 15, 10, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0, False, "width"),
        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 1, 33, 32, 4, [17], "Exp6_out_ds_12_TMMat_tall_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 12, 12, 50, 15, 12, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0, False, "width"),

        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 0, 33, 32, 4, [17], "Exp6_draw_act_stats_out_ds_3_TMMat_tall_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 3, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0, False, "width"),
        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 1, 33, 32, 4, [17], "Exp6_draw_act_stats_out_ds_1_TMMat_tall_tgt", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 1, 1, 50, 15, 1, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0, False, "width"),

        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 0, 37, 32, 4, [17], "Exp8_ds_3_act_penalty", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 3, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0.5),
        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 0, 37, 32, 4, [17], "Exp8_ds_3_act_penalty_layerwise", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 3, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 5),
        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 0, 37, 32, 4, [17], "Exp8_ds_3_act_penalty_end2end", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, False, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 3, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0.5),
        
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 0, 45, 32, 4, [17], "baseline_baodi_TM_matching_ds_15_end2end_only", "none", 3, 3, "phase_mag", 0, True, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.0005, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 3, 45, 32, 4, [1], "baseline_LPA", "none", 3, 3, "phase", 0, False, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0, False, "width"),
        # (0.0002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 0, 45, 32, 4, [1], "test_run", "none", 3, 3, "phase_mag", 0, False, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 45, 32, 4, [17], "baseline_smoothen_transfer_matrix", "none", 3, 3, "phase_mag", 0, False, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0.5, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 2, 45, 32, 4, [17], "baseline_project_smoothen_transfer_matrix", "none", 3, 3, "phase_mag", 0, False, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        (0.001, 0, "end2end", True, True, 2, "phase", 4, 0.3, 0.850, 0, 45, 32, 4, [17], "baseline_end2end", "none", 3, 3, "phase_mag", 0, False, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0, False, "width"),

        # (0.0002, 0, "conv", True, True, 2, "phase", 4, 0.3, 0.850, 3, 45, 32, 4, [17], "baseline_conv_LPA", "none", 3, 3, "phase_mag", 0, False, "last_time", False, True, True, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 13, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),

        # (0.002, 0, True, True, 2, "phase", 4, 0.3, 0.850, 3, 34, 32, 4, [17], "draw_out_ds_5", "none", 3, 3, "phase_mag", 0, True, "last_time", True, True, False, False, True, "both", "constant", 15, 15, 5, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128),
        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 0, 46, 32, 4, [17], "metaline_training", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 15, 15, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-2, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0, False, "height"),

        # (0.002, 0, "default", True, True, 2, "phase", 4, 0.3, 0.850, 0, 0, 32, 4, [17], "draw_pic", "none", 3, 3, "phase_mag", 0, True, "last_time", True, False, False, False, True, "both", "constant", 15, 15, 3, 3, 50, 15, 3, 1, 1, "green_fn", 50, 50, None, "TMMatching", False, "fourier", 9, 20, 0, "diag", "level_set", "avg", "field", "wo_lens", 5e-3, True, 20, 2, "epoch", 72, 54, 36, 3, 0.2, 0.4, False, 1, "per_epoch", 30, 128, 0, False, "width"),
    ]
    with Pool(8) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
