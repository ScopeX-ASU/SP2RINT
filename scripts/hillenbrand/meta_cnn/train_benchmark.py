import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.config import configs, Config
from pyutils.general import ensure_dir, logger

dataset = "hillenbrand"
model = "meta_cnn"
root = f"log/{dataset}/{model}/Benchmark"
script = "projected_classification.py"
config_file = f"configs/{dataset}/{model}/train/train.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    (
        lr,
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
        mode,
        distance_constraint_weight,
        project_GD,
        project_init,
        project_method,
        seperate_loss,
        linear_system,
        downsample_mode,
        downsample_rate_init,
        downsample_rate_final,
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
        invdes_res_match_modes,
        smooth_weight,
        smooth_mode,
        inv_param_method,
        patch_ds_method,
        tm_norm,
        invdes_lr,
        invdes_num_epoch,
        hidden_channel_1,
        hidden_channel_2,
        hidden_channel_3,
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
        # if linear_system:
        #     act_cfg = None
        #     norm_cfg = None
        # else:
        #     act_cfg = Config(
        #         type="ReLU",
        #         inplace=True,
        #     )
        #     norm_cfg = Config(
        #         type="BN1d",
        #         affine=True,
        #     )
        exp = [
            f"--dataset.name={dataset}",
            f"--dataset.root={'./data/vowels'}",
            f"--dataset.n_mfcc={13}",
            f"--dataset.n_valid_speakers={10}",
            f"--dataset.n_test_speakers={10}",
            f"--dataset.num_classes={12}",

            f"--optimizer.lr={lr}",

            f"--run.n_epochs={n_epochs}",
            f"--run.gpu_id={gpu_id}",
            f"--run.random_state={41+18}",
            f"--run.project_GD={project_GD}",
            f"--run.batch_size={4}",

            f"--downsample_rate_scheduler.n_epochs={n_epochs}",
            f"--downsample_rate_scheduler.init_ds_rate={downsample_rate_init}",
            f"--downsample_rate_scheduler.final_ds_rate={downsample_rate_final}",
            f"--downsample_rate_scheduler.milestone={[milestone]}",

            f"--invdes.project_init={project_init}",
            f"--invdes.project_method={project_method}",
            f"--invdes.seperate_loss={seperate_loss}",
            f"--invdes.downsample_mode={downsample_mode}",
            f"--invdes.patch_size={17}",
            f"--invdes.num_atom={kernel_size}",
            f"--invdes.param_method={inv_param_method}",
            f"--invdes.criterion.name={invdes_criterion}",
            f"--invdes.criterion.num_modes={invdes_res_match_modes}",
            f"--invdes.num_epoch={invdes_num_epoch}",
            f"--invdes.lr={invdes_lr}",
            f"--invdes.tm_norm={tm_norm}",
            f"--invdes.downsample_method={patch_ds_method}",

            f"--aux_criterion.distance_constraint.weight={distance_constraint_weight}",
            f"--aux_criterion.smooth_penalty.weight={smooth_weight}",
            f"--aux_criterion.smooth_penalty.mode={smooth_mode}",

            f"--plot.plot_root={f'./figs/{dataset}/{model}/id-{id}_c-{comment}/'}",

            f"--model.linear_system={linear_system}",
            f"--model.encode_mode={en_mode}", 
            f"--model.feature_dim={feature_dim}",
            f"--model.kernel_list={[1]}",
            f"--model.kernel_size_list={[kernel_size]}",
            f"--model.mid_channel_list={[1]}",
            f"--model.feature_extractor_type={feature_extractor_type}",
            f"--model.pool_out_size={pool_out_size}",
            f"--model.hidden_channel_1={hidden_channel_1}",
            f"--model.hidden_channel_2={hidden_channel_2}",
            f"--model.hidden_channel_3={hidden_channel_3}",

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


            f"--model_test.linear_system={linear_system}",
            f"--model_test.encode_mode={en_mode}",
            f"--model_test.pool_out_size={pool_out_size}",
            f"--model_test.feature_extractor_type={feature_extractor_type}",
            f"--model_test.kernel_size_list={[kernel_size]}",
            f"--model_test.feature_dim={feature_dim}",
            f"--model_test.hidden_channel_1={hidden_channel_1}",
            f"--model_test.hidden_channel_2={hidden_channel_2}",
            f"--model_test.hidden_channel_3={hidden_channel_3}",

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


            f"--checkpoint.no_linear={1}",
            f"--checkpoint.model_comment=init_train_lr-{lr:.4f}_c-{comment}",
            f"--checkpoint.resume={1 if ckpt is not None else 0}",
            f"--checkpoint.restore_checkpoint={ckpt}",
        ]
        cmd = " ".join(pres + exp)
        logger.info(f"running command:\n\t{cmd}")
        subprocess.call(cmd, stderr=wfid, stdout=wfid, shell=True)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        (0.0002, True, 2, "phase", 4, 0.3, 0.850, 0, 0, 32, 32, [17], "hybrid_1D_DONN_train", "avg_pool", "phase_mag", 0, False, "last_time", "layer_wise", False, True, "both", 15, 5, 50, 15, 15, 1, 1, "green_fn", 50, 50, None, "ResponseMatching", 10, 0, "diag", "level_set", "avg", "max", 5e-3, 20, 24, 16, 12),
    ]
    with Pool(4) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
