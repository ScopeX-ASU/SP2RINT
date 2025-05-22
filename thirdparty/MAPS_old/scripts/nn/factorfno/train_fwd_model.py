'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-11-24 15:10:46
'''
import os
import subprocess
from multiprocessing import Pool

# import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "fdfd"
model = "factorfno"
exp_name = "train"
root = f"log/{dataset}/{model}/{exp_name}"
script = './core/train/examples/fwd_model_train.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train_fwd.yml'
checkpoint_dir = f'{dataset}/{model}/{exp_name}'
configs.load(config_file, recursive=True)


def task_launcher(args):
    mixup, device_type, data_dir, alg, output_sparam, fourier_feature, dim, num_layers, mode1, mode2, id, description, gpu_id, epochs, alm, lr, criterion, criterion_weight, H_loss, maxwell_loss, grad_loss, s_param_loss, direct_s_param_loss, alm_lambda, alm_mu, mu_growth, constraint_tol, checkpt_fwd, checkpt_adj, bs = args
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    pres = [
            'python3',
            script,
            config_file
            ]
    suffix = f"model-{alg}_dev-{device_type}_field-fwd_id-{id}_dcrp-{description}"
    if "bending" in device_type.lower():
        temp = [300]
        wl = [1.55]
        mode = [1]
        in_out_port_name = [['in_port_1', 'out_port_1']]
        img_size = 256
    else:
        raise ValueError(f"device_type {device_type} not supported")
    with open(os.path.join(root, f'{suffix}.log'), 'w') as wfid:
        exp = [
            f"--dataset.device_type={device_type}",
            f"--dataset.data_dir={data_dir}",
            f"--dataset.processed_dir={device_type}",
            f"--dataset.num_workers={4}",
            f"--dataset.augment.prob={mixup}",

            f"--run.n_epochs={epochs}",
            f"--run.batch_size={bs}",
            f"--run.use_cuda={1}",
            f"--run.gpu_id={gpu_id}",
            f"--run.log_interval={40}",
            f"--run.random_state={59}",
            f"--run.fp16={False}",

            f"--criterion.name={criterion}",
            f"--criterion.weight={criterion_weight}",

            f"--aux_criterion.maxwell_residual_loss.weight={maxwell_loss}",
            f"--aux_criterion.maxwell_residual_loss.using_ALM={alm}",
            f"--aux_criterion.grad_loss.weight={grad_loss}",
            f"--aux_criterion.s_param_loss.weight={s_param_loss}",
            f"--aux_criterion.Hx_loss.weight={H_loss}",
            f"--aux_criterion.Hy_loss.weight={H_loss}",
            f"--aux_criterion.direct_s_param_loss.weight={direct_s_param_loss}",

            f"--log_criterion.maxwell_residual_loss.using_ALM={alm}",
            f"--log_criterion.direct_s_param_loss.weight={direct_s_param_loss}",

            f"--test_criterion.name={'nmse'}",
            f"--test_criterion.weighted_frames={0}",
            f"--test_criterion.weight={1}",

            f"--scheduler.lr_min={lr*5e-3}",

            f"--plot.train={True}",
            f"--plot.valid={True}",
            f"--plot.test={True}",
            f"--plot.interval=1",
            f"--plot.dir_name={model}_{exp_name}_id-{id}_des-{description}",

            f"--optimizer.lr={lr}",
            f"--optimizer.ALM={alm}",
            f"--optimizer.ALM_lambda={alm_lambda}",
            f"--optimizer.ALM_mu={alm_mu}",
            f"--optimizer.ALM_mu_growth={mu_growth}",
            f"--optimizer.ALM_constraint_tol={constraint_tol}",

            f"--model_fwd.type={alg}",
            f"--model_fwd.train_field={'fwd'}",
            f"--model_fwd.dim={dim}",
            f"--model_fwd.in_channels={3}",
            f"--model_fwd.out_channels={2}",
            f"--model_fwd.img_size={img_size}",
            f"--model_fwd.mode_list={[(mode1, mode2)] * num_layers}",
            f"--model_fwd.mapping_size={64}",
            f"--model_fwd.fourier_feature={fourier_feature}",
            f"--model_fwd.temp={temp}",
            f"--model_fwd.wl={wl}",
            f"--model_fwd.mode={mode}",
            f"--model_fwd.in_out_port_name={in_out_port_name}",
            # f"--model_fwd.in_out_port_name={[['in_port_1', f'out_port_{i}'] for i in range(1, 3)]}",
            f"--model_fwd.kernel_list={[dim]*num_layers}",
            f"--model_fwd.kernel_size_list={[1]*num_layers}",
            f"--model_fwd.padding_list={[1]*num_layers}",
            f"--model_fwd.hidden_list={[dim*2]}",
            f"--model_fwd.output_sparam={output_sparam}",

            f"--checkpoint.model_comment={suffix}",
            f"--checkpoint.resume={False}" if checkpt_fwd == "none" else f"--checkpoint.resume={True}",
            f"--checkpoint.restore_checkpoint_fwd={checkpt_fwd}",
            f"--checkpoint.restore_checkpoint_adj={checkpt_adj}",
            f"--checkpoint.checkpoint_dir={checkpoint_dir}",
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    tasks = [
        # [0.0, "bending", "raw_opt_traj_10", "FactorFNO2d", True, "none", 64, 12, 60, 60, 2, "Exp1_FactorFNO_S_param_only", 1, 50, False, 0.002, "nmse", 0, 0, 0, 0, 0, 1, 0, 1, 2, 1e-4, "none", "none", 4],
        [0.0, "bending", "raw_opt_traj_10", "FactorFNO2d", True, "none", 64, 12, 60, 60, 2, "Exp1_FactorFNO_S_param_only_rerun", 1, 50, False, 0.002, "nmse", 0, 0, 0, 0, 0, 1, 0, 1, 2, 1e-4, "none", "none", 4],
    ]   

    with Pool(8) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
