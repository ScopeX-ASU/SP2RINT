'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-13 15:54:53
'''
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "fdfd"
model = "ffno"
exp_name = "train"
root = f"log/{dataset}/{model}/{exp_name}"
script = 'train_NN.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
checkpoint_dir = f'{dataset}/{model}/{exp_name}'
configs.load(config_file, recursive=True)


def task_launcher(args):
    mixup, device_type, alg, n_layers, modes, id, description, gpu_id, epochs, lr, criterion, criterion_weight, maxwell_loss, checkpt, bs = args
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    pres = [
            'python3',
            script,
            config_file
            ]
    suffix = f"model-{alg}_id-{id}_dcrp-{description}"
    with open(os.path.join(root, f'{suffix}.log'), 'w') as wfid:
        exp = [
            f"--dataset.device_type={device_type}",
            f"--dataset.processed_dir={device_type}",
            f"--dataset.num_workers={4}",
            f"--dataset.augment.prob={mixup}",

            f"--run.n_epochs={epochs}",
            f"--run.batch_size={bs}",
            f"--run.gpu_id={gpu_id}",
            f"--run.log_interval={100}",
            f"--run.random_state={59}",
            f"--run.fp16={False}",

            f"--criterion.name={criterion}",
            f"--criterion.weight={criterion_weight}",

            f"-aux_criterion.maxwell_residual_loss.weight={maxwell_loss}",

            f"--test_criterion.name={'nmse'}",
            f"--test_criterion.weighted_frames={0}",
            f"--test_criterion.weight={1}",

            f"--scheduler.lr_min={lr*5e-3}",

            f"--plot.train={True}",
            f"--plot.valid={True}",
            f"--plot.test={True}",
            f"--plot.interval=1",
            f"--plot.dir_name={model}_{exp_name}_des-{description}_id-{id}",
            f"--optimizer.lr={lr}",

            f"--model.name={alg}",
            f"--model.in_channels={3}",
            f"--model.out_channels={2}",
            f"--model.hidden_list={[32]}",
            f"--model.dim={32}",
            f"--model.kernel_list={[32]*n_layers}",
            f"--model.kernel_size_list={[1]*n_layers}",
            f"--model.padding_list={[0]*n_layers}",
            f"--model.mode_list={[modes]*n_layers}",
            f"--model.act_func={'GELU'}",
            f"--model.dropout_rate={0.0}",
            f"--model.drop_path_rate={0.0}",
            f"--model.aux_head={False}",
            f"--model.aux_head_idx={1}",
            f"--model.pos_encoding={'none'}",
            f"--model.with_cp={False}",
            f"--model.conv_stem={False}",
            f"--model.aug_path={True}",
            f"--model.ffn={True}",
            f"--model.ffn_dwconv={True}",

            f"--checkpoint.model_comment={suffix}",
            f"--checkpoint.resume={False}" if checkpt == "none" else f"--checkpoint.resume={True}",
            f"--checkpoint.restore_checkpoint={checkpt}",
            f"--checkpoint.checkpoint_dir={checkpoint_dir}",
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        # [0.0, "metacoupler", "FFNO2d", 8, (76, 113), 1, "full_mode", 2, 50, 0.005, "nmse", 1, 0, "none", 2],
        [0.0, "metacoupler", "FFNO2d", 8, (30, 45), 2, "less_mode", 3, 50, 0.005, "nmse", 1, 0, "none", 2],
    ]

    with Pool(8) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
