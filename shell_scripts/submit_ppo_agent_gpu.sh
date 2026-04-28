#!/bin/bash
#SBATCH --job-name=gpu_ppo_agent
#SBATCH --partition=gpuq

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --array=0-3 #can access only 4 gpus at a time  #0-9 #10 jobs, each gets 1 GPU

#SBATCH --error=outputs/agent_%A_%a.err
#SBATCH --output=outputs/agent_%A_%a.out

source ~/.bashrc 
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate mywandb

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export MUJOCO_GL=egl
export WANDB_MODE=online
export WANDB__SERVICE_WAIT=300

#wandb sweep ppo_sweep_pong.yaml
wandb agent {your-entity-name}/dlrl_policy_opt-sweeps/xmx6qxan




