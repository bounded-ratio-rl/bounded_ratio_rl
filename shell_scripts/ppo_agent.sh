#!/bin/bash
#SBATCH --job-name=ppo_wb_agent
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00

#SBATCH --output=outputs/wb_agent_%A_%a.out
#SBATCH --error=outputs/wb_agent_%A_%a.err
#SBATCH --array=0-3

source ~/.bashrc 
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate mywandb

echo "which python: $(which python)"
echo "env python explicit path: /dartfs-hpc/rc/home/7/f007pc7/.conda/envs/mywandb/bin/python"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export MUJOCO_GL=egl
export WANDB_MODE=online
export WANDB__SERVICE_WAIT=300

#wandb agent {your-entity-name}/sb3/<sweep-id>
#wandb agent {your-entity-name}/dlrl_policy_opt-sweeps/wjz0gzv7
# Atari:
#wandb agent {your-entity-name}/dlrl_policy_opt-sweeps/afxh8mp9
# Atari, breakout-v-5
wandb agent {your-entity-name}/dlrl_policy_opt-sweeps/6u3z603c