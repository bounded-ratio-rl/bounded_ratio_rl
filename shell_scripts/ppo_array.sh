#!/bin/bash
#SBATCH --job-name=ppo_array
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
# Set the array range to the number of lines minus one.
# see file runs.tsv, the number of lines must match the array size
#SBATCH --array=0-5

#SBATCH --error=outputs/ppo_%A_%a.err
#SBATCH --output=outputs/ppo_wandb_agent.out

source ~/.bashrc
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate mywandb

echo "which python: $(which python)"
echo "env python explicit path: /dartfs-hpc/rc/home/7/f007pc7/.conda/envs/mywandb/bin/python"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export MUJOCO_GL=egl
export WANDB_MODE=online
export WANDB__SERVICE_WAIT=300

mkdir -p outputs videos models runs

PARAM_FILE=runs.tsv
LINE=$((SLURM_ARRAY_TASK_ID + 1))

# Read columns from runs.tsv
# Read TAB-separated columns
# env_name, seed, policy_type, total_timesteps, n_envs, n_steps, batch_size, gamma, learning_rate, ent_coef, clip_range, n_epochs, gae_lambda, max_grad_norm, vf_coef
IFS=$'\t' read -r ENV_NAME SEED POLICY_TYPE TOTAL_TIMESTEPS N_ENVS N_STEPS BATCH_SIZE GAMMA LR ENT_COEF CLIP_RANGE N_EPOCHS GAE_LAMBDA MAX_GRAD_NORM VF_COEF < <(sed -n "${LINE}p" "$PARAM_FILE")

# Sanity checks
if [ -z "$ENV_NAME" ]; then
  echo "Empty or missing line $LINE in $PARAM_FILE"; exit 2
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: ${ENV_NAME} seed=${SEED}"

PY=/dartfs-hpc/rc/home/7/f007pc7/.conda/envs/mywandb/bin/python
$PY train.py \
  env=mujoco \
  algo=ppo \
  env.env_id="${ENV_NAME}" \
  training.seed="${SEED}" \
  algo.policy="${POLICY_TYPE}" \
  training.total_timesteps="${TOTAL_TIMESTEPS}" \
  env.n_envs="${N_ENVS}" \
  algo.n_steps="${N_STEPS}" \
  algo.batch_size="${BATCH_SIZE}" \
  algo.gamma="${GAMMA}" \
  algo.learning_rate="${LR}" \
  algo.ent_coef="${ENT_COEF}" \
  algo.clip_range="${CLIP_RANGE}" \
  algo.n_epochs="${N_EPOCHS}" \
  algo.gae_lambda="${GAE_LAMBDA}" \
  algo.max_grad_norm="${MAX_GRAD_NORM}" \
  algo.vf_coef="${VF_COEF}"

echo "PPO job array execution completed."
