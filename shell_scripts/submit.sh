#!/bin/bash
#SBATCH --job-name=BPO
#SBATCH --account=def-professor
#SBATCH --gpus=a100_4g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=16G
#SBATCH --time=5:00:00
#SBATCH --array=0-29 # 10 seeds x 3 envs
#SBATCH --output=logs/wandb_sweep_ppo%x_%A_%a.out

module load python/3.12.4 cuda/12.6
export MUJOCO_GL=egl
export WANDB_API_KEY=""

if [ -z "${SLURM_TMPDIR}" ]; then
  export SLURM_TMPDIR="/tmp"
fi
echo "Creating virtual environment in \$SLURM_TMPDIR..."
VENV_DIR=$SLURM_TMPDIR/venv
python -m venv $VENV_DIR
source $VENV_DIR/bin/activate
echo "Upgrading pip and installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# wandb agent {your-entity-name}/BPO/<SWEEP_ID_HERE>
wandb agent {your-entity-name}/BPO/eexagvim
