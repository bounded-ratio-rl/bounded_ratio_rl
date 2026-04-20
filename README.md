# Bounded Policy Optimization

This repository implements Bounded Policy Optimization (BPO) and other reinforcement learning algorithms using Hydra for configuration management. It also includes [TTRL (Test-Time Reinforcement Learning)](https://arxiv.org/abs/2504.16084) for RL on LLMs without ground-truth labels, built on [verl](https://github.com/volcengine/verl).

## Installation

Install ```uv``` from https://docs.astral.sh/uv/getting-started/installation/

### Only stable-baselines3 + Gym mujoco/atari

```
uv sync --extra sb3
source .venv/bin/activate
```

### Only rsl_rl + isaaclab

```
uv sync --extra isaaclab
```

Then install Isaaclab
```
uv pip install isaaclab[isaacsim,all]==2.1.0 --extra-index-url https://pypi.nvidia.com
```

More details on isaaclab installation: , according to https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/isaaclab_pip_installation.html.

Activate env
```
source .venv/bin/activate
```

### Only TTRL + verl (LLM reinforcement learning)

```
uv sync --extra ttrl
source .venv/bin/activate
bash scripts/install_ttrl_deps.sh
```

### Install all
First sync uv
```
uv sync --extra all \
  --index-strategy unsafe-best-match
```
Then install Isaaclab
```
uv pip install isaaclab[isaacsim,all]==2.1.0 --extra-index-url https://pypi.nvidia.com
```
Then ttrl
```
source .venv/bin/activate
bash scripts/install_ttrl_deps.sh
```

## Basic Usage for SB3 + mujoco + atari

Run training with default configuration:
```bash
python train.py
```
Override environments and algorithms to reproduce results:
```bash
python train.py env=atari algo=bpo/atari env.env_id="PongNoFrameskip-v4"
python train.py env=mujoco algo=sac/mujoco env.env_id="Ant-v4"
python train.py env=mujoco algo=ppo/mujoco env.env_id="Hopper-v4"
```

**Available Algorithms:**
- `algo=bpo/<env_type>` - BPO (Bounded Policy Optimization)
- `algo=ppo/<env_type>` - PPO (Proximal Policy Optimization)
- `algo=sac/<env_type>` - SAC (Soft Actor-Critic)
- `algo=dqn/<env_type>` - DQN (Deep Q-Network)

### Overriding Configuration Parameters

You can override any configuration parameter using Hydra's dot notation:

```bash

# Override training parameters
python train.py env=atari algo=bpo/atari training.seed=123 training.total_timesteps=10000000

# Override algorithm hyperparameters
python train.py env=mujoco algo=bpo/mujoco algo.learning_rate=0.0003 algo.batch_size=256

# Override environment settings
python train.py env=mujoco algo=ppo/mujoco env.n_envs=8
```

## Basic Usage for rsl_rl + isaaclab

Run training with default configuration:
```bash
python train_rsl_rl.py
```
Running isaaclab the first time can take long (10 mins)

Override environments and algorithms to reproduce results:
```bash
python train_rsl_rl.py --algorithm bpo --task "Isaac-Velocity-Rough-Anymal-C-v0" --headless
```
**Available Algorithms:**
- `--algorithm bpo` - BPO (Bounded Policy Optimization)
- `--algorithm ppo` - PPO (Proximal Policy Optimization)

**Available tasks**
- Isaac-Velocity-Rough-Anymal-C-v0
- Isaac-Velocity-Rough-G1-v0
- Isaac-Velocity-Rough-Unitree-Go1-v0
- Isaac-Velocity-Rough-H1-v0
- Isaac-Velocity-Flat-Anymal-C-v0
- Isaac-Velocity-Flat-G1-v0
- Isaac-Velocity-Flat-Unitree-Go1-v0
- Isaac-Velocity-Flat-H1-v0

### Overriding Configuration Parameters

```bash
# Override agent parameters
python train_rsl_rl.py --algorithm bpo agent.seed=5 agent.num_steps_per_env=24

# Override algorithm hyperparameters
python train_rsl_rl.py --algorithm bpo agent.algorithm.clip_param=0.3

# Override environment settings
python train_rsl_rl.py --num_envs 2048
```


## TTRL: Test-Time Reinforcement Learning

[TTRL](https://arxiv.org/abs/2504.16084) performs RL on data without ground-truth labels by using majority voting as a reward signal. It is built on [verl](https://github.com/volcengine/verl) (located in `src/verl/`).

### Data Preprocessing

Training data is in `data/`. To convert JSON data to Parquet format for verl:
```bash
cd data
python preprocess.py
cd ..
```

### Running Experiments

Example scripts are in `ttrl/`, organized by model:

```bash
# Qwen2.5 on AIME 2024
bash ttrl/Qwen2.5/aime.sh

# Qwen2.5-0.5B on AIME 2024
bash ttrl/Qwen2.5-0.5B/aime.sh

# Qwen2.5-Math on MATH
bash ttrl/Qwen2.5-Math/math.sh

# LLaMA3.1-Instruct on AMC
bash ttrl/LLaMA3.1-Instruct/amc.sh
```

Available models and benchmarks:
- **Qwen2.5-0.5B**: `aime.sh`, `math.sh`, `amc.sh`
- **Qwen2.5**: `aime.sh`, `math.sh`, `amc.sh`
- **Qwen2.5-Math**: `aime.sh`, `math.sh`, `amc.sh`
- **LLaMA3.1-Instruct**: `aime.sh`, `math.sh`, `amc.sh`

*All TTRL experiments were conducted on 8 x NVIDIA A100 80GB GPUs.*

### Using GBPO (Bounded Ratio Policy Optimization)

The TTRL scripts support both GRPO and GBPO as advantage estimators. GBPO extends GRPO with a target-ratio mechanism: instead of clipping the probability ratio symmetrically like PPO, GBPO regresses the ratio toward a target derived from how each sample compares to the group median.

To switch between algorithms, edit the `ADVANTAGE` variable in any script:

```bash
# Use GRPO (default in most scripts)
ADVANTAGE="grpo"

# Use GBPO
ADVANTAGE="gbpo"
```

When `ADVANTAGE="gbpo"`, the script automatically sets `POLICY_LOSS_MODE="gbpo"`. Key GBPO hyperparameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `CLIP_RATIO_LOW` | Lower clip bound for target ratio | `0.2` |
| `CLIP_RATIO_HIGH` | Upper clip bound for target ratio | `0.28` |
| `GBPO_TEMPERATURE` | Controls target ratio sharpness (smaller = more binary) | `0.0001` |
| `PPO_EPOCHS` | Number of policy update epochs per batch | `5`-`10` |

The GBPO loss is:

```
scale = sigmoid((score - median(group)) / temperature)
target_ratio = 1 - clip_low + (clip_high + clip_low) * scale
loss = |ratio - target_ratio| * |advantages|
```

Implementation is in `src/verl/trainer/ppo/core_algos.py`.

## Configuration Files

Configuration files are organized in `cfgs/`:
- `cfgs/config.yaml` - Main configuration with defaults
- `cfgs/env/` - Environment configurations
- `cfgs/algo/` - Algorithm configurations

You can modify these files or override values via command line arguments.

## Using Weights & Biases Sweeps

Wandb sweep is a common tool for hyperparameter comparison and visualization (https://docs.wandb.ai/models/sweeps).

The repository includes pre-configured wandb sweep files in the `sweeps/` directory. To run a sweep:

1. **Initialize a sweep:**
   ```bash
   wandb sweep sweeps/atari/bpo_seed.yaml
   ```
   This will output a sweep ID (e.g., `your-entity/your-project/sweep-id`)

2. **Run a sweep agent:**
   ```bash
   wandb agent your-entity/your-project/sweep-id
   ```

3. **Run multiple agents in parallel:**
   ```bash
   # Run 4 agents in parallel
   for i in {1..4}; do
     wandb agent your-entity/your-project/sweep-id &
   done
   ```

4. **Example submission to SLURM cluster like ETH Euler:**

   Check ```shell_scripts/train_job_bpo```, change the final line with ```wandb agent your-entity/your-project/sweep-id```,
   then run
   ```bash
   sbatch train_job_bpo
   ```

### Example Sweep Files

- **Atari sweeps:** `sweeps/atari/`
  - `bpo_seed.yaml` - BPO with seed sweep (fixed tuned hyperparameters, only randomize seeds)
  - `bpo_sweep.yaml` - BPO hyperparameter sweep
  - `ppo_seed.yaml` - PPO with seed sweep

- **MuJoCo sweeps:** `sweeps/mujoco/`
  - Organized by environment (Ant-V4, Hopper-V4, Humanoid-V4, Swimmer-V4, etc.)
  - Each environment has sweeps for different algorithms (bpo, ppo, sac)

- **Isaaclab sweeps:** `sweeps/isaaclab/`
  - Organized by environment (anymal_c_rough, g1_rough, etc.)
  - Each environment has sweeps for different algorithms (bpo, ppo)


