import gymnasium as gym
from stable_baselines3.common.vec_env import VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import ale_py

gym.register_envs(ale_py)

@hydra.main(version_base=None, config_path="cfgs", config_name="config")
def main(cfg : DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(cfg))
    wandb_cfg = OmegaConf.to_container(cfg.wandb, resolve=True)
    wandb_cfg['config'] = config
    run = wandb.init(**wandb_cfg, sync_tensorboard=True )
    env = instantiate(cfg.env)
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 1000000 == 0,
        video_length=300,
    )

    algo_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
    algo_cfg['tensorboard_log'] = f"runs/{run.id}"
    model = instantiate(algo_cfg, env=env)
    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        callback=WandbCallback(
            gradient_save_freq=cfg.training.gradient_save_freq,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    run.finish()


if __name__ == "__main__":
    main()
