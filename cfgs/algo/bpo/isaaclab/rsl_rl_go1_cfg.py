# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from src.rsl_rl.rsl_rl_cfg import RslRlBPOOnPolicyRunnerCfg, RslRlBPOActorCriticCfg, RslRlBPOAlgorithmCfg


@configclass
class UnitreeGo1RoughBPORunnerCfg(RslRlBPOOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 500
    experiment_name = "unitree_go1_rough"
    empirical_normalization = False
    policy = RslRlBPOActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlBPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.3,  # 0.25
        entropy_coef=0.001,
        num_learning_epochs=10,
        num_mini_batches=4,
        learning_rate=1e-3,  # 1e-3
        schedule="adaptive",  # adaptive
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class UnitreeGo1FlatBPORunnerCfg(UnitreeGo1RoughBPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "unitree_go1_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]

