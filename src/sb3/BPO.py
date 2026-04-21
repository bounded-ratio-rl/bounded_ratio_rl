import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

# from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from .policy import BPOPolicy, BPOCnnPolicy
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from torch.nn import functional as F

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor

SelfBPO = TypeVar("SelfBPO", bound="BPO")


class BPO(OnPolicyAlgorithm):
    """
    Bounded Policy Optimization algorithm (BPO) (clip version)

    Paper: https://arxiv.org/abs/2604.18578v1

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param lambda: The temperature parameter
    :param loss_type: type of loss function, include Total Variation (TV), Mean Square (MSE), Forward KL (FKL) and Chi-squared (Chi)
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param u_coef: median value function coefficient for the loss calculation
    :param use_median: Whether to use median value function for the loss calculation
    :param clip_ratio: The clipping ratio, it can be a function
        of the current progress remaining (from 1 to 0)
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`ppo_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
        "BPOPolicy": BPOPolicy,
        "BPOCnnPolicy": BPOCnnPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[BPOPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lambda_: float = 0.001,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        u_coef: float = 0.5,
        use_median: bool = True,
        clip_ratio: Union[float, Schedule] = 0.3,
        max_grad_norm: float = 1.0,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization


        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lambda_ = lambda_
        self.use_median = use_median
        # Convert clip_ratio to schedule if it's a float
        if isinstance(clip_ratio, (int, float)):
            # Constant schedule
            self.clip_ratio_schedule = lambda _: float(clip_ratio)
        else:
            # Callable
            self.clip_ratio_schedule = clip_ratio
        self.u_coef = u_coef

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Get current clip_ratio from schedule
        # Calculate progress_remaining: 1.0 at start, 0.0 at end
        if hasattr(self, "_total_timesteps") and self._total_timesteps > 0:
            progress_remaining = 1.0 - (self.num_timesteps / self._total_timesteps)
        else:
            # Fallback if total_timesteps not available yet
            progress_remaining = 1.0
        clip_ratio = self.clip_ratio_schedule(progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, u_losses = [], [], []

        continue_training = True
        self.max_adv_abs = th.tensor(0.0, device=self.device)
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # First train values

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, u_values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )


                values = values.flatten()
                u_values = u_values.flatten()

                # value loss function
                q_values = rollout_data.returns.flatten()
                target_values = rollout_data.returns.flatten()
                value_loss = F.mse_loss(target_values, values)
                value_losses.append(value_loss.item())

                # u loss
                if self.use_median:
                    adv = (q_values - u_values) / self.lambda_
                    u_loss = th.mean(
                        th.nn.functional.softplus(adv * self.lambda_)
                        - 0.5 * adv * self.lambda_
                    )
                    u_losses.append(u_loss.item())
                else:
                    adv = (q_values - values) / self.lambda_
                    u_loss = th.zeros_like(value_loss)
                    u_losses.append(0.0)

                # policz loss
                with th.no_grad():
                    adv_d = adv.detach()
                target_ratio = (
                    1
                    - clip_ratio
                    + 2 * clip_ratio * th.nn.functional.sigmoid(adv_d)
                )
                target_log_ratio = th.log(target_ratio)
                log_ratio = log_prob.flatten() - rollout_data.old_log_prob.flatten()
                ratio = th.exp(log_ratio)
                policy_loss = th.mean(
                    th.abs((ratio - target_ratio).flatten() * (q_values - values.detach()))
                )
                pg_losses.append(policy_loss.item())

                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                p_coef = 1.0
                loss = p_coef * policy_loss + self.vf_coef * value_loss + self.u_coef * u_loss + self.ent_coef * entropy_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/u_loss", np.mean(u_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/lambda", self.lambda_)
        self.logger.record("train/clip_ratio", clip_ratio)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, u_values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def learn(
        self: SelfBPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "BPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfBPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
