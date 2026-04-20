import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomMLP(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, net_arch, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        obs_dim = observation_space.shape[0]
        model_list = []
        model_list
        net_arch = [obs_dim] + net_arch
        for i in range(len(net_arch) - 1):
            model_list.append(nn.Linear(net_arch[i], net_arch[i + 1]))
            model_list.append(nn.ReLU())
        model_list.append(nn.Linear(net_arch[-1], features_dim))

        self.mlp = nn.Sequential(*model_list)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.mlp(observations)