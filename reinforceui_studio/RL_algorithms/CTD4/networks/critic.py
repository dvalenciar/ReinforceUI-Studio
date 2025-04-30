import torch
from torch import nn


class Critic(nn.Module):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_size: list[int] = None,
    ) -> None:
        """Initialize the critic network.

        Args:
            observation_size: Dimension of the observation/state space.
            num_actions: Dimension of the action space.
            hidden_size: List containing the sizes of hidden layers. Defaults to [256, 256].
        """
        super().__init__()
        if hidden_size is None:
            hidden_size = [256, 256]

        self.mean_layer = nn.Sequential(
            nn.Linear(observation_size + num_actions, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], 1),
        )

        self.std_layer = nn.Sequential(
            nn.Linear(observation_size + num_actions, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], 1),
            nn.Softplus(),
        )

    def forward(  # noqa: D102
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.cat([state, action], dim=1)
        return self.mean_layer(obs_action), self.std_layer(obs_action) + 1e-6
