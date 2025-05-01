import torch
from torch import nn


class Network(nn.Module):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_size: list[int] = None,
    ) -> None:
        """Initialize the network.

        Args:
            observation_size: Dimension of the observation/state space.
            num_actions: Dimension of the action space. For DQN, this is the number of discrete actions.
            hidden_size: List containing the sizes of hidden layers. Defaults to [256, 256].
        """
        super().__init__()
        if hidden_size is None:
            hidden_size = [256, 256]

        self.net = nn.Sequential(
            nn.Linear(observation_size, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(state)
