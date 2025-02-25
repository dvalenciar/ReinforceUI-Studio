import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, observation_size: int, hidden_size: list[int] = None):
        super().__init__()
        if hidden_size is None:
            hidden_size = [256, 256]

        self.Q = nn.Sequential(
            nn.Linear(observation_size, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.Q(state)
