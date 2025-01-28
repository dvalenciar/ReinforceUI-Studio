import torch
from torch import nn


class Critic(nn.Module):
    def __init__(
        self, observation_size: int, num_actions: int, hidden_size: list[int] = None
    ):
        super().__init__()
        if hidden_size is None:
            hidden_size = [256, 256]

        # Single Q architecture
        self.Q = nn.Sequential(
            nn.Linear(observation_size + num_actions, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.cat([state, action], dim=1)
        return self.Q(obs_action)
