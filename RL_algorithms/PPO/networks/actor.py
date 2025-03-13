import torch
from torch import nn


class Actor(nn.Module):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_size: list[int] = None,
    ):
        super().__init__()
        if hidden_size is None:
            hidden_size = [256, 256]

        self.act_net = nn.Sequential(
            nn.Linear(observation_size, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], num_actions),
            nn.Tanh(),
        )

        self.log_std = nn.Parameter(
            torch.zeros(num_actions)
        )  # Learnable log std

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.act_net(state)
        std = torch.exp(self.log_std)
        return mean, std
