import torch
from torch import nn
from torch.nn import functional as functional


class MLP(nn.Module):
    def __init__(
        self, input_size: int, hidden_sizes: list[int], output_size: int
    ) -> None:
        """Initialize MLP network.

        Args:
            input_size: Number of input features.
            hidden_sizes: List of hidden layer sizes.
            output_size: Number of output features.
        """
        super().__init__()

        self.fully_connected_layers = []
        for i, next_size in enumerate(hidden_sizes):
            fully_connected_layer = nn.Linear(input_size, next_size)
            self.add_module(f"fully_connected_layer_{i}", fully_connected_layer)
            self.fully_connected_layers.append(fully_connected_layer)
            input_size = next_size

        self.output_layer = nn.Linear(input_size, output_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:  # noqa: D102
        for fully_connected_layer in self.fully_connected_layers:
            state = functional.relu(fully_connected_layer(state))
        output = self.output_layer(state)
        return output


class Critic(nn.Module):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        num_quantiles: int,
        num_critics: int,
        hidden_size: list[int] = None,
    ) -> None:
        """Initialize the critic networks.

        Args:
            observation_size: Dimension of the observation/state space.
            num_actions: Dimension of the action space.
            num_quantiles: Number of quantiles to estimate for the value distribution.
            num_critics: Number of critic networks to use in the ensemble.
            hidden_size: List containing the sizes of hidden layers. Defaults to [512, 512, 512].
        """
        super().__init__()
        if hidden_size is None:
            hidden_size = [512, 512, 512]

        self.q_networks = []
        self.num_quantiles = num_quantiles
        self.num_critics = num_critics

        for i in range(self.num_critics):
            critic_net = MLP(
                observation_size + num_actions, hidden_size, self.num_quantiles
            )
            self.add_module(f"critic_net_{i}", critic_net)
            self.q_networks.append(critic_net)

    def forward(  # noqa: D102
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        network_input = torch.cat((state, action), dim=1)
        quantiles = torch.stack(
            tuple(critic(network_input) for critic in self.q_networks), dim=1
        )
        return quantiles
