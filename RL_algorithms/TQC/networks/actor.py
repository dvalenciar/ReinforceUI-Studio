import torch
from torch import nn
from torch import distributions as pyd
from torch.distributions.transforms import TanhTransform
from torch.distributions.transformed_distribution import (
    TransformedDistribution,
)


class StableTanhTransform(TanhTransform):
    """Stable version of the TanhTransform for numerical stability.

    Implements a stable version of the arctanh function for the inverse
    transform to avoid numerical issues near the boundaries.
    """

    def __init__(self, cache_size: int = 1) -> None:
        """Initialize a stable tanh transform.

        Args:
            cache_size: Size of cache for the transform. Defaults to 1.
        """
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x: torch.Tensor) -> torch.Tensor:
        """Implement a numerically stable arctanh function.

        Args:
            x: Input tensor.

        Returns:
            Arctanh of the input tensor.
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other: object) -> bool:
        """Compare if two transforms are the same type.

        Args:
            other: Another transform to compare with.

        Returns:
            Boolean indicating if the other transform is a StableTanhTransform.
        """
        return isinstance(other, StableTanhTransform)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Compute the inverse of the tanh transform.

        Args:
            y: Input tensor.

        Returns:
            Inverse transformed tensor.
        """
        return self.atanh(y)


class SquashedNormal(TransformedDistribution):
    """Normal distribution transformed by tanh for bounded actions.

    Creates a transformed distribution that samples from a normal distribution
    and then applies a tanh transform to bound values between -1 and 1.
    """

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        """Initialize the squashed normal distribution.

        Args:
            loc: Mean of the base normal distribution.
            scale: Standard deviation of the base normal distribution.
        """
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)

        transforms = [StableTanhTransform()]
        super().__init__(self.base_dist, transforms, validate_args=False)

    @property
    def mean(self) -> torch.Tensor:
        """Calculate the mean of the transformed distribution.

        Returns:
            Mean of the squashed normal distribution.
        """
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class Actor(nn.Module):
    """Diagonal Gaussian Actor network for the TQC algorithm.

    Implements a stochastic policy using a diagonal Gaussian distribution
    with tanh squashing to ensure bounded actions, suitable for continuous
    control tasks.
    """

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_size: list[int] = None,
        log_std_bounds: list[int] = None,
    ) -> None:
        """Initialize the actor network.

        Args:
            observation_size: Dimension of the observation/state space.
            num_actions: Dimension of the action space.
            hidden_size: List containing the sizes of hidden layers. Defaults to [256, 256].
            log_std_bounds: List containing the bounds for the log standard deviation. Defaults to [-20, 2].
        """
        super().__init__()
        if hidden_size is None:
            hidden_size = [256, 256]
        if log_std_bounds is None:
            log_std_bounds = [-20, 2]

        self.hidden_size = hidden_size
        self.log_std_bounds = log_std_bounds

        self.act_net = nn.Sequential(
            nn.Linear(observation_size, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
        )

        self.mean_linear = nn.Linear(self.hidden_size[1], num_actions)
        self.log_std_linear = nn.Linear(self.hidden_size[1], num_actions)

    def forward(  # noqa: D102
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.act_net(state)
        mu = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)

        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        sample = dist.rsample()
        log_pi = dist.log_prob(sample).sum(-1, keepdim=True)

        return sample, log_pi, dist.mean
