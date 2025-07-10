import torch.nn as nn
from typing import Any


class CriticMLflowWrapperTD3_SAC(nn.Module):  # noqa: N801
    def __init__(self, critic: nn.Module, obs_dim: int) -> None:
        """Mlflow wrapper for TD3/SAC critic networks.

        Args:
            critic (nn.Module): The critic network.
            obs_dim (int): Dimension of the observation space.
        """
        super().__init__()
        self.critic = critic
        self.obs_dim = obs_dim

    def forward(self, x: Any) -> Any:
        """Forward pass for the TD3/SAC critic wrapper."""
        state = x[:, : self.obs_dim]
        action = x[:, self.obs_dim :]
        q1, _ = self.critic(state, action)
        return q1


class CriticMLflowWrapperCtd4(nn.Module):
    def __init__(self, critic: nn.Module, obs_dim: int) -> None:
        """Mlflow wrapper for CTD4 critic networks.

        Args:
            critic (nn.Module): The critic network.
            obs_dim (int): Dimension of the observation space.
        """
        super().__init__()
        self.critic = critic
        self.obs_dim = obs_dim

    def forward(self, x: Any) -> Any:
        """Forward pass for the CTD4 critic wrapper."""
        state = x[:, : self.obs_dim]
        action = x[:, self.obs_dim :]
        mean, std = self.critic(state, action)
        return mean


class CriticMLflowWrapperDDPG(nn.Module):
    def __init__(self, critic: nn.Module, obs_dim: int) -> None:
        """Mlflow wrapper for DDPG critic networks.

        Args:
            critic (nn.Module): The critic network.
            obs_dim (int): Dimension of the observation space.
        """
        super().__init__()
        self.critic = critic
        self.obs_dim = obs_dim

    def forward(self, x: Any) -> Any:
        """Forward pass for the DDPG critic wrapper."""
        state = x[:, : self.obs_dim]
        action = x[:, self.obs_dim :]
        q1 = self.critic(state, action)
        return q1


class CriticMLflowWrapperTQC(nn.Module):
    def __init__(self, critic: nn.Module, obs_dim: int) -> None:
        """Mlflow wrapper for TQC critic networks.

        Args:
            critic (nn.Module): The critic network.
            obs_dim (int): Dimension of the observation space.
        """
        super().__init__()
        self.critic = critic
        self.obs_dim = obs_dim

    def forward(self, x: Any) -> Any:
        """Forward pass for the TQC critic wrapper."""
        state = x[:, : self.obs_dim]
        action = x[:, self.obs_dim :]
        quantiles = self.critic(state, action)
        return quantiles


class ActorMLflowWrapperSAC(nn.Module):
    def __init__(self, actor: nn.Module, obs_dim: int) -> None:
        """Mlflow wrapper for SAC actor networks.

        Args:
            actor (nn.Module): The actor network.
            obs_dim (int): Dimension of the observation space.
        """
        super().__init__()
        self.actor = actor
        self.obs_dim = obs_dim

    def forward(self, x: Any) -> Any:
        """Forward pass for the SAC actor wrapper."""
        state = x[:, : self.obs_dim]
        action, _, _ = self.actor(state)
        return action


class ActorMLflowWrapperTQC(nn.Module):
    def __init__(self, actor: nn.Module, obs_dim: int) -> None:
        """Mlflow wrapper for TQC actor networks.

        Args:
            actor (nn.Module): The actor network.
            obs_dim (int): Dimension of the observation space.
        """
        super().__init__()
        self.actor = actor
        self.obs_dim = obs_dim

    def forward(self, x: Any) -> Any:
        """Forward pass for the TQC actor wrapper."""
        state = x[:, : self.obs_dim]
        _, _, action = self.actor(state)
        return action


class ActorMLflowWrapperPPO(nn.Module):
    def __init__(self, actor: nn.Module, obs_dim: int) -> None:
        """Mlflow wrapper for PPO actor networks.

        Args:
            actor (nn.Module): The actor network.
            obs_dim (int): Dimension of the observation space.
        """
        super().__init__()
        self.actor = actor
        self.obs_dim = obs_dim

    def forward(self, x: Any) -> Any:
        """Forward pass for the PPO actor wrapper."""
        state = x[:, : self.obs_dim]
        action, _ = self.actor(state)
        return action
