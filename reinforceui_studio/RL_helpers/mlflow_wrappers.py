import torch.nn as nn

class CriticMLflowWrapperTD3_SAC(nn.Module):
    def __init__(self, critic, obs_dim):
        super().__init__()
        self.critic = critic
        self.obs_dim = obs_dim

    def forward(self, x):
        state = x[:, :self.obs_dim]
        action = x[:, self.obs_dim:]
        q1, _ = self.critic(state, action)
        return q1

class CriticMLflowWrapperCtd4(nn.Module):
    def __init__(self, critic, obs_dim):
        super().__init__()
        self.critic = critic
        self.obs_dim = obs_dim

    def forward(self, x):
        state = x[:, :self.obs_dim]
        action = x[:, self.obs_dim:]
        mean, std = self.critic(state, action)
        return mean

class CriticMLflowWrapperDDPG(nn.Module):
    def __init__(self, critic, obs_dim):
        super().__init__()
        self.critic = critic
        self.obs_dim = obs_dim

    def forward(self, x):
        state = x[:, :self.obs_dim]
        action = x[:, self.obs_dim:]
        q1 = self.critic(state, action)
        return q1

class CriticMLflowWrapperTQC(nn.Module):
    def __init__(self, critic, obs_dim):
        super().__init__()
        self.critic = critic
        self.obs_dim = obs_dim

    def forward(self, x):
        state = x[:, :self.obs_dim]
        action = x[:, self.obs_dim:]
        quantiles = self.critic(state, action)
        return quantiles


class ActorMLflowWrapperSAC(nn.Module):
    def __init__(self, actor, obs_dim):
        super().__init__()
        self.actor = actor
        self.obs_dim = obs_dim

    def forward(self, x):
        state = x[:, :self.obs_dim]
        action, _, _ = self.actor(state)
        return action

class ActorMLflowWrapperTQC(nn.Module):
    def __init__(self, actor, obs_dim):
        super().__init__()
        self.actor = actor
        self.obs_dim = obs_dim

    def forward(self, x):
        state = x[:, :self.obs_dim]
        _, _, action = self.actor(state)
        return action

class ActorMLflowWrapperPPO(nn.Module):
    def __init__(self, actor, obs_dim):
        super().__init__()
        self.actor = actor
        self.obs_dim = obs_dim

    def forward(self, x):
        state = x[:, :self.obs_dim]
        action, _ = self.actor(state)
        return action