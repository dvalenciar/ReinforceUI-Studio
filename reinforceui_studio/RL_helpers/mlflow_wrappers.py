import torch.nn as nn

class CriticMLflowWrapper(nn.Module):
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