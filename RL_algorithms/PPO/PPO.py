import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from RL_algorithms.PPO.networks import Actor, Critic

class PPO:
    def __init__(self, observation_size, action_num, hyperparameters):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_net = Actor(observation_size, action_num).to(self.device)
        self.critic_net = Critic(observation_size).to(self.device)

        self.gamma = float(hyperparameters.get("gamma"))
        self.actor_lr = float(hyperparameters.get("actor_lr"))
        self.critic_lr = float(hyperparameters.get("critic_lr"))
        self.eps_clip = float(hyperparameters.get("eps_clip"))
        self.updates_per_iteration = int(hyperparameters.get("updates_per_iteration"))

        self.action_num = action_num

        self.actor_net_optimiser = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)
        self.critic_net_optimiser = torch.optim.Adam(self.critic_net.parameters(), lr=self.critic_lr)

        self.cov_var = torch.full(size=(self.action_num,), fill_value=0.5).to(self.device)
        self.cov_mat = torch.diag(self.cov_var)

    def select_action_from_policy(self, state: np.ndarray) ->  tuple[np.ndarray, np.ndarray]:
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            mean = self.actor_net(state_tensor)
            dist = MultivariateNormal(mean, self.cov_mat)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            action = action.cpu().data.numpy().flatten()
            log_prob = log_prob.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action, log_prob

    def _evaluate_policy(self, state, action):
        v = self.critic_net(state).squeeze()  # Ensure shape is (batch_size,)
        mean = self.actor_net(state) # Expected shape: (batch_size, action_dim)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_prob = dist.log_prob(action) # Expected shape: (batch_size,)
        return v, log_prob

    def _calculate_rewards_to_go(self, batch_rewards: torch.Tensor, batch_dones: torch.Tensor) -> torch.Tensor:
        rtgs = []
        discounted_reward = 0.0
        for reward, done in zip(reversed(batch_rewards), reversed(batch_dones)):
            discounted_reward = reward + self.gamma * (1 - done) * discounted_reward
            rtgs.insert(0, discounted_reward)
        return torch.FloatTensor(rtgs).to(self.device)

    def train_policy(self, memory):
        # Get the experiences from the memory and flush it
        experiences = memory.return_flushed_memory()
        states, actions, rewards, _, dones, log_probs = experiences

        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        dones = torch.FloatTensor(np.asarray(dones)).to(self.device)
        log_probs = torch.FloatTensor(np.asarray(log_probs)).to(self.device)

        rtgs = self._calculate_rewards_to_go(rewards, dones)  # the shape here is torch.Size([2400])
        v, _ = self._evaluate_policy(states, actions) # the shape here is torch.Size([2400])

        advantages = rtgs.detach() - v.detach()
        #advantages = rtgs - v
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10) # Expected shape: (batch_size,)

        for _ in range(self.updates_per_iteration):

            v, curr_log_probs = self._evaluate_policy(states, actions)

            # Calculate ratios
            ratios = torch.exp(curr_log_probs - log_probs.detach()) # the shape here is torch.Size([2400])
            #ratios = torch.exp(curr_log_probs - log_probs)

            # Finding Surrogate Loss
            surrogate_loss_one = ratios * advantages
            surrogate_loss_two = (torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages)

            actor_loss = -torch.min(surrogate_loss_one, surrogate_loss_two).mean()
            critic_loss = F.mse_loss(v, rtgs)

            self.actor_net_optimiser.zero_grad()
            actor_loss.backward()
            self.actor_net_optimiser.step()

            self.critic_net_optimiser.zero_grad()
            critic_loss.backward()
            self.critic_net_optimiser.step()


    def save_models(self, filename: str, filepath: str) -> None:
        dir_exists = os.path.exists(filepath)
        if not dir_exists:
            os.makedirs(filepath)

        torch.save(self.actor_net.state_dict(), f"{filepath}/{filename}_actor.pht")
        torch.save(self.critic_net.state_dict(), f"{filepath}/{filename}_critic.pht")

    def load_models(self, filename: str, filepath: str) -> None:
        self.actor_net.load_state_dict(
            torch.load(
                f"{filepath}/{filename}_actor.pht",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.actor_net.eval()
        self.critic_net.load_state_dict(
            torch.load(
                f"{filepath}/{filename}_critic.pht",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.critic_net.eval()

