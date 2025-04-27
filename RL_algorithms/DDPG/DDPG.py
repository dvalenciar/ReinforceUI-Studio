"""Algorithm Name: DDPG: Deep Deterministic Policy Gradient.

Paper name: Continuous control with deep reinforcement learning.
Paper link: https://arxiv.org/abs/1509.02971
Taxonomy: Off policy > Actor-Critic > Continuous action space
"""

import copy
import os
import numpy as np
import torch
import torch.nn.functional as functional
from RL_memory.memory_buffer import MemoryBuffer
from RL_algorithms.DDPG.networks import Actor, Critic


class DDPG:
    def __init__(
        self, observation_size: int, action_num: int, hyperparameters: dict
    ) -> None:
        """Initialize the DDPG agent.

        Args:
            observation_size: Dimension of the state space
            action_num: Dimension of the action space
            hyperparameters: Dictionary containing algorithm parameters like:
                gamma: Discount factor
                tau: Target networks update rate
                actor_lr: Learning rate for actor network
                critic_lr: Learning rate for critic networks
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_net = Actor(observation_size, action_num).to(self.device)
        self.critic_net = Critic(observation_size, action_num).to(self.device)
        self.target_actor_net = copy.deepcopy(self.actor_net).to(self.device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(self.device)

        self.gamma = float(hyperparameters.get("gamma"))
        self.tau = float(hyperparameters.get("tau"))
        self.actor_lr = float(hyperparameters.get("actor_lr"))
        self.critic_lr = float(hyperparameters.get("critic_lr"))

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=self.actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=self.critic_lr
        )

    def select_action_from_policy(
        self,
        state: np.ndarray,
        evaluation: bool = False,
    ) -> np.ndarray:
        """Select action based on current policy.

        Args:
            state: Current state of the environment.
            evaluation: Flag to indicate whether to use exploration. Defaults to False, no used in this algorithm.

        Returns:
            Action to take in the environment as numpy array.
        """
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            action = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            self.target_actor_net.eval()
            next_actions = self.target_actor_net(next_states)
            self.target_actor_net.train()

            target_q_values = self.target_critic_net(next_states, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values = self.critic_net(states, actions)

        critic_loss = functional.mse_loss(q_values, q_target)
        self.critic_net_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_net_optimiser.step()

        return critic_loss.item()

    def _update_actor(self, states: torch.Tensor) -> float:
        self.critic_net.eval()
        actor_q = self.critic_net(states, self.actor_net(states))
        self.critic_net.train()
        actor_loss = -actor_q.mean()
        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        return actor_loss.item()

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> None:
        """Train actor and critic networks using experiences from memory.

        Args:
            memory: Replay buffer containing experiences
            batch_size: Number of experiences to sample
        """
        experiences = memory.sample_experience(batch_size)
        (states, actions, rewards, next_states, dones) = experiences

        # Convert into tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Reshape to batch_size
        rewards = rewards.reshape(batch_size, 1)
        dones = dones.reshape(batch_size, 1)

        # Update Critic
        self._update_critic(states, actions, rewards, next_states, dones)

        # Update Actor
        self._update_actor(states)

        # Update target network params
        for param, target_param in zip(
            self.critic_net.parameters(), self.target_critic_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(
            self.actor_net.parameters(), self.target_actor_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save_models(self, filename: str, filepath: str) -> None:
        """Save actor and critic networks to files.

        Args:
            filename: Base name for the saved model files
            filepath: Directory path where models will be saved
        """
        dir_exists = os.path.exists(filepath)
        if not dir_exists:
            os.makedirs(filepath)

        torch.save(self.actor_net.state_dict(), f"{filepath}/{filename}_actor.pht")
        torch.save(self.critic_net.state_dict(), f"{filepath}/{filename}_critic.pht")

    def load_models(self, filename: str, filepath: str) -> None:
        """Load models previously saved for this algorithm.

        Args:
            filename: Filename of the models, without extension
            filepath: Path to the saved models, usually located in user's home directory
        """
        self.actor_net.load_state_dict(
            torch.load(
                f"{filepath}/{filename}_actor.pht",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.critic_net.load_state_dict(
            torch.load(
                f"{filepath}/{filename}_critic.pht",
                map_location=self.device,
                weights_only=True,
            )
        )
