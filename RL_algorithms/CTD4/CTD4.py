"""Algorithm name: CTD4

Paper Name: A Deep Continuous Distributional Actor-Critic Agent with a Kalman Fusion of Multiple Critics.
Paper link: https://arxiv.org/abs/2405.02576
Taxonomy: Off policy > Actor-Critic > Continuous action space
"""

import copy
import os
import numpy as np
import torch
from RL_memory.memory_buffer import MemoryBuffer
from RL_algorithms.CTD4.networks import Actor, Critic


class CTD4:
    def __init__(
        self, observation_size: int, action_num: int, hyperparameters: dict
    ) -> None:
        """Initialize the CTD4 agent.

        Args:
            observation_size: Dimension of the state space
            action_num: Dimension of the action space
            hyperparameters: Dictionary containing algorithm parameters:
                gamma: Discount factor
                tau: Soft update parameter
                actor_lr: Learning rate for actor network
                critic_lr: Learning rate for critic networks
                ensemble_size: Number of critic networks in the ensemble
                policy_noise_decay: Decay rate for target policy noise
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = float(hyperparameters.get("gamma"))
        self.tau = float(hyperparameters.get("tau"))
        self.actor_lr = float(hyperparameters.get("actor_lr"))
        self.critic_lr = float(hyperparameters.get("critic_lr"))
        self.ensemble_size = int(hyperparameters.get("ensemble_size"))

        self.actor_net = Actor(observation_size, action_num).to(self.device)
        self.target_actor_net = copy.deepcopy(self.actor_net).to(self.device)

        self.ensemble_critics = torch.nn.ModuleList(
            [
                Critic(observation_size, action_num).to(self.device)
                for _ in range(self.ensemble_size)
            ]
        )
        self.target_ensemble_critics = copy.deepcopy(self.ensemble_critics).to(
            self.device
        )

        self.noise_clip = 0.5
        self.target_policy_noise_scale = 0.2
        self.policy_noise_decay = float(hyperparameters.get("policy_noise_decay"))
        self.min_policy_noise = 0.0

        self.learn_counter = 0
        self.policy_update_freq = 2

        self.action_num = action_num

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=self.actor_lr
        )

        self.ensemble_critics_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=self.critic_lr)
            for critic in self.ensemble_critics
        ]

    def select_action_from_policy(
        self,
        state: np.ndarray,
        evaluation: bool = False,
        noise_scale: float = 0.1,
    ) -> np.ndarray:
        """Select action from policy.

        Args:
            state: Input state
            evaluation: When False, no exploration noise is added
            noise_scale: Scale of the exploration noise

        Returns:
            Action array to be applied to the environment
        """
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            action = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
            if not evaluation:
                noise = np.random.normal(0, scale=noise_scale, size=self.action_num)
                action = action + noise
                action = np.clip(action, -1, 1)
        self.actor_net.train()
        return action

    def _fusion_kalman(
        self,
        std_1: torch.Tensor,
        mean_1: torch.Tensor,
        std_2: torch.Tensor,
        mean_2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        kalman_gain = (std_1**2) / (std_1**2 + std_2**2)
        fusion_mean = mean_1 + kalman_gain * (mean_2 - mean_1)
        fusion_variance = (1 - kalman_gain) * std_1**2 + kalman_gain * std_2**2 + 1e-6
        fusion_std = torch.sqrt(fusion_variance)
        return fusion_mean, fusion_std

    def _kalman(
        self, u_set: list[torch.Tensor], std_set: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if len(u_set) == 0 or len(std_set) == 0:
            raise ValueError("Input lists must not be empty.")
        if len(u_set) == 1:
            return u_set[0], std_set[0]

        fusion_u, fusion_std = u_set[0], std_set[0]
        for i in range(1, len(u_set)):
            fusion_u, fusion_std = self._fusion_kalman(
                fusion_std, fusion_u, std_set[i], u_set[i]
            )
        return fusion_u, fusion_std

    def _update_critics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> list[float]:

        with torch.no_grad():
            next_actions = self.target_actor_net(next_states)

            target_noise = self.target_policy_noise_scale * torch.randn_like(
                next_actions
            )
            target_noise = torch.clamp(target_noise, -self.noise_clip, self.noise_clip)
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            u_set = []
            std_set = []

            for target_critic_net in self.target_ensemble_critics:
                u, std = target_critic_net(next_states, next_actions)
                u_set.append(u)
                std_set.append(std)

            fusion_u, fusion_std = self._kalman(u_set, std_set)

            # Create the target distribution = aX+b
            u_target = rewards + self.gamma * fusion_u * (1 - dones)
            std_target = self.gamma * fusion_std
            target_distribution = torch.distributions.normal.Normal(
                u_target, std_target + 1e-6
            )

        critic_loss_totals = []

        for critic_net, critic_net_optimiser in zip(
            self.ensemble_critics, self.ensemble_critics_optimizers
        ):
            u_current, std_current = critic_net(states, actions)
            current_distribution = torch.distributions.normal.Normal(
                u_current, std_current + 1e-6
            )

            # Compute each critic loss
            critic_individual_loss = torch.distributions.kl.kl_divergence(
                current_distribution, target_distribution
            ).mean()

            critic_net_optimiser.zero_grad()
            critic_individual_loss.backward()
            critic_net_optimiser.step()

            critic_loss_totals.append(critic_individual_loss.item())
        return critic_loss_totals

    def _update_actor(self, states: torch.Tensor) -> float:
        actor_q_u_set = []
        actor_q_std_set = []

        actions = self.actor_net(states)
        for critic_net in self.ensemble_critics:
            actor_q_u, actor_q_std = critic_net(states, actions)
            actor_q_u_set.append(actor_q_u)
            actor_q_std_set.append(actor_q_std)

        fusion_u_a, _ = self._kalman(actor_q_u_set, actor_q_std_set)
        actor_loss = -fusion_u_a.mean()

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
        self.learn_counter += 1

        self.target_policy_noise_scale *= self.policy_noise_decay
        self.target_policy_noise_scale = max(
            self.min_policy_noise, self.target_policy_noise_scale
        )

        experiences = memory.sample_experience(batch_size)
        states, actions, rewards, next_states, dones = experiences

        # Convert into tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Reshape to batch_size
        rewards = rewards.reshape(batch_size, 1)
        dones = dones.reshape(batch_size, 1)

        self._update_critics(states, actions, rewards, next_states, dones)

        if self.learn_counter % self.policy_update_freq == 0:
            # Update Actor
            self._update_actor(states)

            for critic_net, target_critic_net in zip(
                self.ensemble_critics, self.target_ensemble_critics
            ):
                for param, target_param in zip(
                    critic_net.parameters(), target_critic_net.parameters()
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
        torch.save(
            self.ensemble_critics.state_dict(),
            f"{filepath}/{filename}_ensemble_critic.pht",
        )

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
        self.ensemble_critics.load_state_dict(
            torch.load(
                f"{filepath}/{filename}_ensemble_critic.pht",
                map_location=self.device,
                weights_only=True,
            )
        )
