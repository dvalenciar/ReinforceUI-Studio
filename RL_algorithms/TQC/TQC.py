"""Algorithm name: TQC

Paper Name: Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics
Paper link: https://arxiv.org/abs/2005.04269
Taxonomy: Off policy > Actor-Critic > Continuous action space
"""

import copy
import os
import numpy as np
import torch
from RL_memory.memory_buffer import MemoryBuffer
from RL_algorithms.TQC.networks import Actor, Critic


class TQC:
    def __init__(
        self, observation_size: int, action_num: int, hyperparameters: dict
    ) -> None:
        """Initializes the TQC algorithm.

        Args:
            observation_size (int): The size of the observation space.
            action_num (int): The number of actions.
            hyperparameters (dict): The hyperparameters used to initialize the algorithm:
                "log_std_bounds" (list): The bounds for the log standard deviation.
                "n_quantiles" (int): The number of quantiles.
                "num_critics" (int): The number of critics.
                "gamma" (float): The discount factor.
                "tau" (float): The target network update rate.
                "top_quantiles_to_drop" (int): The number of top quantiles to drop.
                "actor_lr" (float): The actor learning rate.
                "critic_lr" (float): The critic learning rate.
                "alpha_lr" (float): The alpha learning rate.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_std_bounds = list(hyperparameters.get("log_std_bounds"))
        n_quantiles = int(hyperparameters.get("n_quantiles"))
        num_critics = int(hyperparameters.get("num_critics"))

        self.gamma = float(hyperparameters.get("gamma"))
        self.tau = float(hyperparameters.get("tau"))
        self.top_quantiles_to_drop = int(hyperparameters.get("top_quantiles_to_drop"))
        self.actor_lr = float(hyperparameters.get("actor_lr"))
        self.critic_lr = float(hyperparameters.get("critic_lr"))
        self.alpha_lr = float(hyperparameters.get("alpha_lr"))

        self.actor_net = Actor(
            observation_size, action_num, log_std_bounds=log_std_bounds
        ).to(self.device)
        self.critic_net = Critic(
            observation_size, action_num, n_quantiles, num_critics
        ).to(self.device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(self.device)

        self.learn_counter = 0
        self.policy_update_freq = 1
        self.target_entropy = -action_num

        self.quantiles_total = (
            self.critic_net.num_quantiles * self.critic_net.num_critics
        )

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=self.actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=self.critic_lr
        )

        # Set to initial alpha to 1.0 according to other baselines.
        init_temperature = 1.0
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

    def select_action_from_policy(
        self,
        state: np.ndarray,
        evaluation: bool = False,
        noise_scale: float = 0,
    ) -> np.ndarray:
        """Select action from policy.

        Args:
            state: Input state
            evaluation: When True, the policy is being evaluated
            noise_scale: No use in this algorithm

        Note:
            when evaluating this algorithm we need to select tanh(mean) as action
            so _, _, action = self.actor_net(state_tensor)

        Returns:
            Action array to be applied to the environment
        """
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            if evaluation is False:
                (
                    action,
                    _,
                    _,
                ) = self.actor_net(state_tensor)
            else:
                (
                    _,
                    _,
                    action,
                ) = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    @property
    def alpha(self) -> torch.Tensor:
        """Returns the exponential of self.log_alpha"""
        return self.log_alpha.exp()

    def quantile_huber_loss_f(
        self, quantiles: torch.Tensor, samples: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the quantile Huber loss for a given set of quantiles and samples.

        Args:
            quantiles (torch.Tensor): A tensor of shape (batch_size, num_nets, num_quantiles) representing the quantiles.
            samples (torch.Tensor): A tensor of shape (batch_size, num_samples) representing the samples.

        Returns:
            torch.Tensor: The quantile Huber loss.

        """
        pairwise_delta = (
            samples[:, None, None, :] - quantiles[:, :, :, None]
        )  # batch x nets x quantiles x samples
        abs_pairwise_delta = torch.abs(pairwise_delta)
        huber_loss = torch.where(
            abs_pairwise_delta > 1,
            abs_pairwise_delta - 0.5,
            pairwise_delta**2 * 0.5,
        )

        n_quantiles = quantiles.shape[2]

        tau = (
            torch.arange(n_quantiles, device=pairwise_delta.device).float()
            / n_quantiles
            + 1 / 2 / n_quantiles
        )
        loss = (
            torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float())
            * huber_loss
        ).mean()
        return loss

    def _update_critics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        batch_size = len(states)
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor_net(next_states)

            # compute and cut quantiles at the next state
            # batch x nets x quantiles
            target_q_values = self.target_critic_net(next_states, next_actions)
            sorted_target_q_values, _ = torch.sort(
                target_q_values.reshape(batch_size, -1)
            )
            top_quantile_target_q_values = sorted_target_q_values[
                :, : self.quantiles_total - self.top_quantiles_to_drop
            ]

            # compute target
            q_target = rewards + (1 - dones) * self.gamma * (
                top_quantile_target_q_values - self.alpha * next_log_pi
            )

        q_values = self.critic_net(states, actions)
        critic_loss_total = self.quantile_huber_loss_f(q_values, q_target)

        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        return critic_loss_total.item()

    def _update_actor(self, states: torch.Tensor) -> tuple[float, float]:
        new_action, log_pi, _ = self.actor_net(states)

        mean_qf_pi = self.critic_net(states, new_action).mean(2).mean(1, keepdim=True)
        actor_loss = (self.alpha * log_pi - mean_qf_pi).mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()

        # update the temperature
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss.item(), alpha_loss.item()

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> None:
        """Train actor and critic networks using experiences from memory.

        Args:
            memory: Replay buffer containing experiences
            batch_size: Number of experiences to sample
        """
        self.learn_counter += 1

        experiences = memory.sample_experience(batch_size)
        states, actions, rewards, next_states, dones = experiences

        # Convert into tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        rewards = rewards.reshape(batch_size, 1)
        dones = dones.reshape(batch_size, 1)

        # Update the Critics
        self._update_critics(states, actions, rewards, next_states, dones)

        # Update the Actor
        self._update_actor(states)

        if self.learn_counter % self.policy_update_freq == 0:
            for param, target_param in zip(
                self.critic_net.parameters(),
                self.target_critic_net.parameters(),
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
