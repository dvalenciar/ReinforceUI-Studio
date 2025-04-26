"""Algorithm Name: DQN.

Paper name: Human-level control through deep reinforcement learning
Paper link: https://www.nature.com/articles/nature14236
Taxonomy: Off policy > Value Based > Discrete action space
"""

import os
import copy
import numpy as np
import torch
import torch.nn.functional as functional
from RL_memory.memory_buffer import MemoryBuffer
from RL_algorithms.DQN.networks import Network


class DQN:
    def __init__(
        self, observation_size: int, action_num: int, hyperparameters: dict
    ) -> None:
        """Initialize the DQN agent.

        Args:
            observation_size: Dimension of the state space
            action_num: Dimension of the action space, in this case, the number of discrete actions
            hyperparameters: Dictionary containing algorithm parameters:
                gamma: Discount factor
                lr: Learning rate
                target_update_freq: Frequency of updating the target network
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = float(hyperparameters.get("gamma"))
        self.lr = float(hyperparameters.get("lr"))

        self.net = Network(observation_size, action_num).to(self.device)
        self.target_net = copy.deepcopy(self.net).to(self.device)

        self.learn_counter = 0
        self.target_update_freq = int(hyperparameters.get("target_update_freq"))

        self.action_num = action_num
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def select_action_from_policy(
        self, state: np.ndarray, evaluation: bool = False
    ) -> np.ndarray:
        """Select action based on current policy.

        Args:
            state: Current state of the environment.
            evaluation: Defaults to False, no used in this algorithm.

        Returns:
            Action to take in the environment as numpy array.
        """
        self.net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            q_values = self.net(state_tensor)
            action = np.argmax(q_values.cpu().data.numpy())
        self.net.train()
        return action

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> None:
        """Train network using experiences from memory.

        Args:
            memory: Replay buffer containing experiences
            batch_size: Number of experiences to sample
        """
        experiences = memory.sample_experience(batch_size)
        states, actions, rewards, next_states, dones = experiences

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Reshape to batch_size
        actions = actions.reshape(batch_size, 1)
        rewards = rewards.reshape(batch_size, 1)
        dones = dones.reshape(batch_size, 1)

        # Compute Q-values for the current states
        q_values = self.net(states)
        taken_action_q_values = q_values.gather(1, actions)

        # Compute Q-values for next states and take max over actions
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            best_next_q_values = torch.max(next_q_values, 1)[0].unsqueeze(1)
            target_q_values = rewards + self.gamma * (1 - dones) * best_next_q_values

        loss = functional.mse_loss(taken_action_q_values, target_q_values.detach())
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # Periodic target network update (every `target_update_freq` steps), hard update
        self.learn_counter += 1
        if self.learn_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    def save_models(self, filename: str, filepath: str) -> None:
        """Save the model for this algorithm.

        Args:
            filename: Base name for the saved model files
            filepath: Directory path where models will be saved
        """
        dir_exists = os.path.exists(filepath)
        if not dir_exists:
            os.makedirs(filepath)
        torch.save(self.net.state_dict(), f"{filepath}/{filename}_net.pth")

    def load_models(self, filename: str, filepath: str) -> None:
        """Load the model previously saved for this algorithm.

        Args:
            filename: Filename of the model, without extension
            filepath: Path to the saved models, usually located in user's home directory
        """
        self.net.load_state_dict(
            torch.load(
                f"{filepath}/{filename}_net.pth",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.net.eval()
