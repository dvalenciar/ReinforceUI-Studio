import os
import numpy as np
import torch
import torch.nn.functional as F
from RL_algorithms.DQN.networks import Network


class DQN:
    def __init__(self, observation_size, action_num, hyperparameters):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Network(observation_size, action_num).to(self.device)

        self.gamma = float(hyperparameters.get("gamma"))
        self.epsilon = float(hyperparameters.get("epsilon"))
        self.epsilon_decay = float(hyperparameters.get("epsilon_decay"))
        self.min_epsilon = float(hyperparameters.get("min_epsilon"))
        self.lr = float(hyperparameters.get("lr"))


        self.action_num = action_num
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def select_action_from_policy(
        self, state: np.ndarray, evaluation: bool = False
    ) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            q_values = self.net(state_tensor)
            action = np.argmax(q_values.cpu().data.numpy())
        self.net.train()
        return action

    def train_policy(self, memory, batch_size):
        # todo verify the shaope of the tensors and the operations
        experiences = memory.sample_experience(batch_size)
        states, actions, rewards, next_states, dones = experiences

        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.LongTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.FloatTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size
        rewards = rewards.reshape(batch_size, 1)
        dones = dones.reshape(batch_size, 1)
        actions = actions.reshape(batch_size, 1)

        # Compute Q-values for the current states
        q_values = self.net(states)

        # Get Q-values for the actions actually taken
        taken_action_q_values = q_values.gather(1, actions).squeeze(1)

        # Compute Q-values for next states and take max over actions
        next_q_values = self.net(next_states)
        best_next_q_values = torch.max(next_q_values, 1)[0].unsqueeze(1)

        target_q_values = rewards + self.gamma * (1 - dones) * best_next_q_values

        loss = F.mse_loss(taken_action_q_values, target_q_values.detach())

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


