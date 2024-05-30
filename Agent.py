from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F

# 指定GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, state_dim, action_dim, batch_size):
        self.capacity = capacity
        self.obs_cap = np.empty((self.capacity, obs_dim))
        self.next_obs_cap = np.empty((self.capacity, obs_dim))
        self.state_cap = np.empty((self.capacity, state_dim))
        self.next_state_cap = np.empty((self.capacity, state_dim))
        self.action_cap = np.empty((self.capacity, action_dim))
        self.reward_cap = np.empty((self.capacity, 1))
        self.done_cap = np.empty((self.capacity, 1), dtype=bool)

        self.batch_size = batch_size
        self.current_idx = 0
        self.size = 0  # 记录当前缓冲区中的样本数量

    def add_memo(self, obs, next_obs, state, next_state, action, reward, done):
        self.obs_cap[self.current_idx] = obs
        self.next_obs_cap[self.current_idx] = next_obs
        self.state_cap[self.current_idx] = state
        self.next_state_cap[self.current_idx] = next_state
        self.action_cap[self.current_idx] = action
        self.reward_cap[self.current_idx] = reward
        self.done_cap[self.current_idx] = done

        self.current_idx = (self.current_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if isinstance(batch_size, (range, list, np.ndarray)):  # 如果batch_size是range, list 或 np.ndarray 类型，获取它的长度
            batch_size = len(batch_size)

        if self.size < batch_size:
            batch_size = self.size

        batch_indices = np.random.choice(self.size, batch_size, replace=False)

        obs = self.obs_cap[batch_indices]
        next_obs = self.next_obs_cap[batch_indices]
        state_cap = self.state_cap[batch_indices]
        next_state_cap = self.next_state_cap[batch_indices]
        action_cap = self.action_cap[batch_indices]
        reward_cap = self.reward_cap[batch_indices]
        done_cap = self.done_cap[batch_indices]
        # 打印采样的动作维度
        print(f"Sampled actions shape: {action_cap.shape}")
        return obs, next_obs, action_cap, state_cap, next_state_cap, reward_cap, done_cap


class CriticNetwork(nn.Module):
    def __init__(self, lr_actor, input_dims, fc1_dims, fc2_dims, num_agent, action_dim):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.num_agent = num_agent
        self.action_dim = action_dim

        total_input_dims = input_dims + num_agent * action_dim

        self.fc1 = nn.Linear(total_input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_actor)
        self.to(device)

    def forward(self, state, action):
        print(f"CriticNetwork - State shape: {state.shape}, Action shape: {action.shape}")

        assert state.shape[0] == action.shape[0], "Batch sizes must match"
        assert state.shape[1] == self.input_dims, "State dimensions must match input dimensions"
        assert action.shape[1] == self.num_agent * self.action_dim, "Action dimensions must match"

        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q

    def save_checkpoint(self, filename, is_best=False):
        torch.save(self.state_dict(), filename)
        if is_best:
            torch.save(self.state_dict(), 'best_' + filename)
            torch.save(self.optimizer.state_dict(), 'optimizer_' + filename)
            print('Best checkpoint saved')

    def load_checkpoint(self, filename):
        self.load_state_dict(torch.load(filename))
        print('Checkpoint loaded')
        self.optimizer.load_state_dict(torch.load('optimizer_' + filename))
        print('Checkpoint loaded')


class ActorNetwork(nn.Module):
    def __init__(self, lr_actor, input_size, fc1_dims, fc2_dims, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, action_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_actor)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = torch.softmax(self.pi(x), dim=1)
        return mu

    def save_checkpoint(self, filename, is_best=False):
        torch.save(self.state_dict(), filename)
        if is_best:
            torch.save(self.state_dict(), 'best_' + filename)
            torch.save(self.optimizer.state_dict(), 'optimizer_' + filename)
            print('Best checkpoint saved')

    def load_checkpoint(self, filename):
        self.load_state_dict(torch.load(filename))
        print('Checkpoint loaded')
        self.optimizer.load_state_dict(torch.load('optimizer_' + filename))
        print('Checkpoint loaded')


class Agent(object):
    def __init__(self, memo_size, obs_dim, state_dim, n_agent, action_dim,
                 alpha, beta, fc1_dims, fc2_dims, gamma, tau, batch_size):
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.actor = ActorNetwork(lr_actor=alpha, input_size=obs_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                                  action_dim=action_dim)
        self.critic = CriticNetwork(lr_actor=beta, input_dims=state_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                                    num_agent=n_agent, action_dim=action_dim)
        self.target_actor = ActorNetwork(lr_actor=alpha, input_size=obs_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                                         action_dim=action_dim)
        self.target_critic = CriticNetwork(lr_actor=beta, input_dims=state_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                                           num_agent=n_agent, action_dim=action_dim)

        self.replay_buffer = ReplayBuffer(capacity=memo_size, obs_dim=obs_dim, state_dim=state_dim,
                                          action_dim=action_dim, batch_size=batch_size)

    def get_action(self, obs):
        single_obs = torch.tensor(data=obs, dtype=torch.float).unsqueeze(0).to(device)
        single_action = self.actor.forward(single_obs)
        noise = torch.randn(self.action_dim).to(device) * 0.2
        single_action = torch.clamp(input=single_action + noise, min=0.0, max=1.0)
        print(f"Agent - Action shape: {single_action.shape}")
        return single_action.detach().cpu().numpy()[0]

    def save_model(self, filename, is_best=False):
        self.actor.save_checkpoint(filename, is_best)
        self.target_actor.save_checkpoint(filename, is_best)
        self.critic.save_checkpoint(filename, is_best)
        self.target_critic.save_checkpoint(filename, is_best)

    def load_model(self, filename):
        self.actor.load_checkpoint(filename)
        self.target_actor.load_checkpoint(filename)
        self.critic.load_checkpoint(filename)
        self.target_critic.load_checkpoint(filename)
