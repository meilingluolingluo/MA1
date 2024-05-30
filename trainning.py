from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
from pettingzoo.mpe import simple_adversary_v3

from Agent import Agent

# 设置device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# 辅助函数设计，将多个智能体的观察合并为一个全局状态
def multi_obs_to_state(multi_obs):
    state = np.array([])
    for agent_obs in multi_obs.values():
        state = np.concatenate((state, agent_obs))
    return state


# 设置超参数
# 回合数
NUM_EPISODES = 1000
# 每回合步数
NUM_STEP = 25
# 经验池大小
MEMORY_BUFFER_SIZE = 100
# 批量大小
BATCH_SIZE = 32
# 更新间隔
TARGET_UPDATE_INTERVAL = 10
# 学习率
LR_ACTOR = 1e-3
LR_CRITIC = 1e-2
# 隐藏层维度
HIDDEN_DIM = 64
# 折扣因子
GAMMA = 0.99
# 软更新参数
TAU = 1e-3

# 1. 初始化环境和智能体
env = simple_adversary_v3.parallel_env(N=2, max_cycles=25, continuous_actions=True, render_mode=None)  # 禁用渲染
multi_obs, infos = env.reset()
multi_obs = dict(multi_obs)  # 确保 multi_obs 是字典
single_obs = list(multi_obs.values())[0]

NUM_AGENT = env.num_agents
agent_name_list = env.agents

# 确保在初始化ActorNetwork和CriticNetwork时，输入维度和环境中的状态维度一致
obs_dim = []
for agent_obs in multi_obs.values():
    obs_dim.append(agent_obs.shape[0])
state_dim = sum(obs_dim)

agent_dim = []
for agent_name in agent_name_list:
    agent_dim.append(env.action_space(agent_name).shape[0])

agents = []
for i in range(NUM_AGENT):
    print(f"Initializing agent {i}...")
    agent = Agent(memo_size=MEMORY_BUFFER_SIZE, obs_dim=obs_dim[i], state_dim=state_dim, n_agent=NUM_AGENT,
                  action_dim=agent_dim[i], alpha=LR_ACTOR, beta=LR_CRITIC, fc1_dims=HIDDEN_DIM, fc2_dims=HIDDEN_DIM,
                  gamma=GAMMA, tau=TAU, batch_size=BATCH_SIZE)
    agents.append(agent)
    print(f"Agent {i} action dimension: {agent_dim[i]}")

# 主训练循环
for episode_i in range(NUM_EPISODES):
    multi_obs, infos = env.reset()
    multi_obs = dict(multi_obs)  # 确保 multi_obs 是字典
    episode_reward = 0
    multi_done = {agent_name: False for agent_name in agent_name_list}
    for step_i in range(NUM_STEP):
        total_step = episode_i * NUM_STEP + step_i + 1

        multi_action = {}
        for agent_i, agent_name in enumerate(agent_name_list):
            agent = agents[agent_i]
            single_obs = multi_obs.get(agent_name, None)
            if single_obs is None:
                continue  # 如果当前智能体的观察不存在，跳过该智能体
            single_action = agent.get_action(single_obs)
            multi_action[agent_name] = single_action
            print(f"Agent {agent_i} action shape: {single_action.shape}")

        multi_next_obs, multi_reward, multi_done, multi_truncation, infos = env.step(multi_action)
        multi_next_obs = dict(multi_next_obs)  # 确保 multi_next_obs 是字典
        state = multi_obs_to_state(multi_obs)
        next_state = multi_obs_to_state(multi_next_obs)

        if step_i >= NUM_STEP - 1:
            multi_done = {agent_name: False for agent_name in agent_name_list}

        for agent_i, agent_name in enumerate(agent_name_list):
            agent = agents[agent_i]
            single_obs = multi_obs.get(agent_name, None)
            single_next_obs = multi_next_obs.get(agent_name, None)
            if single_obs is None or single_next_obs is None:
                continue  # 如果当前智能体的观察或下一观察不存在，跳过该智能体
            single_action = multi_action[agent_name]
            single_reward = multi_reward[agent_name]
            single_done = multi_done[agent_name]
            agent.replay_buffer.add_memo(single_obs, single_next_obs, state, next_state, single_action, single_reward,
                                         single_done)

        if total_step > BATCH_SIZE:
            multi_batch_obs = []
            multi_batch_next_obs = []
            multi_batch_states = []
            multi_batch_next_states = []
            multi_batch_actions = []
            multi_batch_next_actions = []
            multi_batch_online_actions = []
            multi_batch_rewards = []
            multi_batch_dones = []

            current_memo_size = min(MEMORY_BUFFER_SIZE, total_step)
            if current_memo_size < BATCH_SIZE:
                batch_idx = list(range(0, current_memo_size))
            else:
                batch_idx = np.random.choice(current_memo_size, BATCH_SIZE, replace=False)

            for agent_i in range(NUM_AGENT):
                agent = agents[agent_i]
                batch_obs, batch_next_obs, batch_actions, batch_states, batch_next_state, batch_reward, batch_done = agent.replay_buffer.sample(
                    batch_idx)

                batch_obs_tensor = torch.tensor(batch_obs, dtype=torch.float).to(device)
                batch_next_obs_tensor = torch.tensor(batch_next_obs, dtype=torch.float).to(device)
                batch_states_tensor = torch.tensor(batch_states, dtype=torch.float).to(device)
                batch_next_state_tensor = torch.tensor(batch_next_state, dtype=torch.float).to(device)
                batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.float).to(device)
                batch_reward_tensor = torch.tensor(batch_reward, dtype=torch.float).to(device)
                batch_done_tensor = torch.tensor(batch_done, dtype=torch.float).to(device)

                multi_batch_obs.append(batch_obs_tensor)
                multi_batch_next_obs.append(batch_next_obs_tensor)
                multi_batch_states.append(batch_states_tensor)
                multi_batch_next_states.append(batch_next_state_tensor)
                multi_batch_actions.append(batch_actions_tensor)
                multi_batch_rewards.append(batch_reward_tensor)
                multi_batch_dones.append(batch_done_tensor)

                single_batch_next_actions = agent.target_actor.forward(batch_next_obs_tensor)
                multi_batch_next_actions.append(single_batch_next_actions)
                single_batch_online_actions = agent.actor.forward(batch_obs_tensor)
                multi_batch_online_actions.append(single_batch_online_actions)

            # 确保动作张量维度一致
            multi_batch_actions_tensor = torch.cat(multi_batch_actions, dim=1).to(device)
            multi_batch_next_actions_tensor = torch.cat(multi_batch_next_actions, dim=1).to(device)
            multi_batch_online_actions_tensor = torch.cat(multi_batch_online_actions, dim=1).to(device)

            # 打印拼接后的动作张量维度
            print(f"Multi batch actions tensor shape: {multi_batch_actions_tensor.shape}")
            print(f"Multi batch next actions tensor shape: {multi_batch_next_actions_tensor.shape}")
            print(f"Multi batch online actions tensor shape: {multi_batch_online_actions_tensor.shape}")

            if (total_step + 1) % TARGET_UPDATE_INTERVAL == 0:
                for agent_i in range(NUM_AGENT):
                    agent = agents[agent_i]

                    batch_obs_tensor = multi_batch_obs[agent_i]
                    batch_states_tensor = multi_batch_states[agent_i]
                    batch_next_state_tensor = multi_batch_next_states[agent_i]
                    batch_reward_tensor = multi_batch_rewards[agent_i]
                    batch_done_tensor = multi_batch_dones[agent_i]
                    batch_actions_tensor = multi_batch_actions[agent_i]

                    # 打印批量状态和动作张量的维度
                    print(
                        f"CriticNetwork - State shape: {batch_states_tensor.shape}, Action shape: {multi_batch_actions_tensor.shape}")

                    critic_target_q = agent.target_critic.forward(batch_next_state_tensor,
                                                                  multi_batch_next_actions_tensor.detach())
                    y = batch_reward_tensor + (
                                agent.gamma * critic_target_q * (1 - batch_done_tensor)).flatten().unsqueeze(1)
                    critic_q = agent.critic.forward(batch_states_tensor, multi_batch_actions_tensor)
                    print(f"Target y shape: {y.shape}, Critic Q shape: {critic_q.shape}")

                    critic_loss = F.mse_loss(y, critic_q)
                    agent.critic.optimizer.zero_grad()
                    critic_loss.backward()
                    agent.critic.optimizer.step()

                    # 使用单个智能体的观察维度来计算 actor_loss
                    actor_loss = -torch.mean(agent.actor.forward(batch_obs_tensor))
                    agent.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    agent.actor.optimizer.step()

                    for target_param, param in zip(agent.target_actor.parameters(), agent.actor.parameters()):
                        target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)

                    for target_param, param in zip(agent.target_critic.parameters(), agent.critic.parameters()):
                        target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)

        multi_obs = multi_next_obs
        episode_reward += sum([single_reward for single_reward in multi_reward.values() if single_reward is not None])
    print(f"Episode {episode_i} reward: {episode_reward}")

# 渲染环境
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    multi_obs, rewards, terminations, truncations, infos = env.step(actions)
env.close()
