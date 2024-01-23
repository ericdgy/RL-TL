import torch
from SACEnv import FeatureClassificationEnv  # 导入环境
from SAC_IDS import SAC  # 导入模型
from collections import deque
import random
import numpy as np
import os

# 设置超参数
state_dim = 79
hidden_dim = 256
action_dim = 3  # 动作空间大小
actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
target_entropy = -np.log(1.0 / action_dim) * 0.98  # 设置为动作空间的负对数乘以系数
tau = 0.005
gamma = 0.99
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化环境和模型
env = FeatureClassificationEnv('D:/dataset/merge_test2.csv')
model = SAC(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device)

# 设置训练相关参数
num_episodes = 1000
batch_size = 64
replay_buffer = deque(maxlen=100000)  # 经验回放缓冲区

# 训练循环
for episode in range(num_episodes):
    state = env.reset()
    state = np.array(state).flatten()
    episode_reward = 0
    done = False

    while not done:
        action = model.take_action(state)
        next_state, reward, done, _ = env.step(action)

        # 存储转换到回放缓冲区
        replay_buffer.append((state, action, reward, next_state, done))

        # 如果回放缓冲区足够大，从中采样并更新网络
        if len(replay_buffer) > batch_size:
            transitions = random.sample(replay_buffer, batch_size)
            batch = list(zip(*transitions))
            transition_dict = {
                'states': batch[0],
                'actions': batch[1],
                'rewards': batch[2],
                'next_states': batch[3],
                'dones': batch[4]
            }
            model.update(transition_dict)

        state = next_state
        episode_reward += reward

    print(f'Episode: {episode}, Reward: {episode_reward}')

# 保存模型参数
save_dir = './saved_models/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
torch.save(model.state_dict(), os.path.join(save_dir, 'sac_model.pth'))
