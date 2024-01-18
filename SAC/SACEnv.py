import gym
from gym import spaces
import numpy as np
import pandas as pd

class FeatureClassificationEnv(gym.Env):
    def __init__(self, dataset):
        super(FeatureClassificationEnv, self).__init__()
        self.dataset = pd.read_csv(dataset)
        self.action_space = spaces.Discrete(3)  # 0: 左移, 1: 右移, 2: 停止
        self.observation_space = spaces.Box(low=np.array([0, -np.inf]), high=np.array([1, np.inf]), dtype=np.float32)
        self.position = 0  # 初始化智能体位置
        self.max_position = 4  # 假设线段上有4个位置（0, 1, 2, 3）

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state, label = self.state
        done = False
        reward = 0

        # 根据动作更新位置
        if action == 0 and self.position > 0:
            self.position -= 1
        elif action == 1 and self.position < self.max_position:
            self.position += 1
        elif action == 2 or self.current_feature == len(self.features) - 1:  # 停止或最后一个特征
            done = True
            reward = 1 if self.position == label else -1  # 根据分类结果给出奖励

        self.current_feature += 1
        next_state = [self.current_feature / len(self.features), self.features[self.current_feature]]

        self.state = (next_state, label)
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.position = 0  # 重置智能体位置
        self.current_feature = 0
        sample = self.dataset.sample().iloc[0]
        self.features = sample[:-1].tolist()  # 假设最后一列是标签
        label = sample[-1]  # 标签
        state = [self.current_feature / len(self.features), self.features[self.current_feature]]
        self.state = (state, label)
        return np.array(self.state)

    def render(self, mode='human', close=False):
        print(f"Current Position: {self.position}, Current Feature: {self.state[0][1]}")