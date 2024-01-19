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

        self.num_features = 79  # 79个特征
        self.max_position = self.num_features - 1  # 设置线段的长度，剪掉Label
        self.position = 0  # 初始化Agent位置

        # 分布标签位置
        self.label_positions = {
            self.num_features * i // 4: label
            for i, label in enumerate(["Benign", "Bot", "DDoS attack-HOIC", "DDoS attack-LOIC-UDP"])
        }

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
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
            # 找到最接近Agent位置的标签
            closest_label_position = min(self.label_positions.keys(), key=lambda k: abs(k - self.position))
            predicted_label = self.label_positions[closest_label_position]
            reward = 1 if predicted_label == label else -1  # 根据分类结果给出奖励

        self.current_feature += 1
        if not done:
            next_state = [self.current_feature / len(self.features), self.features[self.current_feature]]
        else:
            next_state = [1, 0]  # 结束后的状态不再重要

        self.state = (next_state, label)
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.position = 0  # 重置Agent位置
        self.current_feature = 0
        sample = self.dataset.sample().iloc[0]
        self.features = sample[:-1].tolist()
        label = sample[-1]  # 标签
        state = [self.current_feature / len(self.features), self.features[self.current_feature]]
        self.state = (state, label)
        return np.array(self.state)

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(f"Current Position: {self.position}, Current Feature: {self.state[0][1]}")
        else:
            super(FeatureClassificationEnv, self).render(mode=mode)