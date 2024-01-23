import gym
from gym import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class FeatureClassificationEnv(gym.Env):
    def __init__(self, dataset):
        super(FeatureClassificationEnv, self).__init__()
        self.dataset = pd.read_csv(dataset)

        # 初始化标签编码器并对标签进行编码
        self.label_encoder = LabelEncoder()
        self.dataset['Label'] = self.label_encoder.fit_transform(self.dataset['Label'])

        self.action_space = spaces.Discrete(3)  # 0: 左移, 1: 右移, 2: 停止
        self.observation_space = spaces.Box(low=np.array([0, -np.inf]), high=np.array([1, np.inf]), dtype=np.float32)

        self.num_features = 79  # 根据数据集的特征数量调整
        self.max_position = self.num_features - 1  # 设置线段的长度
        self.position = 0  # 初始化智能体位置

        # 分布标签位置
        self.label_positions = {
            self.num_features * i // 4: label
            for i, label in enumerate(self.label_encoder.classes_)
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
            # 找到最接近智能体位置的标签
            closest_label_position = min(self.label_positions.keys(), key=lambda k: abs(k - self.position))
            predicted_label = self.label_positions[closest_label_position]
            predicted_label_encoded = self.label_encoder.transform([predicted_label])[0]
            reward = 1 if predicted_label_encoded == label else -1  # 根据分类结果给出奖励

        self.current_feature += 1
        if not done:
            next_state = [self.current_feature / len(self.features), self.features[self.current_feature]]
        else:
            next_state = [1, 0]  # 结束后的状态不再重要

        self.state = (next_state, label)
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.position = 0  # 重置智能体位置
        self.current_feature = 0
        sample = self.dataset.sample().iloc[0]
        self.features = sample[:-1].tolist()
        label = sample['Label']  # 获取编码后的标签
        state = [self.current_feature / len(self.features), self.features[self.current_feature]]
        self.state = (state, label)
        return np.array(self.state)

    def render(self, mode='human', close=False):
        if mode == 'human':
            original_label = self.label_encoder.inverse_transform([self.state[1]])[0]
            print(f"Current Position: {self.position}, Current Feature: {self.state[0][1]}, Label: {original_label}")
        else:
            super(FeatureClassificationEnv, self).render(mode=mode)
