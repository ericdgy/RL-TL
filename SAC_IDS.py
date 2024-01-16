import gym
from gym import spaces
class SAC_IDS_ENV(gym.Env):
    def __init__(self,dataset):
        super().__init__()
        self.dataset = dataset
