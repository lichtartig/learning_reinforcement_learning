import numpy as np
from abc import ABC


class BaseAgent(ABC):
    def __init__(self, env):
        self.env = env
        self.action_dim = 1 # TODO: Fix in terms of self.env.action_space once needed.
        self.state_dim = 2 # TODO: Fix in terms of self.env.observation_space once needed.

    def get_action(self, state, is_training=False):
        pass

    def _get_random_action(self):
        return self.env.action_space.sample()

    def _get_all_actions(self):
        return np.array([0, 1]) # TODO Fix in terms of self.env.action_space once needed.