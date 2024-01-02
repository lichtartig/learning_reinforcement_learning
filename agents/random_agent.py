from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def get_action(self, state, is_training=False):
        return self._get_random_action()