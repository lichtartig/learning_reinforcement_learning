import gymnasium as gym
from abc import ABC
from enum import Enum


class Evaluation(Enum):
    RESET_MODEL = 1
    AGENT_TRAINED = 2
    NEEDS_MORE_DATA = 3
    UPDATE_WEIGHTS = 4


class EnvironmentHandler(ABC):
    def __init__(self, no_of_benchmark_episodes: int = 5):
        self.env = gym.make(self.env_name)
        self.no_of_benchmark_episodes = no_of_benchmark_episodes
        self.environment_seed = 42 # TODO This should be fixed.

    def get_action_space(self):
        return self.env.action_space

    def get_state_space(self):
        return self.env.observation_space
    
    def get_initial_state(self):
        state, _ = self.env.reset(seed=self.environment_seed)
        self.environment_seed += 1
        return self._preprocess_state(state)

    def perform_action(self, action):
        next_state, reward, terminated, truncated = self._preprocess_action_outcome(*self.env.step(action))
        finished = terminated or truncated
        sample_weight = self._get_sample_weight(action, next_state, reward, finished)
        return next_state, reward, finished, sample_weight

    def benchmark_agent(self, agent):
        raise NotImplementedError

    def evaluate_benchmark(self, prev_benchmark, benchmark_result):
        raise NotImplementedError

    def close_env(self):
        self.env.close()

    def _preprocess_action_outcome(self, state, reward, terminated, truncated, info):
        return state, reward, terminated, truncated

    def _preprocess_state(self, state):
        return state

    def _get_sample_weight(self, action, next_state, reward, finished):
        return 1.0
