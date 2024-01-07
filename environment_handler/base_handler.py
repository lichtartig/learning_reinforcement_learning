import gymnasium as gym
import numpy.typing as npt
from abc import ABC
from enum import Enum


class Evaluation(Enum):
    RESET_MODEL = 1
    AGENT_TRAINED = 2
    NEEDS_MORE_DATA = 3
    UPDATE_WEIGHTS = 4


class EnvironmentHandler(ABC):
    def __init__(self, no_of_benchmark_episodes: int = 5, show_graphics: bool = False):
        self.env = gym.make(self.env_name, render_mode="human") if show_graphics else gym.make(self.env_name)
        self.no_of_benchmark_episodes = no_of_benchmark_episodes
        self.environment_seed = 42 # TODO This should be fixed at some point.
        self.is_model_comparison = False
        self.benchmark_results = []

    def benchmark_agent(self, agent) -> float:
        raise NotImplementedError

    def close_env(self):
        self.env.close()

    def enable_model_comparison_mode(self):
        self.is_model_comparison = True

    def evaluate_benchmark(self, prev_benchmark: float, benchmark_result: float) -> Evaluation:
        raise NotImplementedError

    def get_action_space(self):
        return self.env.action_space

    def get_action_dim(self):
        # TODO write generic code to derive this from action space rather than doing this in Implementation, once needed
        raise NotImplementedError

    def get_all_actions(self):
        # TODO write generic code to derive this from action space rather than doing this in Implementation, once needed
        raise NotImplementedError

    def get_benchmark_results(self):
        return self.benchmark_results

    def get_categorical_action_encoding(self, actions: npt.ArrayLike) -> npt.ArrayLike:
        raise

    def get_initial_state(self):
        state, _ = self.env.reset(seed=self.environment_seed)
        self.environment_seed += 1
        return self._preprocess_state(state)

    def get_state_space(self):
        return self.env.observation_space 

    def get_state_dim(self):
        # TODO write generic code to derive this from action space rather than doing this in Implementation, once needed
        raise NotImplementedError

    def perform_action(self, action: npt.ArrayLike) -> tuple[npt.ArrayLike, float, bool, float]:
        next_state, reward, terminated, truncated = self._preprocess_action_outcome(*self.env.step(action))
        finished = terminated or truncated
        sample_weight = self._get_sample_weight(action, next_state, reward, finished)
        return next_state, reward, finished, sample_weight

    def _get_sample_weight(self, action: npt.ArrayLike, next_state: npt.ArrayLike, reward: float, finished: bool) -> float:
        return 1.0
    
    def _preprocess_action_outcome(self, state: npt.ArrayLike, reward: float, terminated: bool, truncated: bool,
                                   info: npt.ArrayLike):
        return state, reward, terminated, truncated

    def _preprocess_state(self, state: npt.ArrayLike):
        return state
