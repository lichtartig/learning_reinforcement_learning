import gymnasium as gym
import matplotlib.pyplot as plt
import numpy.typing as npt
import os
from abc import ABC
from enum import Enum
from matplotlib import animation


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

    def benchmark_agent(self, agent, replace_last_entry: bool = False) -> float:
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

    def render_animation(self, agent):
        plt.rcParams["animation.html"] = "jshtml"
        env = gym.make(self.env_name, render_mode="rgb_array")
        observation = env.reset()
        frames = []
        for t in range(1000):
            frames.append(env.render())
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        env.close()

        plt.figure(figsize=(frames[0].shape[1] / 200.0, frames[0].shape[0] / 200.0), dpi=100)
        patch = plt.imshow(frames[0])
        plt.axis('off')
    
        anim =  animation.FuncAnimation(plt.gcf(), lambda i: patch.set_data(frames[i]), frames = len(frames), interval=50)
        return anim

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
