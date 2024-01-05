import numpy as np
import numpy.typing as npt
from . import EnvironmentHandler, Evaluation


class CartPoleHandler(EnvironmentHandler):
    env_name = "CartPole-v1"
    state_space_normalization = np.array([4.8, 1.0, 0.418, 1.0])
    previous_benchmark = 20

    def benchmark_agent(self, agent) -> float:
        lengths = []
        
        for e in range(self.no_of_benchmark_episodes):
            state = self.get_initial_state()
            finished = False
            counter = 0
    
            while not finished and counter <= 200:
                action = agent.get_action(state)
                state, reward, finished, _ = self.perform_action(action)
                counter += 1
    
            lengths.append(counter)
            
        median_episode_length = np.median(lengths)
        return median_episode_length
        

    def evaluate_benchmark(self, prev_benchmark: float, benchmark_result: float) -> Evaluation:
        self.previous_benchmark = self.previous_benchmark if prev_benchmark is None else prev_benchmark

        if self.previous_benchmark <= 20.0 and benchmark_result <= 20.0:
            return Evaluation.RESET_MODEL
        elif 200 < benchmark_result:
            return Evaluation.AGENT_TRAINED
        elif benchmark_result <= self.previous_benchmark:
            return Evaluation.NEEDS_MORE_DATA
        else:
            return Evaluation.UPDATE_WEIGHTS

    def get_action_dim(self):
        return 1

    def get_all_actions(self):
        all_actions = np.array([0, 1])
        return all_actions

    def get_state_dim(self):
        return 2

    def get_categorical_action_encoding(self, actions: npt.ArrayLike) -> npt.ArrayLike:
        return np.array([actions, (1.0-actions)])

    def _preprocess_action_outcome(self, state: npt.ArrayLike, reward: float, terminated: bool, truncated: bool,
                                   info: npt.ArrayLike):
        normalised_state = self._preprocess_state(state)
        reward = 0.0 if terminated or truncated else 1.0
        return normalised_state, reward, terminated, truncated

    def _preprocess_state(self, state: npt.ArrayLike):
        normalised_state = state / self.state_space_normalization
        orientation_and_angular_velocity_only = normalised_state[2:]
        return orientation_and_angular_velocity_only
