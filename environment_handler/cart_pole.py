import numpy as np
from . import EnvironmentHandler, Evaluation


class CartPoleHandler(EnvironmentHandler):
    env_name = "CartPole-v1"
    state_space_normalization = np.array([4.8, 1.0, 0.418, 1.0])
    previous_benchmark = 20

    def benchmark_agent(self, agent):
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
        

    def evaluate_benchmark(self, prev_benchmark, benchmark_result):
        self.previous_benchmark = self.previous_benchmark if prev_benchmark is None else prev_benchmark

        if self.previous_benchmark <= 20.0 and benchmark_result <= 20.0:
            return Evaluation.RESET_MODEL
        elif 200 < benchmark_result:
            return Evaluation.AGENT_TRAINED
        elif benchmark_result <= self.previous_benchmark:
            return Evaluation.NEEDS_MORE_DATA
        else:
            return Evaluation.UPDATE_WEIGHTS

    def _preprocess_action_outcome(self, state, reward, terminated, truncated, info):
        normalised_state = self._preprocess_state(state)
        reward = 0.0 if terminated or truncated else 1.0
        return normalised_state, reward, terminated, truncated

    def _preprocess_state(self, state):
        normalised_state = state / self.state_space_normalization
        return normalised_state[2:] # only keep orientation & angular velocity of pole

    def _get_sample_weight(self, action, next_state, reward, finished):
        # This makes sure the bad moves are weighted stronger. Empirically it doesn't work.
        #return 1.0 if finished else 1.0 / self.previous_benchmark
        return 1.0
