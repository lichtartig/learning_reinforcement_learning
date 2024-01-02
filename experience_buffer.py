import numpy as np
from collections import deque
from enum import Enum


class BatchGeneratorType(Enum):
    STATE_AND_ACTION_AS_INPUTS = 1
    ONLY_STATE_AS_INPUT = 2


class ExperienceBuffer():
    def __init__(self, max_buffer_size):
        self.max_buffer_size = max_buffer_size
        
        self.states = deque([])
        self.actions = deque([])
        self.next_states = deque([])
        self.rewards = deque([])
        self.sample_weights = deque([])

        self.state_arr = np.array([])
        self.action_arr = np.array([])
        self.next_state_arr = np.array([])
        self.reward_arr = np.array([])
        self.sample_weight_arr = np.array([])
        self.target_arr = np.array([])

    def save_experience(self, state, action, reward, sample_weight, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.sample_weights.append(sample_weight)

        if self.max_buffer_size < len(self.states):
            self.states.popleft()
            self.actions.popleft()
            self.next_states.popleft()
            self.rewards.popleft()
            self.sample_weights.popleft()

    def prepare_experience_for_training(self, target_builder_fct):
        p = np.random.permutation(len(self.states))
        self.state_arr = np.array(self.states)[p]
        self.action_arr = np.array(self.actions)[p]
        self.next_state_arr = np.array(self.next_states)[p]
        self.reward_arr = np.array(self.rewards)[p]
        self.sample_weight_arr = np.array(self.sample_weights)[p]
        self.target_arr = target_builder_fct(self.action_arr, self.reward_arr, self.state_arr, self.next_state_arr)

    def batch_generator(self, batch_size, batch_generator_type):
        start = 0
        while start + batch_size <= len(self.states):
            end = start + batch_size
            s = slice(start, end)
            
            only_states = batch_generator_type == BatchGeneratorType.ONLY_STATE_AS_INPUT
            inputs = self.state_arr[s] if only_states else [self.state_arr[s], self.action_arr[s]]
            weights = self.sample_weight_arr[s]
            targets = self.target_arr[s]
            yield inputs, targets, weights
            start = end

    def flush(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
