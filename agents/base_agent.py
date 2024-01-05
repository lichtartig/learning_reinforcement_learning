from abc import ABC
from dataclasses import dataclass
from enum import Enum
import numpy as np
import os


class BatchGeneratorType(Enum):
    STATE_AND_ACTION_AS_INPUTS = 1
    ONLY_STATE_AS_INPUT = 2


@dataclass
class ModelHyperParams():
    no_hidden_layers: int
    units_per_hidden_layer: int
    initial_epsilon: float
    epsilon_decay_constant: float
    gamma: float = 0.0
    cycles_per_target_update: int = 1
    target_update_fraction: float = 1.0
    kernel_regularizer: str = None
    kernel_initializer: str = "glorot_uniform"
    optimizer: str = "adam"
    

class BaseAgent(ABC):
    epsilon = 0.0
    epsilon_decay_constant = 0.9
    weights_save_path = "weights"
    name = "base"
    generator_type = None

    def __init__(self, action_space, state_space, hyper_params: ModelHyperParams = None, verbose=1):
        self.action_space = action_space
        self.state_space = state_space
        self.hyper_params = hyper_params
        # TODO: Fix in terms of self.env.action_space & self.env.observation_space once I adapt to other environments.
        # Also: This is confusing with the action space consisting of 2 actions here [0, 1].
        # I.e. I should reduce this to fewer variables.
        self.action_dim = 1 
        self.state_dim = 2
        self.all_actions = np.array([0, 1])
        self.epsilon = hyper_params.initial_epsilon
        self.model = self._get_model(hyper_params)
        self.verbose=verbose

    def build_targets(self, actions, rewards, states, next_states):
        pass

    def decay_epsilon(self):
        new_epsilon = np.round(self.hyper_params.epsilon_decay_constant*self.epsilon, 5)
        self.epsilon = 0.0 if self.epsilon == new_epsilon else new_epsilon
        if self.verbose == 1:
            print("eps: ", self.epsilon)

    def get_action(self, state, is_training=False):
        raise NotImplementedError

    def load_weights(self):
        self.model.load_weights(os.path.join(self.weights_save_path, self.name))

    def reset_model(self):
        self.model = self._get_model(self.hyper_params)
    
    def save_weights(self):
        self.model.save_weights(os.path.join(self.weights_save_path, self.name))

    def train(self, batch_generator):
        self.model.fit(batch_generator, verbose=self.verbose)

    def _get_model(self, hyper_params: ModelHyperParams):
        raise NotImplementedError

    def _get_categorical_action_encoding(self, actions):
        # TODO see above.
        return np.array([actions, (1.0-actions)])

    def _get_random_action(self):
        return self.action_space.sample()

    def _should_make_random_action(self):
        return np.random.random() < self.epsilon