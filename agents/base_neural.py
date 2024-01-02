from abc import ABC
from dataclasses import dataclass
import numpy as np
import os


@dataclass
class HyperParams():
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


class BaseNeural(ABC):
    epsilon = 0.0
    epsilon_decay_constant = 0.9
    weights_save_path = "weights"
    name = "base"
    generator_type = None
    
    def train(self, batch_generator):
        self.model.fit(batch_generator)

    def decay_epsilon(self):
        new_epsilon = np.round(self.epsilon_decay_constant*self.epsilon, 5)
        self.epsilon = 0.0 if self.epsilon == new_epsilon else new_epsilon
        print("eps: ", self.epsilon)

    def build_targets(self, actions, rewards, states, next_states):
        pass

    def save_weights(self):
        self.model.save_weights(os.path.join(self.weights_save_path, self.name))

    def load_weights(self):
        self.model.load_weights(os.path.join(self.weights_save_path, self.name))

    def _should_make_random_action(self):
        return np.random.random() < self.epsilon