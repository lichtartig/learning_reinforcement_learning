import numpy as np
import numpy.typing as npt
import os
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from environment_handler import EnvironmentHandler
from tensorflow.keras.models import Model
from typing import Callable


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

    def __init__(self, env_handler: EnvironmentHandler, hyper_params: ModelHyperParams = None, verbose: int = 1):
        self.env_handler = env_handler
        self.hyper_params = hyper_params
        self.epsilon = hyper_params.initial_epsilon
        self.model = self._get_model(hyper_params)
        self.verbose=verbose

    def build_targets(
        self,
        actions: npt.ArrayLike, rewards: npt.ArrayLike, states: npt.ArrayLike, next_states: npt.ArrayLike
    ) -> npt.ArrayLike:
        pass

    def decay_epsilon(self):
        new_epsilon = np.round(self.hyper_params.epsilon_decay_constant*self.epsilon, 5)
        self.epsilon = 0.0 if self.epsilon == new_epsilon else new_epsilon
        if self.verbose == 1:
            print("eps: ", self.epsilon)

    def get_action(self, state: npt.ArrayLike, is_training: bool = False) -> npt.ArrayLike:
        raise NotImplementedError

    def load_weights(self):
        self.model.load_weights(os.path.join(self.weights_save_path, self.name))

    def reset_model(self):
        self.model = self._get_model(self.hyper_params)
    
    def save_weights(self):
        self.model.save_weights(os.path.join(self.weights_save_path, self.name))

    def train(self, batch_generator: Callable[[int, BatchGeneratorType], npt.ArrayLike]):
        self.model.fit(batch_generator, verbose=self.verbose)

    def _get_model(self, hyper_params: ModelHyperParams) -> Model:
        raise NotImplementedError

    def _get_random_action(self) -> npt.ArrayLike:
        action_space = self.env_handler.get_action_space()
        return action_space.sample()

    def _should_make_random_action(self) -> bool:
        return np.random.random() < self.epsilon