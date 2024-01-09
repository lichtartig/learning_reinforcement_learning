import numpy as np
import numpy.typing as npt
import os
from . import BatchGeneratorType, ModelHyperParams, QLearner
from environment_handler import EnvironmentHandler


class QLearnerWithTargetNetwork(QLearner):
    """ This inherits from the QLearner class and thereby also from the BaseAgent. It adds an additional target-network to the
    agent that serves to maintain training-targets constant during training and thereby address the overestimation problem. """
    
    name = "q_learner_with_target_network"
    generator_type = BatchGeneratorType.STATE_AND_ACTION_AS_INPUTS
    
    def __init__(self, env_handler: EnvironmentHandler, hyper_params: ModelHyperParams = None, verbose: int = 1):
        super().__init__(env_handler, hyper_params, verbose)
        self.target_model = self._get_model(hyper_params)
        self.target_model.set_weights(self.model.get_weights())
        self.training_cycles = 0

    def build_targets(
        self,
        actions: npt.ArrayLike, rewards: npt.ArrayLike, states: npt.ArrayLike, next_states: npt.ArrayLike
    ) -> npt.ArrayLike:
        dim = next_states.shape[0]
        predict = lambda a: self.target_model.predict([next_states, np.repeat(a[None,...], dim)], verbose=0)
        max_qs = np.max([predict(a) for a in self.env_handler.get_all_actions()], axis=0)[:, 0]
        return (rewards + self.hyper_params.gamma * max_qs) / (1 + self.hyper_params.gamma)

    def load_weights(self, file_name: str = None):
        if file_name is None:
            file_name = self.name
        super().load_weights(file_name)
        self.target_model.load_weights(os.path.join(self.weights_save_path, file_name + "_target"))

    def reset_model(self):
        super().reset_model()
        self.target_model.set_weights(self.model.get_weights())

    def save_weights(self, weights_file_name: str = None):
        super().save_weights(weights_file_name)
        self.training_cycles += 1
        if self.training_cycles % self.hyper_params.cycles_per_target_update == 0:
            print("Updating target weights...")
            self._update_target_model_weights()

        if weights_file_name is None:
            weights_file_name = self.name
        self.target_model.save_weights(os.path.join(self.weights_save_path, weights_file_name + "_target"))

    def _update_target_model_weights(self):
        train = self.model.get_weights()
        target = self.target_model.get_weights()
        tau = self.hyper_params.target_update_fraction
        new_weights = [tau * train[i] + (1 - tau) * target[i] for i in range(len(train))]
        self.target_model.set_weights(new_weights)
