import numpy as np
from . import BatchGeneratorType, ModelHyperParams, QLearner


class QLearnerWithTargetNetwork(QLearner):
    name = "q_learner_with_target_network"
    generator_type = BatchGeneratorType.STATE_AND_ACTION_AS_INPUTS
    
    def __init__(self, action_space, state_space, hyper_params: ModelHyperParams, verbose=1):
        super().__init__(action_space, state_space, hyper_params)
        self.target_model = self._get_model(hyper_params)
        self.target_model.set_weights(self.model.get_weights())
        self.training_cycles = 0

    def build_targets(self, actions, rewards, states, next_states):
        dim = next_states.shape[0]
        predict = lambda a: self.target_model.predict([next_states, np.repeat(a[None,...], dim)], verbose=0)
        max_qs = np.max([predict(a) for a in self.all_actions], axis=0)[:, 0]
        return (rewards + self.hyper_params.gamma * max_qs) / (1 + self.hyper_params.gamma)

    def reset_model(self):
        super().reset_model()
        self.target_model.set_weights(self.model.get_weights())

    def save_weights(self):
        super().save_weights()
        self.training_cycles += 1
        if self.training_cycles % self.hyper_params.cycles_per_target_update == 0:
            print("Updating target weights...")
            self._update_target_model_weights()

    def _update_target_model_weights(self):
        train = self.model.get_weights()
        target = self.target_model.get_weights()
        tau = self.hyper_params.target_update_fraction
        new_weights = [tau * train[i] + (1 - tau) * target[i] for i in range(len(train))]
        self.target_model.set_weights(new_weights)
