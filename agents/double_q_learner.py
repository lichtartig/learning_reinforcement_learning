import numpy as np
from . import BatchGeneratorType, ModelHyperParams, QLearnerWithTargetNetwork


class DoubleQLearner(QLearnerWithTargetNetwork):
    name = "double_q_learner"
    generator_type = BatchGeneratorType.STATE_AND_ACTION_AS_INPUTS

    def build_targets(self, actions, rewards, states, next_states):
        dim = next_states.shape[0]
        predict = lambda a: self.model.predict([next_states, np.repeat(a[None,...], dim)], verbose=0)
        max_a = np.argmax([predict(a) for a in self.all_actions], axis=0)[:,0]
        max_qs = self.target_model.predict([next_states, max_a], verbose=0)[:,0]
        return (rewards + self.hyper_params.gamma * max_qs) / (1 + self.hyper_params.gamma)
