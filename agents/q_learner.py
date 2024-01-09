import numpy as np
import numpy.typing as npt
from . import BaseAgent, BatchGeneratorType, ModelHyperParams
from tensorflow.keras.layers import Concatenate, Dense, Input
from tensorflow.keras.models import Model


class QLearner(BaseAgent):
    """ This inherits from the BaseAgent class and implements the very simple Q-Learner algorithm without a target network.
    Note, that this class in turn serves as a base class for further variants of the Q-Learner, such as one with a target
    network and also the Double-Q-Learner addressing the overestimation problem."""
    
    name = "q_learner"
    generator_type = BatchGeneratorType.STATE_AND_ACTION_AS_INPUTS
    
    def get_action(self, state: npt.ArrayLike, is_training: bool = False) -> npt.ArrayLike:
        if self._should_make_random_action and is_training:
            return self._get_random_action()

        actions = self.env_handler.get_all_actions()
        action_dim = self.env_handler.get_action_dim()
        qs = self.model.predict([np.tile(state[None,:], (len(actions), 1)), actions[:,None]], verbose=0)[:,0]
        return  actions[np.argmax(qs)]

    def build_targets(
        self,
        actions: npt.ArrayLike, rewards: npt.ArrayLike, states: npt.ArrayLike, next_states: npt.ArrayLike
    ) -> npt.ArrayLike:
        dim = next_states.shape[0]
        predict = lambda a: self.model.predict([next_states, np.repeat(a[None,...], dim)], verbose=0)
        max_qs = np.max([predict(a) for a in self.env_handler.get_all_actions()], axis=0)[:, 0]
        return (rewards + self.hyper_params.gamma * max_qs) / (1 + self.hyper_params.gamma)

    def _get_model(self, hyper_params: ModelHyperParams) -> Model:
        state_input = Input(shape=(self.env_handler.get_state_dim(),), name="state_input")
        action_input = Input(shape=(self.env_handler.get_action_dim(),), name="action_input")
        merged = Concatenate(axis=1)([state_input, action_input])

        output = merged
        for i in range(hyper_params.no_hidden_layers):
            output = Dense(hyper_params.units_per_hidden_layer, activation="relu",
                           kernel_regularizer=hyper_params.kernel_regularizer,
                           kernel_initializer=hyper_params.kernel_initializer, name="hidden_"+str(i))(output)
            
        output = Dense(1, activation="sigmoid", name="output")(output)
        model = Model(inputs=[state_input, action_input], outputs=output)
        model.compile(optimizer=hyper_params.optimizer, loss='mse')
        return model
