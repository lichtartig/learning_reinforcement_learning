import numpy as np
from . import BaseAgent, BatchGeneratorType, ModelHyperParams
from tensorflow.keras.layers import Concatenate, Dense, Input
from tensorflow.keras.models import Model


class QLearner(BaseAgent):
    name = "q_learner"
    generator_type = BatchGeneratorType.STATE_AND_ACTION_AS_INPUTS
    
    def get_action(self, state, is_training=False):
        if self._should_make_random_action and is_training:
            return self._get_random_action()
        
        qs = [self.model.predict([state[None,:], a.reshape(-1, self.action_dim)], verbose=0) for a in self.all_actions]
        return  self.all_actions[np.argmax(qs)]

    def build_targets(self, actions, rewards, states, next_states):
        dim = next_states.shape[0]
        predict = lambda a: self.model.predict([next_states, np.repeat(a[None,...], dim)], verbose=0)
        max_qs = np.max([predict(a) for a in self.all_actions], axis=0)[:, 0]
        return (rewards + self.hyper_params.gamma * max_qs) / (1 + self.hyper_params.gamma)

    def _get_model(self, hyper_params: ModelHyperParams):
        state_input = Input(shape=(self.state_dim,), name="state_input")
        action_input = Input(shape=(self.action_dim,), name="action_input")
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
