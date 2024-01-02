import numpy as np
from .base_agent import BaseAgent
from .base_neural import BaseNeural, HyperParams
from experience_buffer import BatchGeneratorType
from tensorflow.keras.layers import Concatenate, Dense, Input
from tensorflow.keras.models import Model


class QLearner(BaseAgent, BaseNeural):
    name = "q_learner"
    generator_type = BatchGeneratorType.STATE_AND_ACTION_AS_INPUTS
    
    def __init__(self, env, hyper_params: HyperParams):
        super().__init__(env=env)
        self.model = self._get_model(hyper_params)
        self.epsilon = hyper_params.initial_epsilon
        self.epsilon_decay_constant = hyper_params.epsilon_decay_constant
        self.gamma = hyper_params.gamma
    
    def get_action(self, state, is_training=False):
        if self._should_make_random_action and is_training:
            return self._get_random_action()
        
        actions = self._get_all_actions()
        qs = [self.model.predict([state[None,:], a.reshape(-1, self.action_dim)], verbose=0) for a in actions]
        return actions[np.argmax(qs)]

    def build_targets(self, actions, rewards, states, next_states):
        dim = next_states.shape[0]
        action_space = self._get_all_actions()
        predict = lambda a: self.model.predict([next_states, np.repeat(a[None,...], dim)], verbose=0)
        max_qs = np.max([predict(a) for a in action_space], axis=0)[0]
        return rewards + self.gamma * max_qs

    def _get_model(self, hyper_params: HyperParams):
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
