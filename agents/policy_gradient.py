import numpy as np
from . import BaseAgent, BatchGeneratorType, ModelHyperParams
from tensorflow.keras.layers import Concatenate, Dense, Input
from tensorflow.keras.models import Model


class PolicyGradient(BaseAgent):
    name = "policy_gradient"
    generator_type = BatchGeneratorType.ONLY_STATE_AS_INPUT
    
    def get_action(self, state, is_training=False):
        if self._should_make_random_action and is_training:
            return self._get_random_action()

        pred = self.model.predict(state[None,:], verbose=0)
        return np.argmax(pred)

    def build_targets(self, actions, rewards, states, next_states):
        rescaled_rewards = 2*(rewards - 0.5)
        categorical_actions = self._get_categorical_action_encoding(actions)
        targets = np.transpose(categorical_actions * rescaled_rewards)
        return targets

    def _get_model(self, hyper_params: ModelHyperParams):
        state_input = Input(shape=(self.state_dim,), name="state_input")

        top_layer = state_input
        for i in range(hyper_params.no_hidden_layers):
            top_layer = Dense(hyper_params.units_per_hidden_layer, activation="relu",
                           kernel_regularizer=hyper_params.kernel_regularizer,
                           kernel_initializer=hyper_params.kernel_initializer, name="hidden_"+str(i))(top_layer)

        d = len(self.all_actions)
        output = Dense(d, activation="softmax", name="output")(top_layer)
        model = Model(inputs=state_input, outputs=output)
        model.compile(optimizer=hyper_params.optimizer, loss='categorical_crossentropy')
        return model
