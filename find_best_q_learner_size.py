import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from agents import ModelHyperParams, QLearner
from environment_handler import CartPoleHandler
from scripts import parameter_scan, TrainingHyperParams


ALLOWED_EPOCHS_TO_TRAIN = 10


def get_model_params(no_layers, no_units):
    return ModelHyperParams(
        gamma=0.5,
        initial_epsilon=0.5,
        epsilon_decay_constant=0.7,
        no_hidden_layers=no_layers,
        units_per_hidden_layer=no_units,
        optimizer="adam",
        kernel_initializer="he_uniform"
    )


make_agent_fct=lambda e, p: QLearner(e, p, verbose=1)
make_env_handler_fct=lambda: CartPoleHandler()

training_params = TrainingHyperParams(
    batch_size=256,
    epochs=ALLOWED_EPOCHS_TO_TRAIN,
    steps_per_epoch=400*256,
    max_buffer_size=10*400*256
)

network_sizes = [(1, 6), (1, 12), (1, 18), (1, 24), (2, 6), (2, 12), (2, 18), (2, 24)]
hyper_param_dict = {s: (training_params, get_model_params(*s)) for s in network_sizes}

parameter_scan(hyper_param_dict, make_agent_fct, make_env_handler_fct, file_name="best_q_learner_size", verbose=1)
