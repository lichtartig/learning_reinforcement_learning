import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from agents import ModelHyperParams, QLearner
from environment_handler import CartPoleHandler
from scripts import parameter_scan, TrainingHyperParams


ALLOWED_EPOCHS_TO_TRAIN = 10


def get_model_params(gamma):
    return ModelHyperParams(
        gamma=gamma,
        initial_epsilon=0.5,
        epsilon_decay_constant=0.7,
        no_hidden_layers=2,
        units_per_hidden_layer=12,
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

gamma_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
hyper_param_dict = {g: (training_params, get_model_params(g)) for g in gamma_values}

parameter_scan(hyper_param_dict, make_agent_fct, make_env_handler_fct, file_name="best_gamma", verbose=1)
