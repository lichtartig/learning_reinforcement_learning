import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from agents import ModelHyperParams, PolicyGradient
from environment_handler import CartPoleHandler
from scripts import parameter_scan, TrainingHyperParams


ALLOWED_EPOCHS_TO_TRAIN = 10


def get_training_params(batch_size):
    return TrainingHyperParams(
        batch_size=batch_size,
        epochs=ALLOWED_EPOCHS_TO_TRAIN,
        steps_per_epoch=100*1024,
        max_buffer_size=10*100*1024
    )


make_agent_fct=lambda e, p: PolicyGradient(e, p, verbose=1)
make_env_handler_fct=lambda: CartPoleHandler()

model_params = ModelHyperParams(
    initial_epsilon=0.5,
    epsilon_decay_constant=0.7,
    no_hidden_layers=2,
    units_per_hidden_layer=24,
    optimizer="adam",
    kernel_initializer="he_uniform"
)

batch_sizes = [32, 64, 128, 256, 512, 1024, 2048]
hyper_param_dict = {b: (get_training_params(b), model_params) for b in batch_sizes}

parameter_scan(hyper_param_dict, make_agent_fct, make_env_handler_fct, file_name="best_batch_size", verbose=1)