import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from agents import DoubleQLearner, ModelHyperParams, PolicyGradient, QLearner, QLearnerWithTargetNetwork
from environment_handler import CartPoleHandler
from scripts import get_model_comparison_data, TrainingHyperParams


training_params = TrainingHyperParams(
    batch_size=256,
    epochs=30,
    steps_per_epoch=400*256,
    max_buffer_size=10*400*256
)
policy_gradient_params = ModelHyperParams(
    initial_epsilon=0.5,
    epsilon_decay_constant=0.7,
    no_hidden_layers=2,
    units_per_hidden_layer=12,
    optimizer="adam",
    kernel_initializer="he_uniform"
)
q_learner_params = ModelHyperParams(
    gamma=0.1,
    initial_epsilon=0.5,
    epsilon_decay_constant=0.7,
    no_hidden_layers=2,
    units_per_hidden_layer=12,
    optimizer="adam",
    kernel_initializer="he_uniform"
)
q_learner_with_target_params = ModelHyperParams(
    gamma=0.1,
    initial_epsilon=0.5,
    epsilon_decay_constant=0.7,
    no_hidden_layers=2,
    units_per_hidden_layer=12,
    optimizer="adam",
    kernel_initializer="he_uniform",
    cycles_per_target_update=1,
    target_update_fraction=0.9
)
double_q_learner_params = ModelHyperParams(
    gamma=0.1,
    initial_epsilon=0.5,
    epsilon_decay_constant=0.7,
    no_hidden_layers=2,
    units_per_hidden_layer=12,
    optimizer="adam",
    kernel_initializer="he_uniform",
    cycles_per_target_update=1,
    target_update_fraction=0.9
)

make_env_handler_fct = lambda: CartPoleHandler()
make_agent_fct_dict = {
    "policy_gradient": lambda e: PolicyGradient(e, policy_gradient_params),
    "q_learner": lambda e: QLearner(e, q_learner_params),
    "q_learner_with_target": lambda e: QLearnerWithTargetNetwork(e, q_learner_with_target_params),
    "double_q_learner": lambda e: DoubleQLearner(e, double_q_learner_params),
}


get_model_comparison_data(make_env_handler_fct, make_agent_fct_dict, training_params,
                          runs_per_agent=5, file_name="model_comparison")