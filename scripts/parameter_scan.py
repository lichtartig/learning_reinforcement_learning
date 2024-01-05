import numpy as np
from .training import train
from itertools import permutations


MINIMUM_NUMBER_TRAINING_RUNS = 5
GENERIC_LOG_STRING = "------------------------------------------------------------------------------------------------------\ntraining_runs: {}\nproportion_trained: {}\nmean_episodes_needed: {}\n"
P_VALUE_LOG_STRING = "p-value of {} and {}: {}\n"


def _get_proportion_trained(results):
    trained_episodes = list(filter(lambda i: i!=-1, results))
    return np.round(len(trained_episodes) / len(results), 2)


def _get_mean_episodes_needed(results):
    trained_episodes = list(filter(lambda i: i!=-1, results))
    return np.round(np.mean(trained_episodes), 2) if 0 < len(trained_episodes) else None


def _get_p_value(sample_a, sample_b):
    real_result = _get_proportion_trained(sample_a)
    n = len(sample_a)
    permutation_results = [_get_proportion_trained(p[:n]) for p in permutations(sample_a + sample_b)]
    is_smaller = [r <= real_result for r in permutation_results]
    p_value = sum(is_smaller) / len(is_smaller)
    return np.round(p_value, 2)

    
def _needs_more_data(param_name, results_dict):
    training_runs = {name: len(results) for name, results in results_dict.items()}
    proportion_trained = {name: _get_proportion_trained(results) for name, results in results_dict.items()}
    mean_episodes_needed = {name: _get_mean_episodes_needed(results) for name, results in results_dict.items()}    

    with open('best_batch_size_log', 'a') as file:
        file.write(GENERIC_LOG_STRING.format(training_runs, proportion_trained, mean_episodes_needed))

    if len(results_dict.get(param_name, [])) < MINIMUM_NUMBER_TRAINING_RUNS:
        return True

    best_params = sorted(proportion_trained.keys(), key=lambda k: proportion_trained[k])
    comparison_param_name = best_params[-1] if best_params[-1] != param_name else best_params[-2]
    p_value = _get_p_value(results_dict[param_name], results_dict[comparison_param_name])
    
    with open('best_batch_size_log', 'a') as file:
        file.write(P_VALUE_LOG_STRING.format(param_name, comparison_param_name, p_value))
        
    return False if 0.95 <= p_value or p_value <= 0.05 else True


def parameter_scan(hyper_param_dict, make_agent_fct, make_env_handler_fct, verbose=0):
    results_dict = {}
    param_config_needs_more_data = {param_name: True for param_name in hyper_param_dict}
    
    while any(param_config_needs_more_data.values()):
        for param_name, (training_params, model_params) in hyper_param_dict.items():
            param_config_needs_more_data[param_name] = _needs_more_data(param_name, results_dict)
            if not param_config_needs_more_data[param_name]:
                continue
            
            env_handler = make_env_handler_fct()
            agent = make_agent_fct(env_handler.get_action_space(), env_handler.get_state_space(), model_params)
            episodes_needed = train(env_handler, agent, training_params, verbose=verbose)
            
            results_dict[param_name] = results_dict.get(param_name, []) + [episodes_needed]   
