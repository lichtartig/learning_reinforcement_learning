import numpy as np
import pickle
from .training import train, TrainingHyperParams
from agents import BaseAgent, ModelHyperParams
from environment_handler import EnvironmentHandler
from itertools import permutations
from scipy.stats import permutation_test
from typing import Callable


TRAINING_RUNS_PER_SIGNIFICANCE_TEST = 5
MINIMUM_NUMBER_TRAINING_RUNS = 10
GENERIC_LOG_STRING = "------------------------------------------------------------------------------------------------------\ntraining_runs: {}\nproportion_trained: {}\nmean_episodes_needed: {}\n"
P_VALUE_LOG_STRING = "p-value of {} and {}: {}\n"


def _get_mean_episodes_needed(results: list[int]) -> float:
    a = np.array(results)
    trained_episodes = a[np.where(a != -1)]
    return np.round(np.mean(trained_episodes), 2) if 0 < len(trained_episodes) else None


def _get_p_value(sample_a: list[int], sample_b: list[int]) -> float:
    statistic = lambda a, b: (np.array(a) != -1).sum() - (np.array(b) != -1).sum()
    res = permutation_test((sample_a, sample_b), statistic, alternative='less')
    return np.round(res.pvalue, 2)
    

def _get_proportion_trained(results: list[int]) -> float:
    a = np.array(results)
    return np.round((np.array(a) != -1).sum() / a.size, 2)


def _needs_more_data(param_name: object, results_dict: dict[object, list[int]], file_name: str) -> bool:
    training_runs = {name: len(results) for name, results in results_dict.items()}
    proportion_trained = {name:  _get_proportion_trained(results) for name, results in results_dict.items()}
    mean_episodes_needed = {name: _get_mean_episodes_needed(results) for name, results in results_dict.items()}    

    output = GENERIC_LOG_STRING.format(training_runs, proportion_trained, mean_episodes_needed)
    _user_output(output, file_name)

    lacks_samples_for_p_test = len(results_dict.get(param_name, [])) < MINIMUM_NUMBER_TRAINING_RUNS
    if lacks_samples_for_p_test:
        return True
    
    best_params = sorted(proportion_trained.keys(), key=lambda k: proportion_trained[k])
    comparison_param_name = best_params[-1] if best_params[-1] != param_name else best_params[-2]
    
    p_value = _get_p_value(results_dict[param_name], results_dict[comparison_param_name])
    
    output = P_VALUE_LOG_STRING.format(param_name, comparison_param_name, p_value)
    _user_output(output, file_name)
        
    return False if 0.95 <= p_value or p_value <= 0.05 else True


def _pickle_intermediate_results(results_dict: dict[object, list[int]], file_name: str):
    if file_name is None:
        return
    
    with open(file_name + '.pkl', 'wb') as file:
        pickle.dump(results_dict, file)


def _user_output(text: str, file_name: str):
    if file_name is None:
        print(text)
    else:
        with open(file_name + '.log', 'a') as file:
            file.write(text)


def parameter_scan(hyper_param_dict: dict[object, tuple[TrainingHyperParams, ModelHyperParams]],
                   make_agent_fct: Callable[[EnvironmentHandler, ModelHyperParams], BaseAgent],
                   make_env_handler_fct: Callable[[], EnvironmentHandler],
                   file_name: str = None, verbose: int = 0):
    try:
        results_dict = pickle.load(open(file_name + '.pkl', 'rb'))
    except:
        results_dict = {}
    param_config_needs_more_data = {param_name: True for param_name in hyper_param_dict}
    
    while any(param_config_needs_more_data.values()):
        for param_name, (training_params, model_params) in hyper_param_dict.items():
            param_config_needs_more_data[param_name] = _needs_more_data(param_name, results_dict, file_name)
            if not param_config_needs_more_data[param_name]:
                continue
            
            for _ in range(TRAINING_RUNS_PER_SIGNIFICANCE_TEST):
                env_handler = make_env_handler_fct()
                agent = make_agent_fct(env_handler, model_params)
                episodes_needed = train(env_handler, agent, training_params, verbose=verbose)
                results_dict[param_name] = results_dict.get(param_name, []) + [episodes_needed]
                _pickle_intermediate_results(results_dict, file_name)
            
