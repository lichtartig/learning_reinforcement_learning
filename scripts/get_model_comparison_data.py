import os
import pickle
from .training import train, TrainingHyperParams
from agents import BaseAgent, ModelHyperParams
from environment_handler import EnvironmentHandler
from typing import Callable


FILE_PATH = "pkls_and_logs"


def _get_updated_results_dict(
    results_dict: dict[tuple[str, int, int], list[int]], agent_name: str, run_id: int, new_results: list[list[int]]
) -> dict[tuple[str, int, int], list[int]]:
    for epoch, results in enumerate(new_results):
        results_dict[(agent_name, run_id, epoch)] = results

    return results_dict


def _safe_results(results_dict: dict[tuple[str, int, int], list[int]], file_name: str):
    full_path = os.path.join(FILE_PATH, file_name) + '.pkl'
    with open(full_path, "wb") as file:
        pickle.dump(results_dict, file)


def get_model_comparison_data(
    make_env_handler_fct: Callable[[], EnvironmentHandler],
    make_agent_fct_dict: dict[str, Callable[[EnvironmentHandler], BaseAgent]],
    training_params: TrainingHyperParams,
    runs_per_agent: int,
    file_name: str
):
    try:
        full_path = os.path.join(FILE_PATH, file_name) + '.pkl'
        results_dict = pickle.load(open(full_path, 'rb'))
    except:
        results_dict = {}
    
    for run_id in range(runs_per_agent):
        for agent_name, make_agent_fct in make_agent_fct_dict.items():
            already_has_data_from_previous_execution = (agent_name, run_id, 0) in results_dict.keys()
            if already_has_data_from_previous_execution:
                continue
            
            env_handler = make_env_handler_fct()
            env_handler.enable_model_comparison_mode()
            agent = make_agent_fct(env_handler)
            _ = train(env_handler, agent, training_params)
            results_per_epoch = env_handler.get_benchmark_results()
            results_dict = _get_updated_results_dict(results_dict, agent_name, run_id, results_per_epoch)
            _safe_results(results_dict, file_name)
            
