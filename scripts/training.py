import gymnasium as gym
import numpy as np
from agents import BaseAgent
from dataclasses import dataclass
from environment_handler import EnvironmentHandler, Evaluation
from experience_buffer import ExperienceBuffer, BatchGeneratorType


@dataclass
class TrainingHyperParams():
    """ This contains all parameters related to the training loop. Parameters referring to the model architecture are not part
    of this. """
    batch_size: int
    epochs: int
    steps_per_epoch: int
    max_buffer_size: int
    only_on_policy: bool=True


def train(env_handler: EnvironmentHandler, agent: BaseAgent, train_params: TrainingHyperParams,
          verbose: int = 1, needs_data_for_model_comparison: bool = False) -> int:
    """ This is the main training loop for a given environment and agent. It is configurable through the 'train_params'. """
    
    buffer = ExperienceBuffer(max_buffer_size=train_params.max_buffer_size)
    prev_benchmark = None
    agent.save_weights()
    
    for e in range(train_params.epochs):
        state = env_handler.get_initial_state()
        
        for _ in range(train_params.steps_per_epoch):
            action = agent.get_action(state, is_training=True)
            next_state, reward, finished, sample_weight = env_handler.perform_action(action)
            buffer.save_experience(state, action, reward, sample_weight, next_state)
            state = env_handler.get_initial_state() if finished else next_state

        buffer.prepare_experience_for_training(target_builder_fct=agent.build_targets)
        agent.train(buffer.batch_generator(train_params.batch_size, agent.generator_type))
    
        benchmark = env_handler.benchmark_agent(agent)
        evaluation = env_handler.evaluate_benchmark(prev_benchmark, benchmark)
        
        if evaluation == Evaluation.RESET_MODEL:
            if verbose == 1:
                print("Current Benchmark Result: ", benchmark, "Previously: ", prev_benchmark, ". Reinitializing...")
            agent.reset_model()
        elif evaluation == Evaluation.AGENT_TRAINED:
            if verbose == 1:
                print("Current Benchmark Result: ", benchmark, "Previously: ", prev_benchmark, ". Agent trained. Yay!")
            env_handler.close_env()
            return e+1
        elif evaluation == Evaluation.NEEDS_MORE_DATA:
            if verbose == 1:
                print("Current Benchmark Result: ", benchmark, "Previously: ", prev_benchmark, ". Collecting more data...")
            agent.load_weights()
            if needs_data_for_model_comparison:
                benchmark = env_handler.benchmark_agent(agent, replace_last_entry=True)
                
        elif evaluation == Evaluation.UPDATE_WEIGHTS:
            if verbose == 1:
                print("Current Benchmark Result: ", benchmark, "Previously: ", prev_benchmark, ". Saving weights...")
            agent.save_weights()
            agent.decay_epsilon()
            prev_benchmark = benchmark
        
            if train_params.only_on_policy:
                buffer.flush()
    
    env_handler.close_env()
    if verbose == 1:
        print("Finished training loop. Agent did not converge.")
    return -1
