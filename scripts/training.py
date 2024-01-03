import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from experience_buffer import ExperienceBuffer, BatchGeneratorType
from environment_handler import Evaluation


@dataclass
class TrainingHyperParams():
    batch_size: int
    epochs: int
    steps_per_epoch: int
    max_buffer_size: int
    only_on_policy: bool=True


def train(env_handler, agent, training_params: TrainingHyperParams):
    buffer = ExperienceBuffer(max_buffer_size=training_params.max_buffer_size)
    prev_benchmark = None
    agent.save_weights()
    
    for e in range(training_params.epochs):
        state = env_handler.get_initial_state()
        
        for _ in range(training_params.steps_per_epoch):
            action = agent.get_action(state, is_training=True)
            next_state, reward, finished, sample_weight = env_handler.perform_action(action)
            buffer.save_experience(state, action, reward, sample_weight, next_state)
            state = env_handler.get_initial_state() if finished else next_state

        buffer.prepare_experience_for_training(target_builder_fct=agent.build_targets)
        agent.train(buffer.batch_generator(training_params.batch_size, agent.generator_type))
    
        benchmark = env_handler.benchmark_agent(agent)
        evaluation = env_handler.evaluate_benchmark(prev_benchmark, benchmark)
        
        if evaluation == Evaluation.RESET_MODEL:
            print("Current Benchmark Result: ", benchmark, "Previously: ", prev_benchmark, ". Reinitializing...")
            agent.reset_model()
        elif evaluation == Evaluation.AGENT_TRAINED:
            print("Agent trained. Yay!")
            env_handler.close_env()
            return True 
        elif evaluation == Evaluation.NEEDS_MORE_DATA:
            print("Current Benchmark Result: ", benchmark, "Previously: ", prev_benchmark, ". Collecting more data...")
            agent.load_weights()
        elif evaluation == Evaluation.UPDATE_WEIGHTS:
            print("Current Benchmark Result: ", benchmark, "Previously: ", prev_benchmark, ". Saving weights...")
            agent.save_weights()
            agent.decay_epsilon()
            prev_benchmark = benchmark
        
        if training_params.only_on_policy and evaluation in [Evaluation.UPDATE_WEIGHTS, Evaluation.NEEDS_MORE_DATA]:
            buffer.flush()
    
    env_handler.close_env()
    return False
