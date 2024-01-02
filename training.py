import gymnasium as gym
import numpy as np
from experience_buffer import ExperienceBuffer, BatchGeneratorType


ENV_NAME = "CartPole-v1"
STATE_SPACE_NORMALISATION = np.array([4.8, 1.0, 0.418, 1.0])
env = gym.make(ENV_NAME)


def preprocess_env_data(state, reward=None, terminated=None, truncated=None, info=None):
    normalised_state = state / STATE_SPACE_NORMALISATION
    if reward is None or terminated is None or truncated is None:
        return normalised_state[2:] # only keep orientation of pole

    reward = 0.0 if terminated or truncated else 1.0
    return normalised_state[2:], reward, terminated, truncated, info


def test_agent(agent, episodes):
    lengths = []
    
    for e in range(episodes):
        state, info = env.reset()
        state = preprocess_env_data(state)
        terminated, truncated = False, False
        counter = 0

        while not terminated and not truncated and counter <= 200:
            action = agent.get_action(state)
            state, reward, terminated, truncated, info = preprocess_env_data(*env.step(action))
            counter += 1

        lengths.append(counter)
        
    median_episode_length = np.median(lengths)
    return median_episode_length
    

def train(agent, batch_size, epochs, steps_per_epoch, max_buffer_size, only_on_policy=True):
    buffer = ExperienceBuffer(max_buffer_size=max_buffer_size)
    previous_median_episode_length = 20.0
    agent.save_weights()
    
    for e in range(epochs):
        state, info = env.reset(seed=42+e)
        state = preprocess_env_data(state)
        
        for _ in range(steps_per_epoch):
            action = agent.get_action(state, is_training=True)
            next_state, reward, terminated, truncated, info = preprocess_env_data(*env.step(action))
            sample_weight = previous_median_episode_length if terminated or truncated else 1.0

            buffer.save_experience(state, action, reward, sample_weight, next_state)
            state = next_state
        
            if terminated or truncated:
                state, info = env.reset()
                state = preprocess_env_data(state)

        buffer.prepare_experience_for_training(target_builder_fct=agent.build_targets)
        agent.train(buffer.batch_generator(batch_size, agent.generator_type))
    
        median_episode_length = test_agent(agent, episodes=20)
        ##################################################################################################################
        # TODO: Encapsulate and polish the following. Idea: abstract the training loop from the problem (the number 20).
        if previous_median_episode_length < 20.0 and median_episode_length < 20.0:
            print("Failure...")
            return False
        elif 200 < median_episode_length:
            print("Agent trained. Yay!")
            return True 
        if median_episode_length <= previous_median_episode_length:
            print("Median Episode-length: ", median_episode_length, " Let's get some more training data and try again!")
            agent.load_weights()
        else:
            print("Median Episode-length: ", median_episode_length, " Updating agent.")
            agent.save_weights()
            agent.decay_epsilon()
            previous_median_episode_length = median_episode_length
            if only_on_policy:
                buffer.flush()
    
    env.close()
