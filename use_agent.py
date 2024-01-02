import gymnasium as gym
from agents.random_agent import RandomAgent
from agents.q_learner import QLearner, QLearnerHyperParams


#env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)

#agent = RandomAgent(env)

hyper_params = QLearnerHyperParams(no_hidden_layers=1, units_per_hidden_layer=16)
agent = QLearner(env, hyper_params)

for _ in range(1000):
    action = agent.get_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
        
env.close()
