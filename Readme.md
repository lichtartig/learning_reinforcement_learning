# Reinforcement Learning Study with the OpenAI-Gym
The goal of this project is to implement and train multiple reinforcement learning algorithms as an exercise and for study purposes. Since this is still work in progress, I will incrementally add more algorithms/environments/analyses to this repository.

Currently, it contains fully functional implementations of the following algorithms
* *Policy Gradient*
* *Deep Q-Learner*
* *Deep Q-Learner with Target Network*
* *Double-Q-Learner* using the target network as in orig. reference by van Hasselt, et al. (2015)

Moreover, it contains the code for a generic, tuneable training loop and an experience buffer. Finally, it contains a script to allow for structured hyperparameter search, which computes p-values by means of a permutation test to make sure the differences are statistically significant.
Crucially, all of these components have been implemented in a completely generic way without any reference to the specific environment. This allowed us to start working on the simple environment `CartPole-v1` before moving to more complex environments. The idea behind this is 2-fold:
1. Make sure all the code works correctly. 
2. Learn lessons for hyperparameter tuning in an easier setting, before passing to environments requiring longer training times and more computational resources.

## File Organisation: Where to find what?
* `Results.ipynb`: This file presents the findings of this project and explanations of the details. It should serves as a starting point and we refer to the remaining code only for implementation details.
* `agents/`: This module contains the implementations of all algorithms mentioned above. The class `BaseAgent` serves as an abstract base class for all algorithms, containing common logic - e.g. to save weights - while the `RandomAgent` can be used as a base line against "real" models. Since some models share a lot of code (e.g. the `QLearner` and the `QLearnerWithTargetNetwork`), they inherit from each other.
* `scripts/`: Logic containing the training-loop or concerning parameter search is contained in this module.
* `environment_handler/`: Contains an interface to encapsulate any logic that is specific to an environment. We can then concentrate an environment into a single implementation -- e.g. `CartPole` -- and decouple the remaining code base from it.
* `experience_buffer/`: A module for a completely generic Experience Buffer.

## Next steps
* Move to more complex environments, particularly ones where the *credit assignment problem* becomes more pressing.
* Implement algorithms that address the *credit assignment problem*, i.e. *actor-critic*, *actor-critic with clipped double-Q-critic*, *soft-actor-critic*