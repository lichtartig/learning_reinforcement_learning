from . import BaseAgent
import numpy.typing as npt


class RandomAgent(BaseAgent):
    """ This agent will always randomly select an action from the environments action space. It is not trainable and is
    intended to serve as a baseline. """
    
    name = "random_agent"
    
    def get_action(self, state: npt.ArrayLike, is_training: bool = False) -> npt.ArrayLike:
        return self._get_random_action()