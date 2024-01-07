from . import BaseAgent
import numpy.typing as npt


class RandomAgent(BaseAgent):
    name = "random_agent"

    
    def get_action(self, state: npt.ArrayLike, is_training: bool = False) -> npt.ArrayLike:
        return self._get_random_action()