import numpy as np

from .optimizer import Optimizer

class RandomSearch(Optimizer):

    def __init__(self, config):
        super().__init__(config)

        # population size
        # bounds
        # dim

    def reset(self, seed):
        self.generator = np.random.RandomState(seed)

    def ask(self):
        candidates = self.generator.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        return candidates

    def tell(self, candidates, evaluations):
        pass  

