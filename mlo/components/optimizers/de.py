import numpy as np
import nevergrad as ng

from .optimizer import Optimizer

class DE(Optimizer):

    def __init__(self, config):
        super().__init__(config)

    def reset(self, seed):
        import nevergrad as ng

        self.optimizer = ng.optimizers.DE(parametrization=self.dim, budget=self.budget)
        self.optimizer.parametrization.random_state = np.random.RandomState(seed)

    def ask(self):
        candidate = np.array([self.optimizer.ask().value])
        return candidate

    def tell(self, candidates, evaluations):
        candidate = ng.p.Array(shape=(self.dim,))
        candidate.value = candidates[0]
        self.optimizer.tell(candidate, evaluations[0])