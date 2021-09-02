import numpy as np

from .optimizer import Optimizer

class CMAES(Optimizer):

    def __init__(self, config):
        super().__init__(config)

        #assert(hasattr(self, 'dim'))

    def reset(self, seed):
        import cma
        
        gen = np.random.RandomState(seed)
        x0 = np.ones(self.dim) * gen.uniform(self.x0_interval[0], self.x0_interval[1])
        self.optimizer = cma.CMAEvolutionStrategy(x0, self.sigma0, {'verb_disp': 0, 'seed': seed, 'bounds': self.bounds, 'maxiter': self.maxiter})

    def ask(self):
        candidates = self.optimizer.ask()
        return candidates
    
    def tell(self, candidates, evaluations):
        self.optimizer.tell(candidates, evaluations)
