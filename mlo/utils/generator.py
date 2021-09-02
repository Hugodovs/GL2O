from abc import ABC, abstractmethod

import numpy as np

class Generator(ABC):

    def __init__(self, items, idxs, seed):
        self.items = items
        self.min_idx, self.max_idx = idxs
        self.seed = seed

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        pass

class RandomGenerator(Generator):

    def __init__(self, items, idxs, seed):
        super().__init__(items, idxs, seed)
        self.gen = np.random.RandomState(self.seed)

    def __next__(self):
        item = self.gen.choice(self.items, 1)[0]
        idx = self.gen.randint(self.min_idx, self.max_idx + 1)
        return item, idx

class DeterministicGenerator(Generator):

    def __init__(self, items, idxs, seed=None):
        super().__init__(items, idxs, seed)

        self.generator = self.chain()

    def chain(self):
        while True:
            for i in range(self.min_idx, self.max_idx + 1):
                for t in self.items:
                    yield t, i

    def __next__(self):
        return next(self.generator)