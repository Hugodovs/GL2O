from abc import ABC, abstractmethod

class Optimizer(ABC):

    def __init__(self, config):
        self.__dict__.update(config)

    @abstractmethod
    def reset(self, seed):
        pass

    @abstractmethod
    def ask(self):
        pass

    @abstractmethod
    def tell(self, candidates, evaluations):
        pass  




