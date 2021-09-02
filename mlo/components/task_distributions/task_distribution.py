from abc import ABC, abstractmethod

class TaskDistribution(ABC):

    def __init__(self, config):
        self.__dict__.update(config)
        
    @abstractmethod
    def sample(self):
        pass 

    


