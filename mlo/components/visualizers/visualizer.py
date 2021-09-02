from abc import ABC, abstractmethod

class Visualizer(ABC):

    def __init__(self, config):
        self.__dict__.update(config)
        
    @abstractmethod
    def view(self):
        pass 

    


