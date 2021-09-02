  
#import os
#from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np

class Task(ABC):

    def __init__(self, config):

        self._data = {}
        self._targets = {'targets': None}
        self.nb_evals = 0

        self.__dict__.update(config)

    @abstractmethod        
    def build(self):
        pass

    @abstractmethod
    def evaluate(self, candidate):
        pass

    @abstractmethod        
    def close(self):
        pass