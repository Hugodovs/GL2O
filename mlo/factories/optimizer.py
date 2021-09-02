from ..components.optimizers.cmaes import CMAES
from ..components.optimizers.random_search import RandomSearch
from ..components.optimizers.shiwa import Shiwa
from ..components.optimizers.cmandas2 import CMandAS2
from ..components.optimizers.cma import CMA
from ..components.optimizers.cmandas3 import CMandAS3
from ..components.optimizers.oneplusone import OnePlusOne
from ..components.optimizers.de import DE
from ..components.optimizers.pso import PSO
from ..components.optimizers.lioh import LIOH


class OptimizerFactory:

    def build_optimizer(config):
        _id = config['id']
        if _id == 'CMAES': return CMAES(config)
        elif _id == 'RandomSearch': return RandomSearch(config)
        elif _id == 'Shiwa': return Shiwa(config)
        elif _id == 'CMandAS2': return CMandAS2(config)
        elif _id == 'CMA': return CMA(config)
        elif _id == 'CMandAS3': return CMandAS3(config)
        elif _id == 'OnePlusOne': return OnePlusOne(config)
        elif _id == 'DE': return DE(config)
        elif _id == 'PSO': return PSO(config)
        elif _id == 'LIOH': return LIOH(config)
        