from ..factories.task_distribution import TaskDistributionFactory
from ..factories.meta_optimizer import MetaOptimizerFactory
from ..factories.optimizer import OptimizerFactory
from ..factories.visualizer import VisualizerFactory
from ..components.infoset import InfoSet

class ComponentBuilder:

    @staticmethod
    def build_optimizer(optimizer_config):
        return OptimizerFactory.build_optimizer(optimizer_config)
    
    @staticmethod
    def build_meta_optimizer(meta_optimizer_config):
        return MetaOptimizerFactory.build_meta_optimizer(meta_optimizer_config)
        
    @staticmethod
    def build_task_distribution(task_distribution_config):
        return TaskDistributionFactory.build_task_distribution(task_distribution_config)

    @staticmethod
    def build_visualizer(visualizer_config):
        return VisualizerFactory.build_visualizer(visualizer_config)

    @staticmethod
    def build_infosets(infoset_config):
        return InfoSet(infoset_config)

    @staticmethod
    def build_meta_infosets():
        raise NotImplementedError("ComponentBuilder build_meta_infosets")