import yaml

from .builder import ComponentBuilder
from .preprocessing_optimization import PreprocessingOptimizationExperiment
from .preprocessing_metaoptimization import PreprocessingMetaOptimizationExperiment

from .postprocessing_optimization import PostprocessingOptimizationExperiment

class Experiment:

    def __init__(self, config_path):
        
        with open(f'{config_path}') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def run(self):

        # Build Components
        self.__build_components()

        # Pre-processing Meta-Optimization
        if (self.meta_optimizer and self.optimizer and self.task_distribution and self.infoset):
            PreprocessingMetaOptimizationExperiment(self.meta_optimizer, self.optimizer, self.task_distribution, self.infoset).run()

        # Pre-processing Optimization
        elif (self.optimizer and self.task_distribution and self.infoset):
            PreprocessingOptimizationExperiment(self.optimizer, self.task_distribution, self.infoset).run()

        # Post-processing Optimization
        elif (self.visualizer and self.infoset):
            PostprocessingOptimizationExperiment(self.visualizer, self.infoset).run()

        # Post-processing Meta-Optimization
        #elif (visualizer_config and meta_infosets_config):
        #    print("post meta-opt")
        
        # Configuration File Error
        else:
            raise ValueError('Configuration File Error!')

    def __build_components(self):

        # Optimizer
        optimizer_config = self.config.get('optimizer', None)
        if (optimizer_config):
            self.optimizer = ComponentBuilder.build_optimizer(optimizer_config)
        else:
            self.optimizer = None
        
        # Meta-Optimizer
        meta_optimizer_config = self.config.get('meta_optimizer', None)
        if (meta_optimizer_config):
            self.meta_optimizer = ComponentBuilder.build_meta_optimizer(meta_optimizer_config)
        else:
            self.meta_optimizer = None

        # Task Distribution
        task_distribution_config = self.config.get('task_distribution', None)
        if (task_distribution_config):
            self.task_distribution = ComponentBuilder.build_task_distribution(task_distribution_config)
        else:
            self.task_distribution = None

        # Visualizer
        visualizer_config = self.config.get('visualizer', None)
        if (visualizer_config):
            self.visualizer = ComponentBuilder.build_visualizer(visualizer_config)
        else:
            self.visualizer = None

        # Infoset
        infoset_config = self.config.get('infoset', None)
        if (infoset_config):
            self.infoset = ComponentBuilder.build_infosets(infoset_config)
        else:
            self.infoset = None

        # Meta-Infosets
        #meta_infosets_config = self.config.get('meta_infosets', None)
        #if (meta_infosets_config):
        #    self.meta_infosets = ComponentBuilder.build_meta_infosets(meta_infosets_config)
        #else:
        #    self.optimizer = None
