from ..tasks.bbob import BBOBTask
from .task_distribution import TaskDistribution
from ...utils.generator import DeterministicGenerator, RandomGenerator

class BBOBTaskDistribution(TaskDistribution):

    def __init__(self, config):
        super().__init__(config)
        
        # Create generator:
        if self.generator_config['mode'] == 'deterministic':
            self.generator = DeterministicGenerator(self.generator_config['f_ids'], self.generator_config['i_ids'])
        elif self.generator_config['mode'] == 'random':
            self.generator = RandomGenerator(self.generator_config['f_ids'], self.generator_config['i_ids'], self.generator_config['seed'])

    def sample(self):
        f_id, i_id = next(self.generator)
        return BBOBTask({'f_id': f_id, 'i_id': i_id, 'dim': 2, 'targets_amount': 51, 'targets_precision': -3, 'save_candidates': self.save_candidates, 'simplified_loss': self.simplified_loss}).build()
        
