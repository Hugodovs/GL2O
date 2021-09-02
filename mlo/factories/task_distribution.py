from ..components.task_distributions.bbob import BBOBTaskDistribution

class TaskDistributionFactory:

    def build_task_distribution(config):
        _id = config['id']
        if _id == 'cocobenchmark': return BBOBTaskDistribution(config)
