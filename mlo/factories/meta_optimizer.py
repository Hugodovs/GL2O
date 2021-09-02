from ..components.metaoptimizers.deepga_table.deepga import TruncatedRealMutatorGA_Table

class MetaOptimizerFactory:

    def build_meta_optimizer(config):
        _id = config['id']
        if _id == 'TruncatedRealMutatorGA_Table': return TruncatedRealMutatorGA_Table(config)
