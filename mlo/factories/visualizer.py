from ..components.visualizers.heatmap import HeatmapVisualizer
from ..components.visualizers.ecdf import ECDFVisualizer

class VisualizerFactory:

    def build_visualizer(config):
        _id = config['id']
        if _id == 'heatmap': return HeatmapVisualizer(config)
        if _id == 'ECDF': return ECDFVisualizer(config)
        
