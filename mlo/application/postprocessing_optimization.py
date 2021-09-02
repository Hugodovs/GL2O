
class PostprocessingOptimizationExperiment():

    def __init__(self, visualizer, infoset):
        self.visualizer = visualizer
        self.infoset = infoset

    def run(self):
        print("PostprocessingOptimizationExperiment STARTED...")
        self.visualizer.view(self.infoset)
        print("PostprocessingOptimizationExperiment ENDED...")
        