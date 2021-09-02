class PreprocessingOptimizationExperiment():

    def __init__(self, optimizer, task_distribution, infoset):
        self.optimizer = optimizer
        self.task_distribution = task_distribution
        self.infoset = infoset


    def run(self):
        print("PreprocessingOptimization running...")
        
        # Loop batch of tasks
        for batch in range(self.task_distribution.internals['batch_size']):
            print(batch)
            # Restart episode
            task = self.task_distribution.sample()
            self.optimizer.reset(seed=batch+1)  
            
            # Loop task:
            while (self.task_distribution.internals['max_budget_per_task'] >= task.nb_evals):

                # Ask, evaluate and tell
                candidates = self.optimizer.ask()
                evaluations = [task.evaluate(candidate) for candidate in candidates]
                self.optimizer.tell(candidates, evaluations)

            task_data, targets = task.close()
            self.infoset.append_task_data(self.optimizer.id, task.id, task_data)
            self.infoset.set_task_data(self.optimizer.id, task.id, targets)
        
        # Save infoset
        self.infoset.save()