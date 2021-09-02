import numpy as np

class PreprocessingMetaOptimizationExperiment():

    def __init__(self, meta_optimizer, optimizer, task_distribution, infoset):
        self.meta_optimizer = meta_optimizer
        self.optimizer = optimizer
        self.task_distribution = task_distribution
        self.infoset = infoset

    def run(self):
        print("PreprocessingMetaOptimizationExperiment")

        self.meta_optimizer.initialize(self.optimizer.get_params())
        self.meta_optimizer.run(self.meta_loss, steps=5)



    def meta_loss(self, x):
        # Set Policy:
        self.optimizer.set_params(x)

        # Loop batch of tasks
        for batch in range(self.task_distribution.internals['batch_size']):
        
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
        metaloss = self.calculate_meta_loss()
        return metaloss

    def calculate_meta_loss(self):
        
        def compute_task_rts(task_runs, targets):

            def compute_rts(run, targets):
                run = np.array(run)
                evals = run[:, 0]
                losses = run[:, 1]
                
                rts = []
                for target in targets:
                    rt = np.where(losses <= target)[0]
                    if len(rt) == 0:
                        rt = np.inf
                    else: 
                        rt = evals[rt[0]]
                    rts.append(rt)
                return rts

            task_rts = []
            for run in task_runs:
                rts = compute_rts(run, targets)
                task_rts.append(rts)
            return task_rts

        rts = [] 
        for opt_id in self.infoset.data.keys():
            for task_id in self.infoset.data[opt_id].keys():
                losses = self.infoset.data[opt_id][task_id]['losses']
                targets = self.infoset.data[opt_id][task_id]['targets']
                task_rts = compute_task_rts(losses, targets)
                rts += task_rts
        rts = np.array(rts)

        # Calculate Expected Runtime:
        max_evaluations = 200
        nb_runs = rts.shape[0]
        nb_succ = len(rts[np.isfinite(rts)])
        total_eval_succ = np.sum(rts[np.isfinite(rts)])
        p_succ = nb_succ / nb_runs

        if nb_succ != 0:
            expected_runtime = ((1 - p_succ)/ p_succ) * max_evaluations + (total_eval_succ/nb_succ)
        else: 
            expected_runtime = np.inf

        return expected_runtime