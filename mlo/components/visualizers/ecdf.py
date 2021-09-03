import numpy as np

from .visualizer import Visualizer

class ECDFVisualizer(Visualizer):

    def __init__(self, config):
        super().__init__(config)

    def compute_task_rts(self, task_runs, targets):

        def compute_rts(run, targets):
            run = np.array(run)
            evals = run[:, 0]
            losses = run[:, 1]

            rts = []
            for target in targets:
                idx = None
                for i, loss in enumerate(losses):
                    if loss <= target:
                        idx = i
                        break
                if idx == None:
                    rt = np.inf
                else:
                    rt = evals[idx]
                rts.append(rt)
            return rts

        task_rts = []
        for run in task_runs:
            rts = compute_rts(run, targets)
            task_rts.append(rts)
        return task_rts

    def get_rt_table(self, nb_evaluations, trajectories, targets):

        nb_runs = trajectories.shape[0]
        nb_precisions = targets.shape[1]

        rt_table = np.empty((nb_runs, nb_precisions))
        for i in range(nb_runs):
            for j in range(nb_precisions):
                rt_table[i, j] = self.compute_rt(nb_evaluations[i], trajectories[i], targets[i, j])
        return rt_table

    def get_simulated_rt_table(self, rt_table, nb_bootstrap, max_evaluations, dim):

        def draw_rt(runtimes, max_evaluations, dim):

            nb_runs = len(runtimes)

            #runtimes = np.random.permutation(runtimes)

            # Check if all infinity:
            if np.isinf(runtimes).all():
                return np.inf

            final_rt = 0

            curr_idxs = list(range(nb_runs))
            rt = None
            while True:
                # for i in range(nb_runs):
                # Get an runtime:
                idx = np.random.randint(len(curr_idxs))  # if rt is not None else (nb_bootstrap % nb_runs) + 1
                rt = runtimes[idx]

                # Add runtime:
                final_rt += rt if rt != np.inf else max_evaluations

                # Break if a runtime was found:
                if rt != np.inf:
                    break

            return final_rt


        # Internals:
        nb_runs = rt_table.shape[0]
        nb_precisions = rt_table.shape[1]

        # Getting the simulated rts for each task
        simulated_rt_table = np.empty((nb_bootstrap, nb_precisions))
        for i in range(nb_precisions):
            for j in range(nb_bootstrap):
                simulated_rt_table[j, i] = draw_rt(rt_table[:, i], max_evaluations, dim)

        return simulated_rt_table

    def view(self, infoset):
        print("ecdf viewing")
        
        # Runtimes
        opt_names = []
        rts_tables = [] 
        for opt_id in infoset.data.keys():
            opt_names.append(opt_id)
            print(opt_id)
            rts_table = []
            for task_id in infoset.data[opt_id].keys():
                losses = infoset.data[opt_id][task_id]['losses']
                targets = infoset.data[opt_id][task_id]['targets']
                task_rts = self.compute_task_rts(losses, targets)
                for rt in task_rts:
                    rts_table.append(rt)
            rts_table = np.array(rts_table)
            rts_tables.append(rts_table)
        rts_tables = np.array(rts_tables)
        
        # Simulated runtimes 
        simulated_rts_tables = []
        for rts_table in rts_tables:
            simulated_rts_table = self.get_simulated_rt_table(rts_table, 100, 20, 2)
            simulated_rts_tables.append(simulated_rts_table)

        # Plot ECDFs
        import plotly.graph_objects as go

        fig = go.Figure()

        for i, simulated_rts_table in enumerate(simulated_rts_tables):

            x = np.sort(simulated_rts_table.flatten()) / 2
            y = np.arange(len(x))/float(len(x))
            
            trace = go.Scatter(x=x, y=y, mode='lines', name=opt_names[i], legendgrouptitle={'text': 'Optimizers'})
            fig.add_trace(trace)
        
        fig.update_layout(xaxis = dict(showexponent = 'all', exponentformat = 'power'))
        #fig.update_layout(title_text=self.title, title_x=0.5, title_y=0.85)
        fig.update_xaxes(type="log", title="Function Evaluations / DIM")
        fig.update_yaxes(title="Proportion of trials", range=[0,1])
        #fig.layout.height = 500
        #fig.layout.width = 500
        fig.update_layout(width=500, height=500, font_family='Serif', font_size=12, margin_l=5, margin_t=5, margin_b=5, margin_r=5)
        

        fig.write_image(self.output_path)                
       
        exit()




