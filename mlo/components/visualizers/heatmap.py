import numpy as np
from scipy.stats import mannwhitneyu

from .visualizer import Visualizer

class HeatmapVisualizer(Visualizer):

    def __init__(self, config):
        super().__init__(config)

    def compute_task_rts(self, task_runs, targets):

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

    def calculate_winning_rate(self, rts_table1, rts_table2):

        nb_runs, nb_targets = rts_table1.shape
        nb_trials = nb_runs * nb_targets
        winning1, winning2, tie = 0, 0, 0

        for i in range(nb_runs):
            for j in range(nb_targets):
                if rts_table1[i][j] == rts_table2[i][j]:
                    tie += 1
                elif rts_table1[i][j] < rts_table2[i][j]:
                    winning1 += 1
                else:
                    winning2 += 1
        no_tie = nb_trials - tie
        return winning1/no_tie, winning2/no_tie

            

    def view(self, infoset):

        # Runtimes
        opt_names = []
        rts_tables = [] 
        for opt_id in infoset.data.keys():
            opt_names.append(opt_id)
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

        # Winning Rates
        winning_rates = np.ones((len(opt_names), len(opt_names)))*-1
        stats = np.chararray((len(opt_names), len(opt_names)), unicode=True)
        successful_runs = np.ones((len(opt_names), 2))
        average_winning = np.ones((len(opt_names)))
        
        for i, opt_id in enumerate(infoset.data.keys()):
            rts_table1 = rts_tables[i]
            successful_runs[i][0] = np.isinf(rts_table1.flatten()).sum()  
            successful_runs[i][1] = len(rts_table1.flatten())

            for j, _ in enumerate(infoset.data.keys()):
                if i == j:
                    winning_rates[i][j] = 0.5
                    stats[i][j] = '*'
                if i < j:
                    
                    rts_table2 = rts_tables[j]
                    
                    rate1, rate2 = self.calculate_winning_rate(rts_table1, rts_table2)
                    stat, p = mannwhitneyu(rts_table1.flatten(), rts_table1.flatten())
                    if p > 0.95:
                        stats[i][j] = '*'
                        stats[j][i] = '*'
                    else: 
                        stats[i][j] = ''
                        stats[j][i] = ''
                    winning_rates[i][j] = rate1
                    winning_rates[j][i] = rate2

        self.generate_figure(opt_names, winning_rates, successful_runs, average_winning, stats)

        #opt_names = ['Alg1', 'Alg2', 'Alg3', 'Alg4']
        #winning_rates = np.array(
        #[
        #    [0.6, 0.4, 0.5, 0.5],
        #    [0.4, 0.1, 0.5, 1.0],
        #    [0.1, 0.5, 0.2, 0.3],
        #    [0.5, 0.0, 0.3, 0.7]
        #])
        #stats = np.array(
        #[
        #    ['', '*', '', '*'],
        #    ['', '', '*', ''],
        #    ['', '*', '', ''],
        #    ['*', '*', '', '*']
        #])
        #successful_runs = [[200, 200], [200, 200], [132, 200], [120, 200]]
        #average_winning = [83.4, 73.4, 21.7, 1.4]

        #self.generate_figure(opt_names, winning_rates, successful_runs, average_winning, stats)


    def generate_figure(self, opt_names, winning_rates, successful_runs, average_winning, stats):

        import numpy as np
        import plotly.figure_factory as ff

        x_labels = [f'{alg} ({successful_runs[i][0]}/{successful_runs[i][1]})' for i, alg in enumerate(opt_names)]
        y_labels = [f'{alg} ({average_winning[-i-1]}%)' for i, alg in enumerate(reversed(opt_names))] 
        #x_labels = ['oi' for alg in opt_names]
        #y_labels = ['aoi' for alg in opt_names]
        

        print("TODO average winning!")
        #colorscale = [[0, 'blue'], [0.5, 'white'], [1, 'red']]
        colorscale = "RdBu"
        winning_rates = np.flip(winning_rates, 1)
        fig = ff.create_annotated_heatmap(z=winning_rates,x=x_labels, y=y_labels,annotation_text=stats, colorscale=colorscale, showscale=True)
        
        fig.update_xaxes(tickangle=-90, side='bottom', ticks='outside',  ticklen=10, gridwidth=5)#dtick=1)#linewidth=10, linecolor='#000')
        fig.update_yaxes(tickangle=-45, side='left', ticks='outside', ticklen=10,  gridwidth=5, gridcolor='rgb(0, 0, 0)')
        fig.update_traces(colorbar_ticks='outside', colorbar_outlinewidth=2, zmin=0, zmax=1, reversescale=True, selector=dict(type='heatmap'))
        #fig.update_layout(title_text=self.title, title_x=0.5)#, title_y=0.85)#[0,0.2,0.4,0.6,0.8,1.0]})
        
        #fig.update_traces(colorbar={'cmin':0, 'cmax':1}, selector=dict(type='heatmap'))
        #fig['data'][0]['colorbar']['nticks'] = 10
        #fig['data'][0]['colorbar']['showticklabels'] = True
        #fig['data'][0]['colorbar']['tickformatstops'] = [{'dtickrange': [0, 1], 'enabled':True}]
        
        fig.update_layout(width=500, height=500, font_family='Serif', font_size=12, margin_l=5, margin_t=5, margin_b=5, margin_r=5)
        #fig['data'][0]['colorbar']['tickvals'] = [0.5, 0.55, 0.99]
        #fig['data'][0]['colorbar']['ticktext'] = ['0', '0.2', '0.4', '0.6', '0.8', '1.0']
        #print(fig)
        
        #fig['data'][0]['colorbar']['ticktext'] = [0, 1]
        
        #fig.layout.height = 500
        #fig.layout.width = 400

        #fig.to_image(format="pdf", engine="kaleido")
        #fig.write_html("example.html")

        fig.write_image("images\heatmap_EXP2_2D.png")


