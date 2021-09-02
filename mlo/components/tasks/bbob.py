import numpy as np

from .task import Task
from ..infoset import InfoSet

class BBOBTask(Task):
    r''' 
        http://cma.gforge.inria.fr/apidocs-pycma/cma.bbobbenchmarks.html
    '''

    def __init__(self, config):
        super().__init__(config)

    def build(self):
        
        assert(self.f_id)
        assert(self.i_id)
        assert(self.dim)
        assert(self.targets_amount)
        assert(self.targets_precision)
        #assert(self.save_candidates)
        #assert(self.simplified_loss)
        
        from cma import bbobbenchmarks as bn
        self.f = eval(f'bn.F{self.f_id}({self.i_id})')        
        self.id = f'|F{self.f_id}({self.i_id}) D:{self.dim}|'
        return self

    def evaluate(self, candidate):

        # Evaluate
        loss = self.f(candidate)
        self.nb_evals += 1

        # Update Info
        if self.save_candidates:
            if 'candidates' not in self._data.keys():
                self._data['candidates'] = []
            self._data['candidates'].append(candidate)

        if self.simplified_loss:
            if 'losses' not in self._data.keys():
                self._data['losses'] = [(self.nb_evals, loss)]
            else:
                if loss < self._data['losses'][-1][1]:
                    self._data['losses'].append((self.nb_evals, loss))
        else:
            if 'losses' not in self._data.keys():
                self._data['losses'] = []
            self._data['losses'].append(loss)

        return loss

    def close(self):
        targets = self.get_targets()
        self._targets['targets'] = targets 
        return self._data, self._targets

    def __str__(self):
        return f'TaskBBOB => |F{self.f_id}({self.i_id}) D:{self.dim}|'

    def get_targets(self):

        from copy import deepcopy
        fopt = self.f.fopt
        targets =  fopt + 10 ** np.linspace(2, self.targets_precision, self.targets_amount)
        return targets
        
        if self._targets is None:

            try:
                fopt = self.f.fopt
                temp_f = deepcopy(self.f)
                self._targets = fopt + 10 ** np.linspace(2, self.targets_precision, self.targets_amount)
            except:

                import json
                path_prefix = os.path.join(os.path.dirname(__file__), "hpo_data/profet_data/targets/meta_" + str(self.f_id) + "_noiseless_targets.json")

                #print(path_prefix)
                with open(path_prefix) as f:
                    targets = np.array(json.load(f))
                targets = targets[self.i_id]

                #print(targets)
                traj = []
                curr = 1
                for t in targets:
                    if t < curr:
                        curr = t
                    traj.append(curr)
                traj = np.array(traj)
                self._targets = traj

        return self._targets