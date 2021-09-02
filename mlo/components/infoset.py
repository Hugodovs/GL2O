import pickle
from collections import defaultdict

class InfoSet:

    def __init__(self, config):

        self.data = defaultdict(self.bind_to_data)

        self.output_path = config['output_path']
        if config['load_paths'] != None:
            for alg in config['load_paths']:
                for opt_id, path in alg.items():
                    data = self.load(path).data
                    self.set_opt_data(opt_id, data[opt_id])

    def bind_to_data(self):
        return defaultdict(dict)

    def __repr__(self):
        return f'InfoSet: {self.data.keys()}'

    def append_task_data(self, opt_id, task_id, data):
        for key, value in data.items():
            if key in self.data[opt_id][task_id].keys():
                self.data[opt_id][task_id][key].append(value)
            else:        
                self.data[opt_id][task_id][key] = [value]

    def set_task_data(self, opt_id, task_id, data):
        for key, value in data.items():
            self.data[opt_id][task_id][key] = value

    def set_opt_data(self, opt_id, data):
        self.data[opt_id] = data
        
    def save(self):
        with open(self.output_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        return pickle.load(open(path, "rb"))      