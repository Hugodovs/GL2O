from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
torch.set_default_dtype(torch.float64)

from .optimizer import Optimizer
from ..metaoptimizers.utils import get_best_phenotype

class LIOH(Optimizer):

    def __init__(self, config):
        super().__init__(config)

        # Create model:
        torch.manual_seed(self.seed)

        torch_layers = []
        for block_config in self.blocks:

            # Instantiate:
            layer = instantiate_layer(block_config)

            # Add Activation:
            activation = instantiate_activation(block_config)

            # Initialize:
            initialize_layer(layer, block_config)

            # Append:
            torch_layers.append(layer)
            if activation is not None:
                torch_layers.append(activation)

        self.lstm = torch_layers[0]
        self.mlp = nn.Sequential(*torch_layers[1:])

        #self.model = nn.ModuleDict(OrderedDict({
        #    'lstm': self.lstm, 'mlp': self.mlp}))
        
        self.model = nn.ModuleDict(OrderedDict({
            'lstm': self.lstm, 'mlp': self.mlp}))
        
        #print(self.model)
        
        #def count_parameters(model):
        #    total_params = 0
        #    for name, parameter in model.named_parameters():
        #        if not parameter.requires_grad: 
        #            continue
        #        param = parameter.numel()
        #        print(name, param)
        #        total_params+=param
        #    print(f"Total Trainable Params: {total_params}")
        #    return total_params

        #count_parameters(self.model)
        
        self.first_input = True

        if (hasattr(self, "load_path")):
            best_phenotype = get_best_phenotype(self.load_path)
            self.set_params(best_phenotype)
        
    def get_params(self):
        parameters = np.concatenate([p.detach().numpy().ravel() for p in self.model.parameters()])
        return parameters

    def set_params(self, new_weights):
        last_slice = 0
        for n, p in self.model.named_parameters():
            size_layer_parameters = np.prod(np.array(p.data.size()))
            new_parameters = new_weights[last_slice:last_slice + size_layer_parameters].reshape(p.data.shape)
            last_slice += size_layer_parameters
            p.data = torch.from_numpy(new_parameters).detach()

    def reset(self, seed):
        lstm_num_layers = self.blocks[0]['args']['num_layers']
        _hidden_size = self.blocks[0]['args']['hidden_size']

        gen = torch.Generator()
        gen = gen.manual_seed(624234)

        # Reset hidden and cell states:
        self._hidden = torch.rand(lstm_num_layers, self.population_size*self.dim, _hidden_size, generator=gen)
        self._cell = torch.rand(lstm_num_layers, self.population_size*self.dim, _hidden_size, generator=gen)

    def ask(self):
        with torch.no_grad():

            if (self.first_input is True):
                self.params = np.zeros((self.population_size, self.dim))
                self.ranks = np.zeros(self.population_size)

            batch = np.empty((1, self.population_size*self.dim, 2))
            for i in range(self.population_size):
                for j in range(self.dim):
                    param = self.params[i][j]
                    rank = self.ranks[i]
                    batch[0][i*self.dim + j] = np.array((param, rank))
            batch = torch.from_numpy(batch)

            _output, (self._hidden, self._cell) = self.model['lstm'](batch, (self._hidden, self._cell))
            _output = _output[0]
            
            final_output = self.model['mlp'](_output).numpy()
            final_output = final_output.reshape((self.population_size, self.dim))

            return final_output

    def tell(self, candidates, evaluations):

        self.idxs = np.argsort(evaluations)
        self.ranks = np.linspace(0, 1, self.population_size)
        self.ranks = self.ranks[self.idxs]
        self.params = candidates


class BayesLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))

        self.bias = bias
        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features))
        else:
            print("BayesLinear: Bias should be True")
            exit()

    def forward(self, input):

        weight = self.weight_mu

        bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)

        return nn.functional.linear(input, weight, bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)

def instantiate_layer(block_config):

    # Read Type:
    curr_type = block_config['type']

    # Build layer:
    if curr_type == 'linear':
        layer = nn.Linear(**block_config['args'])
    elif curr_type == 'bayes_linear':
        layer = BayesLinear(**block_config['args'])
    elif curr_type == 'lstm':
        layer = nn.LSTM(**block_config['args'])#, batch_first=True)
    else:
        print("instantiate_layer problem!")
        exit()
    return layer

def instantiate_activation(block_config):

    if block_config['type'] == 'lstm':
        return None

    # Read Type:
    curr_type = block_config['activation']['type']

    # Build activation:
    if curr_type == 'tanh':
        activation = nn.Tanh()
    elif curr_type == 'softplus':
        activation = nn.Softplus(**block_config['activation']['args'])
    elif curr_type == 'sigmoid':
        activation = nn.Sigmoid()
    elif curr_type == 'relu':
        activation = nn.ReLU()
    elif curr_type == 'leakyrelu':
        activation = nn.LeakyReLU(**block_config['activation']['args'])
    elif curr_type == None:
        activation = None
    else:
        print("instantiate_activation problem!")
        exit()
    return activation


def initialize_layer(layer, block_config):

    def calculate_gain(block_config):

        if block_config['type'] == 'lstm':
            activation_type = 'tanh'
        else:
            activation_type = block_config['activation']['type']

        if activation_type in ['sigmoid', 'tanh', 'relu']:
            return init.calculate_gain(activation_type)
        elif activation_type == 'leakyrelu':
            return init.calculate_gain('leaky_relu', block_config['activation']['args']['negative_slope'])
        else:
            return 1.0

    # Read Type:
    curr_type = block_config['initialization']['type']

    # Initialize layer::
    if curr_type == 'default':
        return

    for n, param in layer.named_parameters():
        if 'bias' in n:
            init.zeros_(param.data)
        if 'weight' or 'bias' in n:
            if curr_type == 'uniform':
                init.uniform_(param.data, **block_config['initialization']['args'])
            elif curr_type == 'normal':
                init.normal_(param.data, **block_config['initialization']['args'])
            elif curr_type == 'constant':
                init.constant_(param.data, **block_config['initialization']['args'])
            elif 'xavier' in curr_type:
                gain = calculate_gain(block_config)
                if curr_type == 'xavier_uniform':
                    init.xavier_uniform_(param.data, gain)
                elif curr_type == 'xavier_normal':
                    init.xavier_normal_(param.data, gain)
            elif 'kaiming' in curr_type:

                if block_config['activation']['type'] == 'leakyrelu':
                    args = {'a': block_config['activation']['args']['negative_slope'], 'nonlinearity': 'leaky_relu'}
                elif block_config['activation']['type'] == 'relu':
                    args = {'nonlinearity': 'relu'}

                if curr_type == 'kaiming_uniform':
                    init.kaiming_uniform_(param.data, **args)
                elif curr_type == 'kaiming_normal':
                    init.kaiming_normal_(param.data, **args)
            else:
                print("initialize_layer problem!")
                exit()


