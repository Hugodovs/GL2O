def load_mpidata(name, folder_path):

    import json
    import glob
    import numpy as np

    # Internals:
    nb_files = len(glob.glob1(f'{folder_path}', f'{name}*'))
    with open(f'{folder_path}/info.json', 'rb') as f:
        info = json.load(f)
    nb_generations = info['nb_generations']

    # Get data:
    if name in ['costs', 'genotypes']:
        data = [[] for i in range(nb_files)]
        for i in range(nb_files):
            generation_data = []
            with open(f'{folder_path}/{name}_w{i}.npy', 'rb') as f:
                for g in range(nb_generations):
                    generation_data.append(np.load(f, allow_pickle=True).tolist())
            data[i] = generation_data
        data = np.array(data, object)
        data = np.transpose(data, (1, 0, 2))
        data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
        return data
    elif name in ['initial_guess']:
        with open(f'{folder_path}/initial_guess.npy', 'rb') as f:
            return np.load(f, allow_pickle=True)
    else:
        print(f'load_mpidata failed, name "{name}" not found!')
        exit()

def get_best_phenotype(folder_path):

    import json
    import numpy as np
    #from DNE4py.optimizers.deepga.member import Member as DeepGAMember
    from .deepga_table.member import Member as DeepGATableMember

    # Read Input:
    costs = load_mpidata("costs", f"{folder_path}")
    genotypes = load_mpidata("genotypes", f"{folder_path}")
    initial_guess = load_mpidata("initial_guess", f"{folder_path}")

    # Select Best Idx:
    best_idx = np.unravel_index(costs.argmin(), costs.shape)

    # Load Member:
    with open(f'{folder_path}/info.json', 'rb') as f:
        info = json.load(f)
    _id = info['id']

    if _id == 'TruncatedRealMutatorGA_Table':
        global_seed = info['global_seed']
        table_rng = np.random.RandomState(global_seed)
        random_table = table_rng.randn(20000000)
        return DeepGATableMember(initial_guess, genotypes[best_idx], random_table).phenotype

