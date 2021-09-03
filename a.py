import cProfile
import numpy as np

def compute_rts3(run, targets):
    run = np.array(run)
    evals = run[:, 0]
    losses = run[:, 1]

    rts = []
    for target in targets:
        idx = None
        for i, loss in enumerate(losses):
            if loss >= target:
                idx = i
                break
        if idx == None:
            rt = np.inf
        else:
            rt = evals[idx]
        rts.append(rt)
    return rts

def compute_rts2(run, targets):
    last_idx, idx = 0, 0
    rts = np.zeros(len(targets))
    for i, (f_val, y_value) in enumerate(run):
        last_idx = idx
        idx = np.searchsorted(targets, y_value)
        if i == 0:
            rts[:idx] = f_val
        else:
            rts[last_idx:idx] = f_val

def compute_rts1(run, targets):
    run = np.array(run)
    evals = run[:, 0]
    losses = run[:, 1]

    rts = []
    for target in targets:
        rt = np.where(losses >= target)[0]
        if len(rt) == 0:
            rt = np.inf
        else: 
            rt = evals[rt[0]]
        rts.append(rt)
    return rts

run = [(1, 947.750467218043), (3, 950.3200015392013), (4, 953.2619324454406)]
targets = [900.,920.56717653,936.90426555,949.88127664,960.18928294,999.99841511]
result = [1.0, 1.0, 1.0, 3.0, np.inf, np.inf]

with cProfile.Profile() as pr:
    for i in range(1000000):
        rts = compute_rts3(run, targets)
pr.dump_stats('rts3.prof')







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