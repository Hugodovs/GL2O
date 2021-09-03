import numpy as np

costs = np.load("data/exp1/train/LIOH_2D_20I_5R/costs_w0.npy")
genotypes = np.load("data/exp1/train/LIOH_2D_20I_5R/genotypes_w0.npy", allow_pickle=True)
initial_guess = np.load("data/exp1/train/LIOH_2D_20I_5R/initial_guess.npy")

print()
print(costs.shape)
print(genotypes.shape)
print(initial_guess.shape)
print()

len_costs = 0
for i, cost in enumerate(costs):
    if i == 0: print(cost)
    len_costs += 1
print(len_costs)

len_genotype = 0
for i, genotype in enumerate(genotypes):
    if i == 0: print(genotype)
    len_genotype += 1
print(len_genotype)