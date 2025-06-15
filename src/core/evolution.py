import numpy as np

def uniform_crossover(parent1_genome, parent2_genome):
    """
    Uniform crossover: każdy gen losowo od jednego z rodziców.
    """
    return np.array([np.random.choice([g1, g2]) for g1, g2 in zip(parent1_genome, parent2_genome)], dtype=np.uint8)

def mutate_genome(genome, rate=0.01):
    """
    Mutuje genom z podanym prawdopodobieństwem.
    Z małą szansą (np. 1%) mutacja jest 10x silniejsza.
    """
    mutated = genome.copy()
    # 1% szans na silną mutację
    if np.random.rand() < 0.1:
        rate = rate * 10
    for i in range(len(mutated)):
        if np.random.rand() < rate:
            mutated[i] = np.random.randint(0, 256)
    return mutated