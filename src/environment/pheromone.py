import numpy as np

class PheromoneField:
    """
    Przechowuje wiele rodzajów feromonów (np. 'food', 'danger', 'predator').
    """
    def __init__(self, width, height, pheromone_types=None):
        if pheromone_types is None:
            pheromone_types = ['default']
        self.width = width
        self.height = height
        self.types = pheromone_types
        self.grids = {ptype: np.zeros((height, width), dtype=np.float32) for ptype in pheromone_types}

    def add(self, x, y, amount, ptype='default'):
        self.grids[ptype][y % self.height, x % self.width] += amount

    def evaporate(self, rate=0.05):
        for grid in self.grids.values():
            grid *= (1 - rate)
            grid[grid < 0] = 0

    def get(self, x, y, ptype='default'):
        return self.grids[ptype][y % self.height, x % self.width]