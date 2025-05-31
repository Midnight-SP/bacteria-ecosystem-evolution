import random
from typing import Any

import numpy as np
from agents.cell import Cell


class Protist(Cell):
    def __init__(self, x: int, y: int, name: str, genome=None, nn=None, input_size=18, output_size=4):
        super().__init__(x, y, genome, nn=nn, input_size=input_size, output_size=output_size)
        self.name = name

    def divide(self, neighbors) -> list[tuple[Cell, Any]]:
        available_neighbors = [pos for pos in neighbors]
        if len(available_neighbors) < 2:
            raise ValueError("Not enough free neighboring positions to place both children.")

        chosen_positions = random.sample(available_neighbors, 2)
        mid = len(self.genome) // 2
        genome_child1 = np.concatenate([self.genome[:mid], self.mutate(self.genome[mid:])])
        genome_child2 = np.concatenate([self.genome[mid:], self.mutate(self.genome[:mid])])

        nn_child1 = self.nn.__class__.crossover(self.nn, self.nn, mutation_rate=self.mutation_rate/128)
        nn_child2 = self.nn.__class__.crossover(self.nn, self.nn, mutation_rate=self.mutation_rate/128)

        # Przekazujemy self.name do potomków (lub możesz dodać np. sufiks "_c")
        child1 = Protist(chosen_positions[0][0], chosen_positions[0][1], name=self.name, genome=genome_child1, nn=nn_child1, input_size=self.input_size, output_size=self.output_size)
        child2 = Protist(chosen_positions[1][0], chosen_positions[1][1], name=self.name, genome=genome_child2, nn=nn_child2, input_size=self.input_size, output_size=self.output_size)
        return [(child1, chosen_positions[0]), (child2, chosen_positions[1])]

    def __repr__(self):
        return f"Protist({self.x}, {self.y}, {self.name})"

    def __eq__(self, other):
        if not isinstance(other, Protist):
            return False
        return super().__eq__(other) and self.name == other.name

    def __hash__(self):
        return hash((super().__hash__(), self.name))