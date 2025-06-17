from abc import ABC, abstractmethod
import random
from typing import Any, Optional, Tuple

class Agent(ABC):
    _id_counter = 0

    def __init__(self, genome, neural_network, position, parent_ids=None, founder_id=None):
        self.id = Agent._id_counter
        Agent._id_counter += 1
        self.genome = genome
        self.neural_network = neural_network
        self.position = position
        self.parent_ids = parent_ids if parent_ids is not None else []
        self.founder_id = founder_id if founder_id is not None else self.id
        self.energy = genome.initial_energy if hasattr(genome, "initial_energy") else 0
        self.age = 0
        self.is_alive = True
        # Kolor na podstawie genomu
        self.color = (
            int(genome.genes[5] if len(genome.genes) > 5 else 128),
            int(genome.genes[6] if len(genome.genes) > 6 else 128),
            int(genome.genes[7] if len(genome.genes) > 7 else 128),
        )
        self.infections = []

    @property
    def agent_type(self):
        return self.__class__.__name__

    @abstractmethod
    def act(self, environment) -> Optional[Tuple[int, int]]:
        pass

    def mutate(self):
        self.genome.mutate()
        self.neural_network.mutate()

    def count_occupied_neighbors(self, environment):
        x, y = self.position
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        occupied = 0
        for dx, dy in directions:
            nx, ny = (x + dx) % environment.width, (y + dy) % environment.height
            if environment.grid[ny][nx] is not None:
                occupied += 1
        return occupied

    def can_reproduce(self, environment=None):
        # bardziej wymagające rozmnażanie dla bakterii
        if self.agent_type == "Algae":
            # jeszcze bardziej wymagające rozmnażanie
            min_energy, min_age, repro_chance = 100, 15, 1
        elif self.agent_type == "Fungi":
            min_energy, min_age, repro_chance = 50, 5, 1
        else:  # Algae, Protozoa
            min_energy, min_age, repro_chance = 300, 30, 0.10
        if environment is not None:
            # zabroń rozmnażania, jeśli co najmniej połowa sąsiadów (4 kierunki) jest zajęta
            if self.count_occupied_neighbors(environment) >= 2:
                return False
        return self.energy > min_energy and self.age > min_age and random.random() < repro_chance

    def reproduce(self, partner=None):
        from src.core.evolution import uniform_crossover, mutate_genome
        from src.core.neural_network import NeuralNetwork
        import numpy as np

        if partner:
            child_genes = uniform_crossover(self.genome.genes, partner.genome.genes)
            parent_ids = [self.id, partner.id]
            child_weights = (self.neural_network.weights + partner.neural_network.weights) / 2
        else:
            child_genes = self.genome.genes.copy()
            parent_ids = [self.id]
            child_weights = self.neural_network.weights.copy()
        child_genes = mutate_genome(child_genes, rate=0.05)
        child_genome = type(self.genome)(child_genes)
        child_nn = NeuralNetwork(self.neural_network.input_size, self.neural_network.output_size, weights=child_weights)
        child_nn.mutate(rate=0.05)
        return type(self)(child_genome, child_nn, self.position, parent_ids=parent_ids, founder_id=self.founder_id)
