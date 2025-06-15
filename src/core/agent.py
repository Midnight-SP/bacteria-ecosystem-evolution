from abc import ABC, abstractmethod
import random
from typing import Any, Optional, Tuple

class Agent(ABC):
    _id_counter = 0  # globalny licznik

    def __init__(self, genome, neural_network, position, parent_ids=None):
        self.genome = genome
        self.nn = neural_network
        self.position = position
        self.energy = genome.initial_energy
        self.age = 0
        self.is_alive = True
        self.id = Agent._id_counter
        Agent._id_counter += 1
        self.parent_ids = parent_ids if parent_ids is not None else []
        # Kolor na podstawie genomu
        self.color = (
            int(genome.genes[5] if len(genome.genes) > 5 else 128),
            int(genome.genes[6] if len(genome.genes) > 6 else 128),
            int(genome.genes[7] if len(genome.genes) > 7 else 128),
        )
        self.infections = []  # lista aktywnych infekcji (np. obiektów Virus lub ich genomów)

    @abstractmethod
    def act(self, environment) -> Optional[Tuple[int, int]]:
        pass

    def mutate(self):
        self.genome.mutate()
        self.nn.mutate()

    def can_reproduce(self):
        return self.energy > 300 and self.age > 20 and random.random() < 0.1

    def reproduce(self, partner=None):
        from src.core.evolution import uniform_crossover, mutate_genome
        from src.core.neural_network import NeuralNetwork

        if partner:
            child_genes = uniform_crossover(self.genome.genes, partner.genome.genes)
            parent_ids = [self.id, partner.id]
        else:
            child_genes = self.genome.genes.copy()
            parent_ids = [self.id]
        child_genes = mutate_genome(child_genes, rate=0.15)
        child_genome = type(self.genome)(child_genes)
        child_nn = NeuralNetwork(self.nn.input_size, self.nn.output_size)
        return type(self)(child_genome, child_nn, self.position, parent_ids=parent_ids)