import random
import numpy as np
from neural_network.neural_network import NeuralNetwork

class Cell:
    def __init__(self, x: int, y: int, genome=None, nn=None, input_size=18, output_size=4):
        self.x = x
        self.y = y
        self.is_alive = True
        self.neighbors = [
            [self.x + dx, self.y + dy]
            for dx, dy in [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (-1, -1), (1, -1), (-1, 1)
            ]
        ]
        self.genome = genome if genome is not None else np.random.randint(-128, 128, 255, dtype=np.uint8)
        self.initial_energy = int(self.genome[0]) if self.genome is not None else 64
        self.energy = self.initial_energy
        self.age = 0
        self.max_age = int(self.genome[1]) if len(self.genome) > 1 else 10
        self.mutation_rate = abs(int(self.genome[2])) if len(self.genome) > 2 else 16
        self.pheromone_strength = int(self.genome[3]) if len(self.genome) > 3 else 64
        self.input_size = input_size
        self.output_size = output_size

        # Inicjalizacja sieci neuronowej: z genomu lub nowa
        if nn is not None:
            self.nn = nn
        else:
            # Jeśli genom jest dłuższy niż 255, użyj końcówki jako genom NN
            nn_genome = None
            if len(self.genome) >= self.input_size * self.output_size:
                nn_genome = self.genome[-(self.input_size * self.output_size):]
            self.nn = NeuralNetwork(input_size=self.input_size, output_size=self.output_size, genome=nn_genome, mutation_rate=self.mutation_rate/255)

    def __eq__(self, other):
        if not isinstance(other, Cell):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Cell({self.x}, {self.y})"

    def sense_environment(self, environment_state):
        inputs = []
        for nx, ny in self.neighbors:
            cell = environment_state.get((nx, ny), None)
            # Przykład: 1 jeśli jest komórka, 0 jeśli pusto
            inputs.append(1 if cell and cell['type'] == 'cell' else 0)
            # Przykład: 1 jeśli jest feromon, 0 jeśli nie
            inputs.append(1 if cell and cell.get('pheromone', 0) > 0 else 0)
        # Możesz dodać własne cechy, np. energia, wiek
        inputs.append(self.energy / 100)  # normalizacja
        inputs.append(self.age / (self.max_age + 1))
        return np.array(inputs)

    def mutate(self, genome_part):
        mutated = []
        for gene in genome_part:
            if random.random() < self.mutation_rate/255:
                new_val = int(gene) + int(random.uniform(-5, 5))
                new_val = max(0, min(255, new_val))
                mutated.append(new_val)
            else:
                mutated.append(int(gene))
        return np.array(mutated, dtype=np.uint8)

    def vitals(self):
        if self.energy <= 0:
            self.is_alive = False
        return self.is_alive

    def spread_pheromones(self):
        """
        Spread pheromones to neighboring positions.
        Each neighbor receives pheromone_strength // 2, and the cell's own position receives pheromone_strength.
        Returns a list of (position, pheromone_amount) tuples.
        """
        pheromone_actions = []
        # Spread to self
        pheromone_actions.append(((self.x, self.y), self.pheromone_strength))
        # Spread to neighbors
        neighbor_strength = self.pheromone_strength // 2
        for nx, ny in self.neighbors:
            pheromone_actions.append(((nx, ny), neighbor_strength))
        return pheromone_actions

    def can_reproduce_with(self, other, input_size=None, output_size=None, tolerance=16, similarity=0.9):
        """
        Sprawdza, czy dwie komórki mogą się rozmnażać na podstawie genomu (bez wag NN).
        """
        if input_size is None:
            input_size = self.input_size
        if output_size is None:
            output_size = self.output_size

        nn_len = input_size * output_size
        # Geny bez wag NN
        g1 = self.genome[:-nn_len] if nn_len > 0 else self.genome
        g2 = other.genome[:-nn_len] if nn_len > 0 else other.genome

        if len(g1) != len(g2):
            return False

        similar = np.sum(np.abs(g1 - g2) <= tolerance)
        return (similar / len(g1)) >= similarity
    
    def neighboring_cells(self, environment_state):
        """
        Zwraca listę sąsiednich komórek z otoczenia.
        """
        neighbors = []
        for nx, ny in self.neighbors:
            cell_info = environment_state.get((nx, ny), None)
            if cell_info and cell_info.get('type') == 'cell':
                neighbors.append(cell_info['object'])
        return neighbors