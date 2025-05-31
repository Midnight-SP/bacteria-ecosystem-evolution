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
        self.genome = genome if genome is not None else np.random.randint(-128, 128, 255, dtype=np.int8)
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
            self.nn = NeuralNetwork(input_size=self.input_size, output_size=self.output_size, genome=nn_genome, mutation_rate=self.mutation_rate/128)

    def __eq__(self, other):
        if not isinstance(other, Cell):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Cell({self.x}, {self.y})"

    def act(self, environment_state):
        # Odczytaj sensory z otoczenia
        nn_input = self.sense_environment(environment_state)
        action = np.argmax(self.nn.forward(nn_input))
        # Każda akcja zużywa energię
        if action == 0:
            self.energy -= 5
            return self.divide(self.neighbors)
        elif action == 1:
            self.energy -= 2
            self.spread_pheromones()
        else:
            self.energy -= 1  # brak akcji

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

    def divide(self, neighbors):
        available_neighbors = [pos for pos in neighbors if not (len(pos) > 2 and pos[2])]
        if len(available_neighbors) < 2:
            raise ValueError("Not enough free neighboring positions to place both children.")

        chosen_positions = random.sample(available_neighbors, 2)
        mid = len(self.genome) // 2
        genome_child1 = np.concatenate([self.genome[:mid], self.mutate(self.genome[mid:])])
        genome_child2 = np.concatenate([self.genome[mid:], self.mutate(self.genome[:mid])])

        # Sieć potomków: crossover + mutacja
        nn_child1 = NeuralNetwork.crossover(self.nn, self.nn, mutation_rate=self.mutation_rate/128)
        nn_child2 = NeuralNetwork.crossover(self.nn, self.nn, mutation_rate=self.mutation_rate/128)

        child1 = self.__class__(chosen_positions[0][0], chosen_positions[0][1], genome=genome_child1, nn=nn_child1, input_size=self.input_size, output_size=self.output_size)
        child2 = self.__class__(chosen_positions[1][0], chosen_positions[1][1], genome=genome_child2, nn=nn_child2, input_size=self.input_size, output_size=self.output_size)
        return [(child1, chosen_positions[0]), (child2, chosen_positions[1])]

    def mutate(self, genome_part):
        mutated = []
        for gene in genome_part:
            if random.random() < self.mutation_rate/128:
                mutated.append(gene + random.uniform(-0.1, 0.1))
            else:
                mutated.append(gene)
        return np.array(mutated, dtype=np.int8)

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