import logging
import random
from typing import Any

import numpy as np
from agents.cell import Cell
from agents.names import get_species_name

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class Protist(Cell):
    def __init__(self, x: int, y: int, genome=None, nn=None, input_size=5, output_size=2, population_genomes=None):
        expected_len = 15 + input_size * output_size
        if genome is not None:
            genome = np.array(list(genome[:expected_len]) + [0] * max(0, expected_len - len(genome)), dtype=np.uint8)
        else:
            genome = np.random.randint(0, 255, expected_len, dtype=np.uint8)
        super().__init__(x, y, genome, nn=nn, input_size=input_size, output_size=output_size)
        if population_genomes is not None and genome is not None:
            self.name = get_species_name(genome, population_genomes)
        else:
            self.name = "Protist_sp"
        self.population_genomes = population_genomes
        self.max_energy = int(genome[0]) * 10 if len(genome) > 0 else 64
        self.max_age = int(genome[1]) if len(genome) > 1 else 10
        self.mutation_rate = abs(int(genome[2])) if len(genome) > 2 else 16
        self.pheromone_strength = int(genome[3]) if len(genome) > 3 else 64
        self.max_health = int(genome[4]) if len(genome) > 4 else 100
        self.health = self.max_health  # Dodaj to!
        self.color = (
            int(genome[5]) if len(genome) > 5 else 0,
            int(genome[6]) if len(genome) > 6 else 0,
            int(genome[7]) if len(genome) > 7 else 0,
        )
        self.metabolism_rate = int(genome[8]) if len(genome) > 8 else 1
        self.photosynthesis_rate = int(genome[9]) if len(genome) > 9 else 1
        self.division_threshold = int(genome[10]) if len(genome) > 10 else 50
        self.chemotaxis_strength = int(genome[11]) if len(genome) > 11 else 1
        self.chemotaxis_sensitivity = int(genome[12]) if len(genome) > 12 else 1
        self.reproduction_cooldown = int(genome[13]) if len(genome) > 13 else 0
        self.lifespan_variation = int(genome[14]) if len(genome) > 14 else 0
        self.energy = self.max_energy

    def divide(self, environment_state) -> list[tuple[Cell, Any]]:
        # Szukaj partnera w sąsiedztwie
        partner = None
        for cell in self.neighboring_cells(environment_state):
            if self.can_reproduce_with(cell):
                partner = cell
                break

        available_neighbors = [pos for pos in self.neighbors]
        if len(available_neighbors) < 2:
            raise ValueError("Not enough free neighboring positions to place both children.")

        chosen_positions = random.sample(available_neighbors, 2)

        if partner:
            # Uniform crossover: każdy gen losowo od jednego z rodziców
            genome_child1 = np.array([random.choice([g1, g2]) for g1, g2 in zip(self.genome, partner.genome)], dtype=np.uint8)
            genome_child2 = np.array([random.choice([g1, g2]) for g1, g2 in zip(self.genome, partner.genome)], dtype=np.uint8)
            # Mutacja genomu dzieci
            genome_child1 = self.mutate(genome_child1)
            genome_child2 = self.mutate(genome_child2)
            # Krzyżowanie sieci neuronowych
            nn_child1 = self.nn.__class__.crossover(self.nn, partner.nn, mutation_rate=self.mutation_rate/255)
            nn_child2 = self.nn.__class__.crossover(partner.nn, self.nn, mutation_rate=partner.mutation_rate/255)
        else:
            # Klonowanie z mutacją
            mid = len(self.genome) // 2
            genome_child1 = np.concatenate([self.genome[:mid], self.mutate(self.genome[mid:])])
            genome_child2 = np.concatenate([self.genome[mid:], self.mutate(self.genome[:mid])])
            nn_child1 = self.nn.__class__.crossover(self.nn, self.nn, mutation_rate=self.mutation_rate/255)
            nn_child2 = self.nn.__class__.crossover(self.nn, self.nn, mutation_rate=self.mutation_rate/255)

        child1 = Protist(chosen_positions[0][0], chosen_positions[0][1], genome=genome_child1, nn=nn_child1, input_size=self.input_size, output_size=self.output_size, population_genomes=self.population_genomes)
        child2 = Protist(chosen_positions[1][0], chosen_positions[1][1], genome=genome_child2, nn=nn_child2, input_size=self.input_size, output_size=self.output_size, population_genomes=self.population_genomes)
        # Ustaw cooldown dzieciom, żeby nie dzieliły się natychmiast
        child1.reproduction_cooldown = max(5, child1.reproduction_cooldown)
        child2.reproduction_cooldown = max(5, child2.reproduction_cooldown)
        # Odejmij koszt podziału od energii dzieci
        child1.energy = max(0, child1.energy - 5)
        child2.energy = max(0, child2.energy - 5)
        return [(child1, chosen_positions[0]), (child2, chosen_positions[1])]

    def __repr__(self):
        return f"Protist({self.x}, {self.y}, {self.name}, color={self.color})"

    def __eq__(self, other):
        if not isinstance(other, Protist):
            return False
        return super().__eq__(other) and self.name == other.name

    def __hash__(self):
        return hash((super().__hash__(), self.name))

    def sense_environment(self, environment_state):
        # energia (gen[0]), wiek, zdrowie (gen[4]), chemotaksja (suma feromonów wokół), cooldown
        energy = self.energy / (self.max_energy if self.max_energy else 100)
        age = self.age / (self.max_age + 1)
        health = self.max_health / 100
        # Suma feromonów na sąsiadujących polach, z uwzględnieniem chemotaxis_strength i sensitivity
        chemotaxis = 0
        for nx, ny in self.neighbors:
            cell = environment_state.get((nx, ny), None)
            pher = cell.get('pheromone', 0) if cell else 0
            chemotaxis += pher * self.chemotaxis_strength * self.chemotaxis_sensitivity
        chemotaxis = chemotaxis / (len(self.neighbors) * 100)
        cooldown = self.reproduction_cooldown / (int(self.genome[13]) + 1) if len(self.genome) > 13 else 0
        return np.array([energy, age, health, chemotaxis, cooldown])

    def maintenance_energy_cost(self):
        cost = 1
        cost += self.metabolism_rate * 0.5
        cost += self.chemotaxis_strength * 0.2
        cost += self.chemotaxis_sensitivity * 0.2
        cost += self.pheromone_strength / 100
        cost += self.max_health / 200
        cost -= self.reproduction_cooldown
        cost -= self.photosynthesis_rate
        # Dodaj inne geny jeśli chcesz
        return cost

    def act(self, environment_state):
        nn_input = self.sense_environment(environment_state)
        output = self.nn.forward(nn_input)
        action = np.argmax(output)
        logging.debug(f"Protist at ({self.x}, {self.y}) action: {action}, energy: {self.energy}, age: {self.age}")

        result = None
        if action == 0:
            result = self.spread_pheromones()
        elif action == 1 and self.reproduction_cooldown == 0 and self.energy > self.division_threshold:
            self.energy -= 5
            self.reproduction_cooldown = int(self.genome[13]) if len(self.genome) > 13 else 0
            result = self.divide(environment_state)

        # Always update state!
        self.energy -= self.maintenance_energy_cost()
        self.energy = max(0, self.energy)
        self.vitals()
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1
        self.age += 1

        return result

    def spread_pheromones(self):
        # Rozsyłanie feromonów zależne od self.pheromone_strength
        pheromone_actions = []
        pheromone_actions.append(((self.x, self.y), self.pheromone_strength))
        neighbor_strength = self.pheromone_strength // 2
        for nx, ny in self.neighbors:
            pheromone_actions.append(((nx, ny), neighbor_strength))
        return pheromone_actions