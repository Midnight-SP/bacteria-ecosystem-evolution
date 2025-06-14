import logging
from typing import Any, List, Tuple, Optional, cast
import numpy as np
from agents.cell import Cell
import random
from agents.names import get_species_name

class Bacteria(Cell):
    def __init__(self, x: int, y: int, genome=None, nn=None, input_size=28, output_size=5, grid_width=100, grid_height=100, population_genomes=None):
        # Przytnij genom do 22 genów (0-21) + wagi NN
        if genome is not None:
            # Jeśli genom jest za krótki, dopełnij zerami
            expected_len = 22 + input_size * output_size
            genome = np.array(list(genome[:22]) + [0]*(22-len(genome)) + list(genome[22:expected_len]), dtype=np.uint8)
        else:
            genome = np.random.randint(0, 255, 22 + input_size * output_size, dtype=np.uint8)
        super().__init__(x, y, genome, nn=nn, input_size=input_size, output_size=output_size)
        if population_genomes is not None and genome is not None:
            self.name = get_species_name(genome, population_genomes)
        else:
            self.name = "Bacteria_sp"
        self.initial_rotation = random.randint(0, 7)
        self.rotation = self.initial_rotation
        self.grid_width = grid_width
        self.grid_height = grid_height

        # Przypisz wszystkie geny do cech
        self.max_energy = int(genome[0]) * 10 if len(genome) > 0 else 64
        self.max_age = int(genome[1]) if len(genome) > 1 else 10
        self.mutation_rate = abs(int(genome[2])) if len(genome) > 2 else 16
        self.pheromone_strength = int(genome[3]) if len(genome) > 3 else 64
        self.max_health = int(genome[4]) if len(genome) > 4 else 100
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
        self.speed = int(genome[15]) if len(genome) > 15 else 1
        self.sight_range = int(genome[16]) if len(genome) > 16 else 1
        self.sight_angle = int(genome[17]) if len(genome) > 17 else 120
        self.dietary_preference = int(genome[18]) if len(genome) > 18 else 128
        self.pheromone_preference = int(genome[19]) if len(genome) > 19 else 0
        self.turning_speed = int(genome[20]) if len(genome) > 20 else 1
        self.aggression_level = int(genome[21]) if len(genome) > 21 else 0
        self.energy = self.max_energy

    def sense_environment(self, environment_state):
        # energia, wiek, zdrowie, chemotaksja, cooldown
        energy = self.energy / (self.max_energy if self.max_energy else 100)
        age = self.age / (self.max_age + 1)
        health = self.max_health / 100
        chemotaxis = 0
        for nx, ny in self.neighbors:
            cell = environment_state.get((nx, ny), None)
            chemotaxis += (cell.get('pheromone', 0) if cell else 0) * self.chemotaxis_strength * self.chemotaxis_sensitivity
        chemotaxis = chemotaxis / (len(self.neighbors) * 100)
        cooldown = self.reproduction_cooldown / (int(self.genome[13]) + 1) if len(self.genome) > 13 else 0

        # Widzenie: 8 kierunków, 2 cechy (czy jest komórka, czy jest feromon)
        sight = []
        for dx, dy in [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]:
            sx, sy = self.x + dx, self.y + dy
            cell = environment_state.get((sx, sy), None)
            sight.append(1 if cell and cell['type'] == 'cell' else 0)
            sight.append(1 if cell and cell.get('pheromone', 0) > 0 else 0)

        # Dodaj inne cechy genomu jako inputy (np. speed, sight_range, aggression_level, dietary_preference)
        extra_inputs = [
            self.speed / 10,
            self.sight_range / 10,
            self.sight_angle / 255,
            self.dietary_preference / 255,
            self.pheromone_preference / 255,
            self.turning_speed / 10,
            self.aggression_level / 10,
        ]

        return np.array([energy, age, health, chemotaxis, cooldown] + sight + extra_inputs)

    def act(self, environment_state):
        nn_input = self.sense_environment(environment_state)
        output = self.nn.forward(nn_input)
        action = np.argmax(output)
        # 0: rozsyłanie feromonów, 1: rozmnażanie, 2: ruch, 3: skręt, 4: jedzenie
        if action == 0:
            self.energy -= 2
            result = self.spread_pheromones()
        elif action == 1 and self.reproduction_cooldown == 0 and self.energy > self.division_threshold:
            self.energy -= 5
            self.reproduction_cooldown = int(self.genome[13]) if len(self.genome) > 13 else 0
            result = self.divide(environment_state)
        elif action == 2:
            self.move_forward()
            result = None
        elif action == 3:
            self.turn(direction=random.choice([-1, 1]))
            result = None
        elif action == 4:
            self.eat(environment_state)
            result = None
        else:
            self.energy -= 1
            result = None
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1
        self.photosynthesize()
        self.age += 1
        return result

    def divide(self, environment_state) -> list[tuple[Cell, Any]]:
        import numpy as np
        from agents.bacteria import Bacteria

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
            # Krzyżowanie genomów (połowa od siebie, połowa od partnera)
            mid = len(self.genome) // 2
            partner_mid = len(partner.genome) // 2
            genome_child1 = np.concatenate([self.genome[:mid], partner.genome[partner_mid:]])
            genome_child2 = np.concatenate([partner.genome[:partner_mid], self.genome[mid:]])

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

        child1 = Bacteria(chosen_positions[0][0], chosen_positions[0][1], genome=genome_child1, nn=nn_child1, input_size=self.input_size, output_size=self.output_size)
        child2 = Bacteria(chosen_positions[1][0], chosen_positions[1][1], genome=genome_child2, nn=nn_child2, input_size=self.input_size, output_size=self.output_size)
        return [(child1, chosen_positions[0]), (child2, chosen_positions[1])]

    def move_forward(self):
        directions = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
        dx, dy = directions[self.rotation % 8]
        for _ in range(self.speed):  # ruch zależny od genu speed
            new_x = self.x + dx
            new_y = self.y + dy
            if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height:
                self.x = new_x
                self.y = new_y
            else:
                break
        self.energy -= 2 * self.metabolism_rate  # koszt ruchu zależny od metabolizmu

    def turn(self, direction=None):
        if direction is None:
            direction = random.choice([-1, 1])
        self.rotation = (self.rotation + direction * self.turning_speed) % 8
        self.energy -= 1

    def eat(self, environment_state):
        # Sprawdza sąsiadów i preferencje żywieniowe, zjada jeśli spełnia warunki
        for nx, ny in self.neighbors:
            cell_info = environment_state.get((nx, ny))
            if cell_info and cell_info["type"] == "cell":
                obj = cell_info["object"]
                # Importy lokalne by uniknąć cyklicznych zależności
                from agents.protist import Protist
                from agents.bacteria import Bacteria
                # Preferencja: 0-85 tylko bakterie, 86-170 wszystko, 171-255 tylko protisty
                if isinstance(obj, Protist) and obj.is_alive:
                    if self.dietary_preference >= 171 or (86 <= self.dietary_preference <= 170):
                        obj.is_alive = False
                        self.energy += 20 + self.aggression_level
                        break
                elif isinstance(obj, Bacteria) and obj.is_alive and obj != self:
                    if self.dietary_preference <= 85 or (86 <= self.dietary_preference <= 170):
                        obj.is_alive = False
                        self.energy += 10 + self.aggression_level
                        break

    def __repr__(self):
        return f"Bacteria({self.x}, {self.y}, {self.name}, color={self.color})"

    def __eq__(self, other):
        if not isinstance(other, Bacteria):
            return False
        return super().__eq__(other) and self.name == other.name

    def __hash__(self):
        return hash((super().__hash__(), self.name))

    def photosynthesize(self):
        # Energia z fotosyntezy zależy od genów: gen[9] (wydajność fotosyntezy) i gen[8] (metabolizm)
        if self.energy < self.max_energy:
            energy_gain = max(1, self.photosynthesis_rate - self.metabolism_rate)
            self.energy += energy_gain
            if self.energy > self.max_energy:
                self.energy = self.max_energy