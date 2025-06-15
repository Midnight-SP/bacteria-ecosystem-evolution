import logging
from typing import Any, List, Tuple, Optional, cast
import numpy as np
from agents.cell import Cell
import random
from agents.names import get_species_name

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class Bacteria(Cell):
    def __init__(self, x: int, y: int, genome=None, nn=None, input_size=30, output_size=7, grid_width=100, grid_height=100, population_genomes=None):
        expected_len = 23 + input_size * output_size
        if genome is not None:
            genome = np.array(list(genome[:expected_len]) + [0] * max(0, expected_len - len(genome)), dtype=np.uint8)
        else:
            genome = np.random.randint(0, 255, expected_len, dtype=np.uint8)
        super().__init__(x, y, genome, nn=nn, input_size=input_size, output_size=output_size)
        if population_genomes is not None and genome is not None:
            self.name = get_species_name(genome, population_genomes)
        else:
            self.name = "Bacteria_sp"
        self.initial_rotation = random.randint(0, 7)
        self.rotation = self.initial_rotation
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.population_genomes = population_genomes

        # Przypisz wszystkie geny do cech
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
        self.speed = int(genome[15]) if len(genome) > 15 else 1
        self.sight_range = int(genome[16]) if len(genome) > 16 else 1
        self.sight_angle = int(genome[17]) if len(genome) > 17 else 120
        self.dietary_preference = int(genome[18]) if len(genome) > 18 else 128
        self.pheromone_preference = int(genome[19]) if len(genome) > 19 else 0
        self.turning_speed = int(genome[20]) if len(genome) > 20 else 1
        self.aggression_level = int(genome[21]) if len(genome) > 21 else 0
        self.eating_strength = int(genome[22]) if len(genome) > 22 else 10
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
        can_eat_ahead = 0
        can_reproduce_ahead = 0
        for idx, (dx, dy) in enumerate([(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]):
            sx, sy = self.x + dx, self.y + dy
            cell = environment_state.get((sx, sy), None)
            is_cell = 1 if cell and cell['type'] == 'cell' else 0
            is_pheromone = 1 if cell and cell.get('pheromone', 0) > 0 else 0
            sight.append(is_cell)
            sight.append(is_pheromone)
            # Sprawdź tylko przed siebie (idx == 0)
            if idx == 0 and is_cell:
                obj = cell['object']
                from agents.protist import Protist
                from agents.bacteria import Bacteria
                # Czy można zjeść?
                can_eat = False
                if isinstance(obj, Protist) and obj.is_alive:
                    if self.dietary_preference >= 171 or (86 <= self.dietary_preference <= 170):
                        can_eat = True
                elif isinstance(obj, Bacteria) and obj.is_alive and obj != self:
                    if self.dietary_preference <= 85 or (86 <= self.dietary_preference <= 170):
                        can_eat = True
                if can_eat:
                    can_eat_ahead = 1
                # Czy można się rozmnożyć?
                if self.can_reproduce_with(obj, input_size=self.input_size, output_size=self.output_size) and self.energy > self.division_threshold and self.reproduction_cooldown == 0:
                    can_reproduce_ahead = 1

        # Dodaj inne cechy genomu jako inputy (np. speed, sight_range, aggression_level, dietary_preference)
        extra_inputs = [
            self.speed / 10,
            self.sight_range / 10,
            self.sight_angle / 255,
            self.dietary_preference / 255,
            self.pheromone_preference / 255,
            self.turning_speed / 10,
            self.aggression_level / 10,
            can_eat_ahead,
            can_reproduce_ahead,
        ]

        return np.array([energy, age, health, chemotaxis, cooldown] + sight + extra_inputs)

    def maintenance_energy_cost(self):
        cost = 2
        cost += self.speed
        cost += self.sight_range
        cost += (self.sight_angle / 255)
        cost += self.aggression_level
        cost += self.pheromone_strength
        cost += self.max_health
        cost += self.metabolism_rate
        cost += self.chemotaxis_strength
        cost += self.chemotaxis_sensitivity
        cost += self.turning_speed
        cost += self.eating_strength
        cost -= self.photosynthesis_rate * 2
        cost -= self.reproduction_cooldown * 2
        return cost

    def act(self, environment_state, world=None):
        nn_input = self.sense_environment(environment_state)
        output = self.nn.forward(nn_input)
        action = np.argmax(output)
        # 0: pheromones, 1: reproduce, 2: move, 3: turn_left, 4: turn_right, 5: eat, 6: sprint
        result = None

        if action == 0:
            self.energy -= 2
            result = self.spread_pheromones()
        elif action == 1 and self.reproduction_cooldown == 0 and self.energy > self.division_threshold:
            self.energy -= 20
            self.reproduction_cooldown = int(self.genome[13]) if len(self.genome) > 13 else 0
            result = self.divide(environment_state)
        elif action == 2:
            self.move_forward(world)
        elif action == 3:
            self.turn(direction=-1)  # w lewo
        elif action == 4:
            self.turn(direction=1)   # w prawo
        elif action == 5:
            self.eat(environment_state)
        elif action == 6:
            self.sprint(world)
        else:
            self.energy -= 1

        self.energy -= self.maintenance_energy_cost()
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1
        self.age += 1
        self.energy = max(0, self.energy)
        self.vitals()
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

        child1 = Bacteria(chosen_positions[0][0], chosen_positions[0][1], genome=genome_child1, nn=nn_child1, input_size=self.input_size, output_size=self.output_size, population_genomes=self.population_genomes)
        child2 = Bacteria(chosen_positions[1][0], chosen_positions[1][1], genome=genome_child2, nn=nn_child2, input_size=self.input_size, output_size=self.output_size, population_genomes=self.population_genomes)
        return [(child1, chosen_positions[0]), (child2, chosen_positions[1])]

    def move_forward(self, world=None):
        directions = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
        dx, dy = directions[self.rotation % 8]
        moved = False
        for _ in range(self.speed):
            new_x = self.x + dx
            new_y = self.y + dy
            # Wrap-around (torus)
            if new_x < 0:
                new_x = self.grid_width - 1
            elif new_x >= self.grid_width:
                new_x = 0
            if new_y < 0:
                new_y = self.grid_height - 1
            elif new_y >= self.grid_height:
                new_y = 0
            if world and world.grid[new_y][new_x] is None:
                # Przenieś komórkę w siatce
                world.grid[self.y][self.x] = None
                world.grid[new_y][new_x] = self
                self.x = new_x
                self.y = new_y
                moved = True
            else:
                break
        if not moved:
            self.turn(direction=random.choice([-1, 1]))
        self.energy -= 2 * self.metabolism_rate  # koszt ruchu zależny od metabolizmu

    def turn(self, direction=None):
        if direction is None:
            direction = random.choice([-1, 1])
        self.rotation = (self.rotation + direction * self.turning_speed) % 8
        self.energy -= 1

    def eat(self, environment_state):
        # Sprawdza sąsiadów i preferencje żywieniowe, zadaje obrażenia i zyskuje energię
        for nx, ny in self.neighbors:
            cell_info = environment_state.get((nx, ny))
            if cell_info and cell_info["type"] == "cell":
                obj = cell_info["object"]
                from agents.protist import Protist
                from agents.bacteria import Bacteria
                # Preferencja: 0-85 tylko bakterie, 86-170 wszystko, 171-255 tylko protisty
                can_eat = False
                damage = 0
                energy_gain = 0
                if isinstance(obj, Protist) and obj.is_alive:
                    if self.dietary_preference >= 171 or (86 <= self.dietary_preference <= 170):
                        can_eat = True
                        damage = self.eating_strength + self.aggression_level  # obrażenia dla protistów
                        energy_gain = damage
                elif isinstance(obj, Bacteria) and obj.is_alive and obj != self:
                    if self.dietary_preference <= 85 or (86 <= self.dietary_preference <= 170):
                        can_eat = True
                        damage = self.eating_strength + self.aggression_level  # obrażenia dla bakterii
                        energy_gain = damage
                if can_eat:
                    # Zadaj obrażenia zdrowiu
                    if hasattr(obj, "max_health") and hasattr(obj, "is_alive"):
                        if not hasattr(obj, "health"):
                            obj.health = obj.max_health
                        obj.health -= damage
                        self.energy += damage + energy_gain
                        # Jeśli zdrowie spadło do zera lub niżej, komórka umiera
                        if obj.health <= 0:
                            obj.is_alive = False
                    break

    def __repr__(self):
        return f"Bacteria({self.x}, {self.y}, {self.name}, color={self.color})"

    def __eq__(self, other):
        if not isinstance(other, Bacteria):
            return False
        return super().__eq__(other) and self.name == other.name

    def __hash__(self):
        return hash((super().__hash__(), self.name))

    def sprint(self, world=None):
        # Sprint: porusz się 2x szybciej, koszt energii 4x większy
        directions = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
        dx, dy = directions[self.rotation % 8]
        for _ in range(self.speed * 2):
            new_x = self.x + dx
            new_y = self.y + dy
            if new_x < 0:
                new_x = self.grid_width - 1
            elif new_x >= self.grid_width:
                new_x = 0
            if new_y < 0:
                new_y = self.grid_height - 1
            elif new_y >= self.grid_height:
                new_y = 0
            if world and world.grid[new_y][new_x] is None:
                world.grid[self.y][self.x] = None
                world.grid[new_y][new_x] = self
                self.x = new_x
                self.y = new_y
            else:
                break
        self.energy -= 8 * self.metabolism_rate  # sprint kosztuje 4x więcej niż normalny ruch