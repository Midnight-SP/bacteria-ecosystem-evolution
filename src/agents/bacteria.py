from typing import Any, List, Tuple, Optional, cast
import numpy as np
from agents.cell import Cell
import random

class Bacteria(Cell):
    def __init__(self, x: int, y: int, name: str, genome=None, nn=None, input_size=18, output_size=5):
        super().__init__(x, y, genome, nn=nn, input_size=input_size, output_size=output_size)
        self.name = name
        self.initial_rotation = random.randint(0, 7)  # 8 kierunków
        self.rotation = self.initial_rotation

    def act(self, environment_state) -> Optional[List[Tuple[Cell, Any]]]:
        nn_input = self.sense_environment(environment_state)
        action = np.argmax(self.nn.forward(nn_input))
        # Każda akcja zużywa energię
        if action == 0:
            self.energy -= 5
            # Rzutowanie typu na List[Tuple[Cell, Any]]
            return cast(Optional[List[Tuple[Cell, Any]]], self.divide(self.neighbors))
        elif action == 1:
            self.energy -= 2
            self.spread_pheromones()
        elif action == 2:
            self.move_forward()
        elif action == 3:
            self.turn(direction=random.choice([-1, 1]))
        elif action == 4:
            self.eat_protist(environment_state)
        else:
            self.energy -= 1  # brak akcji

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
        child1 = Bacteria(chosen_positions[0][0], chosen_positions[0][1], name=self.name, genome=genome_child1, nn=nn_child1, input_size=self.input_size, output_size=self.output_size)
        child2 = Bacteria(chosen_positions[1][0], chosen_positions[1][1], name=self.name, genome=genome_child2, nn=nn_child2, input_size=self.input_size, output_size=self.output_size)
        return [(child1, chosen_positions[0]), (child2, chosen_positions[1])]

    def move_forward(self):
        directions = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
        dx, dy = directions[self.rotation % 8]
        new_x = self.x + dx
        new_y = self.y + dy

        # Ensure the new position is within the grid boundaries
        if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height:
            self.x = new_x
            self.y = new_y
        else:
            # Optionally handle out-of-bound behavior, e.g., wrap around or stop movement
            pass

        self.energy -= 2

    def turn(self, direction=None):
        if direction is None:
            direction = random.choice([-1, 1])
        self.rotation = (self.rotation + direction) % 8
        self.energy -= 1

    def eat_protist(self, environment_state):
        # Sprawdź czy na sąsiednim polu jest protist i "zjedz" go
        for nx, ny in self.neighbors:
            cell_info = environment_state.get((nx, ny))
            if cell_info and cell_info["type"] == "cell":
                obj = cell_info["object"]
                # Import lokalny, żeby uniknąć cyklicznych importów
                from agents.protist import Protist
                if isinstance(obj, Protist) and obj.is_alive:
                    obj.is_alive = False
                    self.energy += 20  # np. +20 energii za zjedzenie protista
                    break

    def __repr__(self):
        return f"Bacteria({self.x}, {self.y}, {self.name})"

    def __eq__(self, other):
        if not isinstance(other, Bacteria):
            return False
        return super().__eq__(other) and self.name == other.name

    def __hash__(self):
        return hash((super().__hash__(), self.name))