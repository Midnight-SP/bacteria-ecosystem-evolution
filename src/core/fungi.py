from typing import Optional, Tuple
from src.core.agent import Agent
import random

class Fungi(Agent):
    species_id = 2 

    @property
    def agent_type(self):
        return "Fungi"

    def get_species_name(self, population_genomes):
        from src.core.species_naming import get_species_name
        return get_species_name(self.genome, population_genomes)

    def act(self, environment) -> Optional[Tuple[int, int]]:
        x, y = self.position
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        blocked = True
        for dx, dy in directions:
            nx, ny = (x + dx) % environment.width, (y + dy) % environment.height
            if environment.grid[ny][nx] is None:
                blocked = False
                break
        if blocked:
            # Agent nie może się ruszyć ani wykonać innych akcji
            self.energy -= 1
            self.age += 1
            if self.energy <= 0 or self.age > self.genome.max_age:
                self.is_alive = False
            return None

        spawn_pos = None
        # Grzyb może się rozprzestrzeniać tylko jeśli jest wystarczająco stary i ma dużo energii
        if self.energy > 20 and self.age > 10:
            if random.random() < (self.genome.spore_spread / 5000):
                directions = [(-1,0), (1,0), (0,-1), (0,1)]
                random.shuffle(directions)
                for dx, dy in directions:
                    nx, ny = (x + dx) % environment.width, (y + dy) % environment.height
                    if environment.grid[ny][nx] is None:
                        spawn_pos = (nx, ny)
                        break

        # Konsumuj surowce (np. rozkład materii)
        eaten = environment.resources.consume(x, y, amount=2)
        self.energy += eaten - 1

        # Zostaw feromon "rozkładu"
        environment.pheromones.add(x, y, amount=5, ptype='algae')

        self.energy -= 1
        self.age += 1
        if self.energy <= 0 or self.age > self.genome.max_age:
            self.is_alive = False

        return spawn_pos