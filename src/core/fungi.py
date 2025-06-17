from src.core.agent import Agent
import random
from typing import Optional, Tuple


class Fungi(Agent):
    species_id = 2

    @property
    def agent_type(self):
        return "Fungi"

    def generate_founder_species_name(self, population_genomes):
        from src.core.species_naming import get_species_name
        return get_species_name(self.genome, population_genomes)

    def get_species_name(self, population_genomes):
        if hasattr(self, "simulation_engine") and self.simulation_engine is not None:
            fn = self.simulation_engine.founder_names
            if self.founder_id in fn:
                return fn[self.founder_id]
        from src.core.species_naming import get_species_name
        return get_species_name(self.genome, population_genomes)

    def __init__(self, genome, nn, position, parent_ids=None, founder_id=None):
        # <-- teraz dziedziczymy po Agent, w tym kolor, id, energy, age itp.
        super().__init__(genome, nn, position, parent_ids, founder_id)
        # alias do sieci jeśli potrzebujesz
        self.nn = nn
        # engine zostaje podpięty później z main.py
        self.simulation_engine = None

    def act(self, environment) -> Optional[Tuple[int, int]]:
        x, y = self.position
        grid = environment.grid

        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        blocked = True
        for dx, dy in directions:
            nx, ny = (x + dx) % environment.width, (y + dy) % environment.height
            if grid[ny][nx] is None:
                blocked = False
                break
        if blocked:
            self.energy -= 1
            self.age += 1
            if self.energy <= 0 or self.age > self.genome.max_age:
                self.is_alive = False
            return None

        # --- SPORY GRZYBÓW: losowa szansa wg gen.spore_spread ---
        spawn_pos = None
        if self.energy >= 50 and self.age >= 5 and random.random() < (self.genome.spore_spread / 100):
            directions = [(-1,0), (1,0), (0,-1), (0,1)]
            random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = (x + dx) % environment.width, (y + dy) % environment.height
                if grid[ny][nx] is None:
                    spawn_pos = (nx, ny)
                    break

        eaten = environment.resources.consume(x, y, amount=2)
        self.energy += eaten - 1

        environment.pheromones.add(x, y, amount=5, ptype='algae')

        self.energy -= 1
        self.age += 1
        if self.energy <= 0 or self.age > self.genome.max_age:
            self.is_alive = False

        return spawn_pos