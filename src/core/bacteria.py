import random
from src.core.agent import Agent

class Bacteria(Agent):
    species_id = 0

    @property
    def agent_type(self):
        return "Bacteria"

    def get_species_name(self, population_genomes):
        from src.core.species_naming import get_species_name
        return get_species_name(self.genome, population_genomes)

    def act(self, environment):
        x, y = self.position
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        blocked = True
        for dx, dy in directions:
            nx, ny = (x + dx) % environment.width, (y + dy) % environment.height
            if environment.grid[ny][nx] is None:
                blocked = False
                break
        if blocked:
            self.energy -= 1
            self.age += 1
            if self.energy <= 0 or self.age > self.genome.max_age:
                self.is_alive = False
            return None

        # Sprawdź poziom zasobów w sąsiedztwie
        resource_levels = []
        for dx, dy in directions:
            nx, ny = (x + dx) % environment.width, (y + dy) % environment.height
            resource_levels.append(environment.resources.get(nx, ny))

        if resource_levels and max(resource_levels) > 0:
            idx = resource_levels.index(max(resource_levels))
            dx, dy = directions[idx]
        else:
            # Odczyt feromonów 'food' w sąsiedztwie
            pheromone_levels = []
            for dx, dy in directions:
                nx, ny = (x + dx) % environment.width, (y + dy) % environment.height
                pheromone_levels.append(environment.pheromones.get(nx, ny, ptype='food'))
            if pheromone_levels and max(pheromone_levels) > 0:
                idx = pheromone_levels.index(max(pheromone_levels))
                dx, dy = directions[idx]
            else:
                dx, dy = random.choice(directions)

        nx, ny = (x + dx) % environment.width, (y + dy) % environment.height
        if environment.grid[ny][nx] is None:
            self.position = (nx, ny)
        else:
            self.position = (x, y)
        # Zostaw feromon 'food'
        environment.pheromones.add(self.position[0], self.position[1], amount=10, ptype='food')

        # Konsumuj surowce
        eaten = environment.resources.consume(x, y, amount=5)
        self.energy += eaten - 1

        # Interakcja z innymi agentami (np. agresja)
        cell = environment.grid[y][x]
        if cell and cell is not self:
            if hasattr(self.genome, "diet") and hasattr(cell, "species_id"):
                if cell.species_id in self.genome.diet and cell.species_id != self.species_id:
                    cell.is_alive = False
                    self.energy += 60

        self.energy -= 1
        self.age += 1
        if self.energy <= 0 or self.age > self.genome.max_age:
            self.is_alive = False