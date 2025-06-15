from src.core.agent import Agent
import random

class Algae(Agent):
    species_id = 1

    @property
    def agent_type(self):
        return "Algae"

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
            self.energy -= 2  # większy koszt stania w miejscu
            self.age += 1
            if self.energy <= 0 or self.age > self.genome.max_age:
                self.is_alive = False
            return None

        # Rozprzestrzenianie gdy osiągnięto limit energii i odpowiedni wiek
        spawn_pos = None
        min_divide_age = 200  # większy minimalny wiek do podziału
        if self.energy >= 800 and self.age >= min_divide_age:
            random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = (x + dx) % environment.width, (y + dy) % environment.height
                if environment.grid[ny][nx] is None:
                    spawn_pos = (nx, ny)
                    self.energy = self.energy // 2 - 100  # podziel energię i odejmij koszt podziału
                    break

        # Minimalny ruch (algi mogą się powoli rozprzestrzeniać)
        if self.age % 20 == 0 and not spawn_pos:
            random.shuffle(directions)
            resource_levels = []
            for dx, dy in directions:
                nx, ny = (x + dx) % environment.width, (y + dy) % environment.height
                resource_levels.append(environment.resources.get(nx, ny))
            max_resource = max(resource_levels)
            if max_resource > 0:
                idx = resource_levels.index(max_resource)
                dx, dy = directions[idx]
            else:
                pheromone_levels = []
                for dx, dy in directions:
                    nx, ny = (x + dx) % environment.width, (y + dy) % environment.height
                    pheromone_levels.append(environment.pheromones.get(nx, ny, ptype='danger'))
                if pheromone_levels and min(pheromone_levels) < 1e6:
                    idx = pheromone_levels.index(min(pheromone_levels))
                    dx, dy = directions[idx]
                else:
                    dx, dy = random.choice(directions)
            nx, ny = (x + dx) % environment.width, (y + dy) % environment.height
            if environment.grid[ny][nx] is None:
                x, y = nx, ny
            self.position = (x, y)

        # Fotosynteza: zysk energii co turę (zmniejszony)
        self.energy += max(1, self.genome.photosynthesis // 16)

        # Konsumuj minimalnie surowce (np. minerały)
        eaten = environment.resources.consume(x, y, amount=1)
        self.energy += eaten

        # Zostaw feromon "pokarmowy"
        environment.pheromones.add(x, y, amount=2, ptype='algae')

        self.age += 1
        self.energy -= 1  # koszt utrzymania co turę
        if self.energy <= 0 or self.age > self.genome.max_age:
            self.is_alive = False

        return spawn_pos