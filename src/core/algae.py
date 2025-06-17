from src.core.agent import Agent
import random

class Algae(Agent):
    species_id = 1

    @property
    def agent_type(self):
        return "Algae"
    
    def generate_founder_species_name(self, population_genomes):
        from src.core.species_naming import get_species_name
        return get_species_name(self.genome, population_genomes)


    def get_species_name(self, population_genomes):
        # Jeśli jestem founderem, generuję nazwę na podstawie genomu
        if hasattr(self, "simulation_engine") and self.simulation_engine is not None:
            founder_names = self.simulation_engine.founder_names
            if self.founder_id in founder_names:
                return founder_names[self.founder_id]
        # fallback: jeśli nie mam dostępu do simulation_engine, generuj nazwę jak dawniej
        from src.core.species_naming import get_species_name
        return get_species_name(self.genome, population_genomes)

    def act(self, environment):
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
            self.energy -= 2  # większy koszt stania w miejscu
            self.age += 1
            if self.energy <= 0 or self.age > self.genome.max_age:
                self.is_alive = False
            return None

        # --- ROZMNAŻANIE ALGI ---
        spawn_pos = None
        # obniżone progi: energia>=100, wiek>=10
        if self.energy >= 100 and self.age >= 10:
            random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = (x + dx) % environment.width, (y + dy) % environment.height
                if grid[ny][nx] is None:
                    spawn_pos = (nx, ny)
                    # dzielimy energię po równo
                    self.energy //= 2
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
            if grid[ny][nx] is None:
                x, y = nx, ny
            self.position = (x, y)

        # Fotosynteza: podbijamy zysk energii
        self.energy += max(1, self.genome.photosynthesis // 4)

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

    def __init__(self, genome, neural_network, position, parent_ids=None, founder_id=None):
        super().__init__(genome, neural_network, position, parent_ids, founder_id)
        self.simulation_engine = None  # <- dodaj tę linię