import random
from src.core.agent import Agent

class Bacteria(Agent):
    species_id = 0

    @property
    def agent_type(self):
        return "Bacteria"

    def generate_founder_species_name(self, population_genomes):
        from src.core.species_naming import get_species_name
        return get_species_name(self.genome, population_genomes)

    def get_species_name(self, population_genomes):
        if hasattr(self, "simulation_engine") and self.simulation_engine is not None:
            founder_names = self.simulation_engine.founder_names
            if self.founder_id in founder_names:
                return founder_names[self.founder_id]
        from src.core.species_naming import get_species_name
        return get_species_name(self.genome, population_genomes)

    def act(self, environment):
        x, y = self.position
        grid = environment.grid
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]

        # 1) ZJEDZ SĄSIADA
        for dx, dy in dirs:
            nx, ny = (x+dx)%environment.width, (y+dy)%environment.height
            nbr = grid[ny][nx]
            if (
                nbr and nbr is not self
                and hasattr(self.genome, "diet")
                and getattr(nbr, "species_id", None) in self.genome.diet
                and nbr.is_alive
            ):
                nbr.is_alive = False
                self.energy += 60
                # koszt tury
                self.energy -= 1
                self.age += 1
                if self.energy <= 0 or self.age > self.genome.max_age:
                    self.is_alive = False
                return

        # 2) RUCH (jeśli nie zjadł)
        free = [( (x+dx)%environment.width, (y+dy)%environment.height )
                for dx,dy in dirs
                if grid[(y+dy)%environment.height][(x+dx)%environment.width] is None]
        if free:
            # wybór po zasobach/feromonach lub losowo
            # tu zachowujemy istniejącą logikę wyboru pola…
            # dla uproszczenia: losowo
            self.position = random.choice(free)
        # jeśli otoczony, stoi w miejscu

        # 3) zostaw feromon FOOD
        fx, fy = self.position
        environment.pheromones.add(fx, fy, amount=10, ptype='food')

        # 4) POBIERZ ZASOBY NA NOWEJ POZYCJI
        eaten = environment.resources.consume(fx, fy, amount=5)
        self.energy += eaten

        # 5) KOSZT UTRZYMANIA I STARZENIE
        self.energy -= 1
        self.age += 1
        if self.energy <= 0 or self.age > self.genome.max_age:
            self.is_alive = False

    def __init__(self, genome, neural_network, position, parent_ids=None, founder_id=None):
        super().__init__(genome, neural_network, position, parent_ids, founder_id)
        self.simulation_engine = None  # <- dodaj tę linię
