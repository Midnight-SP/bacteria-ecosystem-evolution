from src.core.agent import Agent
import random

class Protozoa(Agent):
    species_id = 3

    @property
    def agent_type(self):
        return "Protozoa"

    def __init__(self, genome, neural_network, position, parent_ids=None, founder_id=None):
        super().__init__(genome, neural_network, position, parent_ids, founder_id)
        # jeżeli gdzieś używasz self.nn, zachowaj alias:
        self.nn = neural_network
        self.simulation_engine = None

    def act(self, environment):
        x, y = self.position
        grid = environment.grid
        directions = [(-1,0), (1,0), (0,-1), (0,1)]

        # --- ZJEDZ SĄSIADÓW ---
        for dx, dy in directions:
            nx, ny = (x+dx) % environment.width, (y+dy) % environment.height
            nbr = grid[ny][nx]
            if (
                nbr is not None and nbr is not self
                and hasattr(self.genome, "diet")
                and hasattr(nbr, "species_id")
                and nbr.species_id in self.genome.diet
                and nbr.species_id != self.species_id
                and getattr(nbr, "is_alive", True)
            ):
                nbr.is_alive = False
                self.energy += 40
                self._post_action()
                return

        # --- RUCH I ZBIERANIE ZASOBÓW (skrótowo) ---
        blocked = True
        for dx, dy in directions:
            nx, ny = (x+dx) % environment.width, (y+dy) % environment.height
            if grid[ny][nx] is None:
                blocked = False
                break
        if blocked:
            self._post_action()
            return

        # tu Twoja logika wyboru pola (zasoby / feromony / losowo)…
        dx, dy = random.choice(directions)
        nx, ny = (x+dx) % environment.width, (y+dy) % environment.height
        if grid[ny][nx] is None:
            self.position = (nx, ny)

        # zostaw feromon, zbierz zasób, itd.
        eaten = environment.resources.consume(*self.position, amount=3)
        self.energy += eaten
        self._post_action()

    def _post_action(self):
        # starzenie + zużycie energii, zamykające życie
        self.energy -= 1
        self.age += 1
        if self.energy <= 0 or (hasattr(self.genome, "max_age") and self.age > self.genome.max_age):
            self.is_alive = False

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