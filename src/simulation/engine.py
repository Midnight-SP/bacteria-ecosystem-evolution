from src.core.fungi import Fungi
import numpy as np
from typing import Optional
from src.utils.logger import StatsLogger
from src.core.evolution import mutate_genome
from threading import Lock, Thread

class SimulationEngine:
    def __init__(self, world, agents, config=None):
        self.world = world
        self.agents = agents
        self.config = config or {}
        self.genealogy = {}
        self.species_tree = {}
        self.species_counts = {}
        self.founders = {}
        self.founder_names = {}  # founder_id -> species_name
        self.logger: Optional[StatsLogger] = None
        self.step_count = 0

        # Buforuj agentów według id i founder_id
        self.agent_by_id = {a.id: a for a in self.agents}
        self.agents_by_founder = {}
        for agent in self.agents:
            self.agents_by_founder.setdefault(agent.founder_id, []).append(agent)

        # Dodaj ancestor do species_tree jeśli jest wśród agentów
        ancestor_name = "Ancestor ancestor [Ancestor]"
        if any(getattr(a, "agent_type", None) == "Ancestor" for a in self.agents):
            self.species_tree[ancestor_name] = set()

        # Buforuj population_genomes dla każdego typu agenta
        self.population_genomes = {}
        for agent in self.agents:
            if agent.agent_type not in self.population_genomes:
                self.population_genomes[agent.agent_type] = [
                    a.genome.genes for a in self.agents if a.agent_type == agent.agent_type and a.is_alive
                ]

        for agent in self.agents:
            population_genomes = self.population_genomes[agent.agent_type]
            self.founders[agent.founder_id] = agent.genome.genes.copy()
            # Jeśli to founder, nadaj mu nazwę na podstawie genomu
            if agent.founder_id == agent.id:
                name = agent.generate_founder_species_name(population_genomes)
                self.founder_names[agent.founder_id] = name

            name = agent.get_species_name(population_genomes)
            if name not in self.species_tree:
                parent_species = set()
                for pid in getattr(agent, "parent_ids", []):
                    parent_agent = self.agent_by_id.get(pid)
                    if parent_agent:
                        parent_population = self.population_genomes[parent_agent.agent_type]
                        parent_species.add(parent_agent.get_species_name(parent_population))
                self.species_tree[name] = parent_species
            self.species_counts[name] = self.species_counts.get(name, 0) + 1
            self.founders[agent.founder_id] = agent.genome.genes.copy()

        # Buforuj statystyki
        self._cached_stats = None
        self._stats_dirty = True

        self.founder_parents = {}  # founder_id -> parent_founder_id

        # Ustal parent_founder_id dla founderów
        for agent in self.agents:
            if agent.founder_id == agent.id:
                # founderzy pra-agentów mają parent_ids
                if agent.parent_ids:
                    parent_id = agent.parent_ids[0]
                    parent_agent = self.agent_by_id.get(parent_id)
                    if parent_agent:
                        self.founder_parents[agent.founder_id] = parent_agent.founder_id
                    else:
                        self.founder_parents[agent.founder_id] = None
                else:
                    self.founder_parents[agent.founder_id] = None

    def _update_population_genomes(self):
        self.population_genomes = {}
        for agent in self.agents:
            if agent.agent_type not in self.population_genomes:
                self.population_genomes[agent.agent_type] = [
                    a.genome.genes for a in self.agents if a.agent_type == agent.agent_type and a.is_alive
                ]

    def _update_species_counts(self):
        self.species_counts = {}
        for agent in self.agents:
            population_genomes = self.population_genomes[agent.agent_type]
            name = agent.get_species_name(population_genomes)
            self.species_counts[name] = self.species_counts.get(name, 0) + 1

    def _update_stats(self):
        self._update_population_genomes()
        self._update_species_counts()
        self._cached_stats = {
            "step": self.step_count,
            "n_agents": len(self.agents),
            "n_families": len(set(a.founder_id for a in self.agents)),
            "n_species": len(set(
                a.get_species_name(self.population_genomes[a.agent_type]) for a in self.agents
            )),
        }
        self._stats_dirty = False

    def step(self):
        new_agents = []
        # Czyść grid
        for y in range(self.world.height):
            for x in range(self.world.width):
                self.world.grid[y][x] = None

        # Wpisz wszystkich żywych agentów na ich pozycje
        for agent in self.agents:
            if agent.is_alive:
                x, y = agent.position
                self.world.grid[y][x] = agent

        # SEKWENCYJNA PĘTLA AGENTÓW (brak wątków, brak snapshotu)
        for agent in self.agents:
            if agent.is_alive:
                result = agent.act(self.world)
                x, y = agent.position
                self.world.grid[y][x] = agent
                if hasattr(agent, "can_reproduce") and agent.can_reproduce(self.world):
                    child = agent.reproduce()
                    child.energy = agent.energy // 2
                    agent.energy = agent.energy // 2

                    founder_genome = self.founders.get(agent.founder_id, agent.genome.genes)
                    import numpy as np
                    diff = np.sum(np.abs(np.array(child.genome.genes) - np.array(founder_genome)))
                    threshold = 512

                    if diff > threshold:
                        parent_founder_id = agent.founder_id
                        subfamilies = [
                            fid for fid, parent in self.founder_parents.items()
                            if parent == parent_founder_id
                        ]
                        found_similar = False
                        for subfid in subfamilies:
                            subfounder_genome = self.founders.get(subfid)
                            if subfounder_genome is not None:
                                subdiff = np.sum(np.abs(np.array(child.genome.genes) - np.array(subfounder_genome)))
                                if subdiff < threshold:
                                    child.founder_id = subfid
                                    self.agents_by_founder.setdefault(subfid, []).append(child)
                                    found_similar = True
                                    break
                        if not found_similar:
                            child.founder_id = child.id
                            self.founders[child.founder_id] = child.genome.genes.copy()
                            self.agents_by_founder[child.founder_id] = [child]
                            self.founder_parents[child.founder_id] = agent.founder_id
                            population_genomes = self.get_population_genomes(child.agent_type)
                            name = child.generate_founder_species_name(population_genomes)
                            self.founder_names[child.founder_id] = name
                    else:
                        child.founder_id = agent.founder_id
                        self.agents_by_founder.setdefault(child.founder_id, []).append(child)

                    child.simulation_engine = self

                    new_agents.append(child)
                    self.genealogy[child.id] = child.parent_ids

                    if isinstance(agent, Fungi) and result is not None:
                        from src.core.genome import FungiGenome
                        from src.core.neural_network import NeuralNetwork
                        x, y = result
                        neighbor = None
                        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nx, ny = (x + dx) % self.world.width, (y + dy) % self.world.height
                            cell = self.world.grid[ny][nx]
                            if isinstance(cell, Fungi) and cell.is_alive and cell is not agent:
                                neighbor = cell
                                break
                        if neighbor:
                            from src.core.evolution import uniform_crossover, mutate_genome
                            child_genes = uniform_crossover(agent.genome.genes, neighbor.genome.genes)
                            child_genes = mutate_genome(child_genes, rate=0.05)
                            genome = FungiGenome(child_genes)
                            child_weights = (agent.nn.weights + neighbor.nn.weights) / 2
                            nn = NeuralNetwork(agent.nn.input_size, agent.nn.output_size, weights=child_weights)
                            nn.mutate(rate=0.05)
                            parent_ids = [agent.id, neighbor.id]
                        else:
                            from src.core.evolution import mutate_genome
                            child_genes = mutate_genome(agent.genome.genes.copy(), rate=0.3)
                            genome = FungiGenome(child_genes)
                            nn = NeuralNetwork(agent.nn.input_size, agent.nn.output_size)
                            parent_ids = [agent.id]
                        new_fungi = Fungi(genome, nn, result, parent_ids=parent_ids, founder_id=agent.founder_id)
                        new_agents.append(new_fungi)
                        self.genealogy[new_fungi.id] = new_fungi.parent_ids

        max_agents = self.config.get("evolution", {}).get("max_agents", 20000)
        if len(self.agents) + len(new_agents) > max_agents:
            new_agents = new_agents[:max_agents - len(self.agents)]
        self.agents.extend(new_agents)
        self.agents = [a for a in self.agents if a.is_alive]
        self.agent_by_id = {a.id: a for a in self.agents}
        self.agents_by_founder = {}
        for agent in self.agents:
            self.agents_by_founder.setdefault(agent.founder_id, []).append(agent)
        self._stats_dirty = True

        self.world.update()

        if self.logger is not None and self.step_count % 100 == 0:
            if self._stats_dirty or self._cached_stats is None:
                self._update_stats()
            if self._cached_stats is not None:
                self.logger.log(self._cached_stats)
        self.step_count += 1

    def get_population_genomes(self, agent_type):
        # Zwróć z bufora, jeśli istnieje
        if agent_type in self.population_genomes:
            return self.population_genomes[agent_type]
        genomes = [a.genome.genes for a in self.agents if a.agent_type == agent_type and a.is_alive]
        self.population_genomes[agent_type] = genomes
        return genomes

def get_sector(x, y, width, height):
    if x < width // 2 and y < height // 2:
        return 0  # lewy górny
    elif x >= width // 2 and y < height // 2:
        return 1  # prawy górny
    elif x < width // 2 and y >= height // 2:
        return 2  # lewy dolny
    else:
        return 3  # prawy dolny