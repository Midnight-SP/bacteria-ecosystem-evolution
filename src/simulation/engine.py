from src.core.fungi import Fungi

class SimulationEngine:
    def __init__(self, world, agents):
        self.world = world
        self.agents = agents
        self.genealogy = {}
        self.species_tree = {}
        self.species_counts = {}

        # Dodaj ancestor do species_tree jeśli jest wśród agentów
        ancestor_name = "Ancestor ancestor [Ancestor]"
        if any(getattr(a, "agent_type", None) == "Ancestor" for a in self.agents):
            self.species_tree[ancestor_name] = set()

        for agent in self.agents:
            population_genomes = [a.genome.genes for a in self.agents if a.agent_type == agent.agent_type and a.is_alive]
            name = agent.get_species_name(population_genomes)
            if name not in self.species_tree:
                parent_species = set()
                for pid in getattr(agent, "parent_ids", []):
                    parent_agent = next((a for a in self.agents if a.id == pid), None)
                    if parent_agent:
                        parent_population = [a.genome.genes for a in self.agents if a.agent_type == parent_agent.agent_type and a.is_alive]
                        parent_species.add(parent_agent.get_species_name(parent_population))
                self.species_tree[name] = parent_species
            self.species_counts[name] = self.species_counts.get(name, 0) + 1

    def step(self):
        # Zbierz pozycje zajęte przez agentów
        occupied_positions = [agent.position for agent in self.agents if agent.is_alive]
        # Wyczyść tylko zajęte komórki
        for x, y in occupied_positions:
            self.world.grid[y][x] = None

        new_agents = []
        for agent in self.agents:
            if agent.is_alive:
                result = agent.act(self.world)
                x, y = agent.position
                self.world.grid[y][x] = agent
                if hasattr(agent, "can_reproduce") and agent.can_reproduce():
                    child = agent.reproduce()
                    child.energy = agent.energy // 2
                    agent.energy = agent.energy // 2
                    new_agents.append(child)
                    self.genealogy[child.id] = child.parent_ids
                    # --- Nowe: budowanie drzewa rodzin ---
                    population_genomes = [a.genome.genes for a in self.agents if a.agent_type == child.agent_type and a.is_alive]
                    child_species = child.get_species_name(population_genomes)
                    parent_species = set()
                    for pid in child.parent_ids:
                        parent_agent = next((a for a in self.agents if a.id == pid), None)
                        if parent_agent:
                            parent_population = [a.genome.genes for a in self.agents if a.agent_type == parent_agent.agent_type and a.is_alive]
                            parent_species.add(parent_agent.get_species_name(parent_population))
                    if child_species not in self.species_tree:
                        self.species_tree[child_species] = parent_species
                # Rozprzestrzenianie grzybów – przekazujemy parent_ids=[agent.id]
                if isinstance(agent, Fungi) and result is not None:
                    from src.core.genome import FungiGenome
                    from src.core.neural_network import NeuralNetwork
                    genes = agent.genome.genes.copy()
                    genome = FungiGenome(genes)
                    nn = NeuralNetwork(agent.nn.input_size, agent.nn.output_size)
                    new_fungi = Fungi(genome, nn, result, parent_ids=[agent.id])
                    new_agents.append(new_fungi)
                    self.genealogy[new_fungi.id] = new_fungi.parent_ids
        self.agents.extend(new_agents)
        # Usuń martwych agentów
        self.agents = [a for a in self.agents if a.is_alive]
        self.world.update()
        # --- Nowe: liczenie agentów w rodzinach ---
        self.species_counts = {}
        for agent in self.agents:
            population_genomes = [a.genome.genes for a in self.agents if a.agent_type == agent.agent_type and a.is_alive]
            name = agent.get_species_name(population_genomes)
            self.species_counts[name] = self.species_counts.get(name, 0) + 1

    def get_population_genomes(self, agent_type):
        return [a.genome.genes for a in self.agents if a.agent_type == agent_type and a.is_alive]