import numpy as np
from src.environment.world import World
from src.core.bacteria import Bacteria
from src.core.algae import Algae
from src.core.fungi import Fungi
from src.core.protozoa import Protozoa
from src.core.genome import BacteriaGenome, AlgaeGenome, FungiGenome, ProtozoaGenome, Genome
from src.core.neural_network import NeuralNetwork
from src.simulation.engine import SimulationEngine
from src.ui.gui import run_gui
from src.utils.config import load_config
from src.utils.logger import StatsLogger
from src.core.agent import Agent

# --- PRZENIESIONA KLASA AncestorAgent ---
class AncestorAgent(Agent):
    def __init__(self, genome, nn, pos, parent_ids=None, founder_id=None):
        super().__init__(genome, nn, pos, parent_ids=parent_ids, founder_id=founder_id)
        self.founder_id = self.id
        self.is_alive = True

    @property
    def agent_type(self):
        return "Ancestor"
    def get_species_name(self, population_genomes):
        return "Ancestor ancestor [Ancestor]"
    def generate_founder_species_name(self, population_genomes):
        return "Ancestor ancestor [Ancestor]"
    def act(self, environment):
        return None

def create_agents(world, n_bacteria, n_algae, n_fungi, n_protozoa):
    agents = []

    # 0. Stwórz pra_pra founder (ancestor)
    pra_pra_genome = Genome(np.array([128, 128, 128, 128, 128, 128], dtype=np.uint8))
    pra_pra_nn = NeuralNetwork(5, 3)
    pra_pra = AncestorAgent(pra_pra_genome, pra_pra_nn, (0, 0), parent_ids=[], founder_id=None)
    agents.append(pra_pra)

    # 1. Stwórz pra-founderów z sensownymi genami
    pra_bacteria_genes = np.array([
        220, 120, 220, 50, 50, 220, 240, 1, 2
    ], dtype=np.uint8)
    pra_bacteria = Bacteria(BacteriaGenome(pra_bacteria_genes), NeuralNetwork(5, 3), (0, 0), parent_ids=[pra_pra.id])
    pra_bacteria.founder_id = pra_bacteria.id

    pra_algae_genes = np.array([
        180, 200, 50, 220, 50, 120, 0
    ], dtype=np.uint8)
    pra_algae = Algae(AlgaeGenome(pra_algae_genes), NeuralNetwork(5, 3), (0, 0), parent_ids=[pra_pra.id])
    pra_algae.founder_id = pra_algae.id

    pra_fungi_genes = np.array([
        160, 180, 120, 50, 220, 200, 3
    ], dtype=np.uint8)
    pra_fungi = Fungi(FungiGenome(pra_fungi_genes), NeuralNetwork(5, 3), (0, 0), parent_ids=[pra_pra.id])
    pra_fungi.founder_id = pra_fungi.id

    pra_protozoa_genes = np.array([
        210, 140, 50, 50, 220, 240, 240, 0, 1, 2
    ], dtype=np.uint8)
    pra_protozoa = Protozoa(ProtozoaGenome(pra_protozoa_genes), NeuralNetwork(5, 3), (0, 0), parent_ids=[pra_pra.id])
    pra_protozoa.founder_id = pra_protozoa.id

    agents.extend([pra_bacteria, pra_algae, pra_fungi, pra_protozoa])

    # 2. Twórz agentów z founder_id ustawionym na pra-founderów i identycznymi genami
    for _ in range(n_bacteria):
        genome = BacteriaGenome(pra_bacteria_genes.copy())
        nn = NeuralNetwork(5, 3)
        pos = (np.random.randint(0, world.width), np.random.randint(0, world.height))
        agents.append(Bacteria(genome, nn, pos, parent_ids=[pra_bacteria.id], founder_id=pra_bacteria.id))

    for _ in range(n_algae):
        genome = AlgaeGenome(pra_algae_genes.copy())
        nn = NeuralNetwork(5, 3)
        pos = (np.random.randint(0, world.width), np.random.randint(0, world.height))
        agents.append(Algae(genome, nn, pos, parent_ids=[pra_algae.id], founder_id=pra_algae.id))

    for _ in range(n_fungi):
        genome = FungiGenome(pra_fungi_genes.copy())
        nn = NeuralNetwork(5, 3)
        pos = (np.random.randint(0, world.width), np.random.randint(0, world.height))
        agents.append(Fungi(genome, nn, pos, parent_ids=[pra_fungi.id], founder_id=pra_fungi.id))

    for _ in range(n_protozoa):
        genome = ProtozoaGenome(pra_protozoa_genes.copy())
        nn = NeuralNetwork(5, 3)
        pos = (np.random.randint(0, world.width), np.random.randint(0, world.height))
        agents.append(Protozoa(genome, nn, pos, parent_ids=[pra_protozoa.id], founder_id=pra_protozoa.id))

    return agents

def main():
    config = load_config("config.yaml")
    width = config["world"]["width"]
    height = config["world"]["height"]
    world = World(width, height)
    agents = create_agents(
        world,
        n_bacteria=config["agents"]["n_bacteria"],
        n_algae=config["agents"]["n_algae"],
        n_fungi=config["agents"]["n_fungi"],
        n_protozoa=config["agents"]["n_protozoa"]
    )
    engine = SimulationEngine(world, agents, config)
    for agent in agents:
        agent.simulation_engine = engine
    engine.logger = StatsLogger("stats.csv")
    engine.step_count = 0
    if config.get("batch", False):
        for _ in range(config["evolution"]["max_steps"]):
            engine.step()
        # eksportuj wyniki/statystyki
    else:
        run_gui(engine)

if __name__ == "__main__":
    main()