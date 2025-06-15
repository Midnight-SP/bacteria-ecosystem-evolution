import numpy as np
from src.environment.world import World
from src.core.bacteria import Bacteria
from src.core.algae import Algae
from src.core.fungi import Fungi
from src.core.protozoa import Protozoa
from src.core.genome import BacteriaGenome, AlgaeGenome, FungiGenome, ProtozoaGenome
from src.core.neural_network import NeuralNetwork
from src.simulation.engine import SimulationEngine
from src.ui.gui import run_gui

def create_agents(world, n_bacteria, n_algae, n_fungi, n_protozoa):
    agents = []

    # 0. Stwórz praprarodzica (uniwersalny agent)
    from src.core.genome import Genome
    from src.core.agent import Agent
    from src.core.neural_network import NeuralNetwork

    class AncestorAgent(Agent):
        @property
        def agent_type(self):
            return "Ancestor"
        def get_species_name(self, population_genomes):
            return "Ancestor ancestor [Ancestor]"
        def act(self, environment):
            return None

    pra_pra_genome = Genome([128, 128, 128, 128, 128, 128])
    pra_pra_nn = NeuralNetwork(5, 3)
    pra_pra = AncestorAgent(pra_pra_genome, pra_pra_nn, (0, 0), parent_ids=[])
    agents.append(pra_pra)

    # 1. Stwórz prarodziców, każdy z parent_ids=[pra_pra.id]
    pra_bacteria = Bacteria(BacteriaGenome(np.random.randint(0, 256, 9)), NeuralNetwork(5, 3), (0, 0), parent_ids=[pra_pra.id])
    pra_algae = Algae(AlgaeGenome(np.random.randint(0, 256, 7)), NeuralNetwork(5, 3), (0, 0), parent_ids=[pra_pra.id])
    pra_fungi = Fungi(FungiGenome(np.random.randint(0, 256, 7)), NeuralNetwork(5, 3), (0, 0), parent_ids=[pra_pra.id])
    pra_protozoa = Protozoa(ProtozoaGenome(np.random.randint(0, 256, 10)), NeuralNetwork(5, 3), (0, 0), parent_ids=[pra_pra.id])

    agents.extend([pra_bacteria, pra_algae, pra_fungi, pra_protozoa])

    # 2. Twórz agentów z parent_ids ustawionym na id prarodzica
    for _ in range(n_bacteria):
        genome = BacteriaGenome(np.random.randint(0, 256, 9))
        nn = NeuralNetwork(5, 3)
        pos = (np.random.randint(0, world.width), np.random.randint(0, world.height))
        agents.append(Bacteria(genome, nn, pos, parent_ids=[pra_bacteria.id]))

    for _ in range(n_algae):
        genome = AlgaeGenome(np.random.randint(0, 256, 7))
        nn = NeuralNetwork(5, 3)
        pos = (np.random.randint(0, world.width), np.random.randint(0, world.height))
        agents.append(Algae(genome, nn, pos, parent_ids=[pra_algae.id]))

    for _ in range(n_fungi):
        genome = FungiGenome(np.random.randint(0, 256, 7))
        nn = NeuralNetwork(5, 3)
        pos = (np.random.randint(0, world.width), np.random.randint(0, world.height))
        agents.append(Fungi(genome, nn, pos, parent_ids=[pra_fungi.id]))

    for _ in range(n_protozoa):
        genome = ProtozoaGenome(np.random.randint(0, 256, 10))
        nn = NeuralNetwork(5, 3)
        pos = (np.random.randint(0, world.width), np.random.randint(0, world.height))
        agents.append(Protozoa(genome, nn, pos, parent_ids=[pra_protozoa.id]))

    return agents

def main():
    width, height = 100, 100
    world = World(width, height)
    agents = create_agents(
        world,
        n_bacteria=50,
        n_algae=20,
        n_fungi=20,
        n_protozoa=50
    )
    engine = SimulationEngine(world, agents)
    run_gui(engine)

if __name__ == "__main__":
    main()