import pickle
import numpy as np

from environment.world import World
from agents.bacteria import Bacteria
from agents.protist import Protist
from neural_network.neural_network import NeuralNetwork

WORLD_WIDTH = 100
WORLD_HEIGHT = 100

def make_handcrafted_protist_nn():
    weights = np.array([
        [-2.0,  4.0],   # energy
        [-4.0,  0.0],   # age
        [ 0.0,  0.0],   # health
        [ 0.0,  0.0],   # chemotaxis
        [ 0.0, -2.0],   # cooldown
    ])
    nn = NeuralNetwork(input_size=5, output_size=2)
    nn.weights = weights
    return nn

def make_handcrafted_herbivore_bacteria_nn():
    # 28 wejść, 5 wyjść
    weights = np.zeros((28, 5))

    # Wzrok: 8 kierunków, co 2 wejścia (czy jest komórka, czy jest feromon)
    # Kierunek 0 (przed siebie): sight[0] (czy jest komórka)
    # Kierunek 0 (przed siebie): sight[1] (czy jest feromon)
    # Kierunki: 0,2,4,6,8,10,12,14 (czy jest komórka)
    # Feromony: 1,3,5,7,9,11,13,15

    # Jeśli przed sobą jest komórka (potencjalny protist) → jedz
    weights[5 + 0, 4] = 5.0  # sight[0] (czy jest komórka przed) → eat

    # Jeśli w innych kierunkach jest komórka → skręć
    for i in range(1, 8):
        weights[5 + i*2, 3] = 2.0  # sight[i*2] (czy jest komórka) → turn

    # Jeśli nigdzie nie widzi komórki → idź do przodu
    weights[2, 2] = 1.0  # health (zawsze trochę motywuje do ruchu)

    # Jeśli przed sobą jest feromon → rozmnażaj się
    weights[5 + 1, 1] = 3.0  # sight[1] (feromon przed) → reproduce

    # Jeśli gdziekolwiek jest feromon → rozsyłaj feromony (sygnał do kolonii)
    for i in range(0, 8):
        weights[5 + i*2 + 1, 0] = 1.0  # sight feromony → pheromones

    # Chemotaksja (jeśli dużo feromonów) → rozmnażaj się
    weights[3, 1] = 2.0  # chemotaxis → reproduce

    # Trochę biasu do ruchu do przodu
    weights[0, 2] = 0.5  # energy → move forward

    nn = NeuralNetwork(input_size=28, output_size=5)
    nn.weights = weights
    return nn

def make_handcrafted_carnivore_bacteria_nn():
    weights = np.zeros((28, 5))

    # 0: pheromones, 1: reproduce, 2: move, 3: turn, 4: eat

    # Jeśli przed sobą jest komórka (potencjalna ofiara) → jedz
    weights[5 + 0, 4] = 5.0  # sight[0] (czy jest komórka przed) → eat

    # Jeśli w innych kierunkach jest komórka → skręć i próbuj gryźć
    for i in range(1, 8):
        weights[5 + i*2, 3] = 2.0  # sight[i*2] (czy jest komórka) → turn
        weights[5 + i*2, 4] = 1.0  # sight[i*2] (czy jest komórka) → eat

    # Jeśli widzi ofiarę (gdziekolwiek) → rozsyłaj feromony
    for i in range(0, 8):
        weights[5 + i*2, 0] = 2.0  # sight[i*2] (czy jest komórka) → pheromones

    # Jeśli przed sobą jest feromon → skręć i próbuj gryźć
    weights[5 + 1, 3] = 2.0  # sight[1] (feromon przed) → turn
    weights[5 + 1, 4] = 1.0  # sight[1] (feromon przed) → eat

    # Jeśli gdziekolwiek jest feromon → skręć
    for i in range(0, 8):
        weights[5 + i*2 + 1, 3] = 1.0  # sight feromony → turn

    # Chemotaksja (wyczuwa dużo feromonów) → skręć i próbuj gryźć
    weights[3, 3] = 2.0  # chemotaxis → turn
    weights[3, 4] = 1.0  # chemotaxis → eat

    # Dużo energii → rozmnażaj się
    weights[0, 1] = 4.0  # energy → reproduce

    # Domyślnie idź do przodu
    weights[2, 2] = 1.0  # health → move forward

    nn = NeuralNetwork(input_size=28, output_size=5)
    nn.weights = weights
    return nn

def make_protist(x, y, genome):
    nn = make_handcrafted_protist_nn()
    return Protist(x, y, genome=genome, nn=nn, population_genomes=[genome]*10)

def make_bacteria(x, y, genome):
    nn = make_handcrafted_herbivore_bacteria_nn()
    return Bacteria(x, y, genome=genome, nn=nn, population_genomes=[genome]*10)

def make_carnivore_bacteria(x, y, genome):
    nn = make_handcrafted_carnivore_bacteria_nn()
    return Bacteria(x, y, genome=genome, nn=nn, population_genomes=[genome]*10)

def main():
    world = World(WORLD_WIDTH, WORLD_HEIGHT)

    # Pełne genomy
    herb_genome = np.array([100,100,5,20,100,50,200,50,1,10,50,2,2,5,0,1,1,240,0,0,1,0], dtype=np.uint8)
    herb_genome = np.concatenate([herb_genome, np.random.randint(0, 256, 28*5, dtype=np.uint8)])
    carn_genome = np.array([100,100,5,20,100,200,50,50,1,5,50,2,2,5,0,1,1,120,255,0,1,10], dtype=np.uint8)
    carn_genome = np.concatenate([carn_genome, np.random.randint(0, 256, 28*5, dtype=np.uint8)])
    protist1_genome = np.array([100,100,5,20,100,255,0,0,1,10,80,2,2,5,0], dtype=np.uint8)
    protist2_genome = np.array([100,100,5,20,100,0,255,0,1,10,80,2,2,5,0], dtype=np.uint8)

    positions = set()
    rng = np.random.default_rng(42)
    while len(positions) < 40:
        x, y = rng.integers(0, WORLD_WIDTH), rng.integers(0, WORLD_HEIGHT)
        positions.add((x, y))
    positions = list(positions)

    for i in range(10):
        x, y = positions[i]
        world.grid[y][x] = make_bacteria(x, y, herb_genome)
    for i in range(10, 20):
        x, y = positions[i]
        world.grid[y][x] = make_carnivore_bacteria(x, y, carn_genome)
    for i in range(20, 30):
        x, y = positions[i]
        world.grid[y][x] = make_protist(x, y, protist1_genome)
    for i in range(30, 40):
        x, y = positions[i]
        world.grid[y][x] = make_protist(x, y, protist2_genome)

    with open("test_world.sim", "wb") as f:
        pickle.dump(world, f)
    print("Zapisano testowy świat do test_world.sim")

if __name__ == "__main__":
    main()