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
    # Jeśli cooldown > 0, bardzo nie chcemy rozmnażać się
    weights[4, 1] = -100.0  # wejście cooldown → wyjście reproduce
    nn = NeuralNetwork(input_size=5, output_size=2)
    nn.weights = weights
    return nn

def make_handcrafted_herbivore_bacteria_nn():
    weights = np.zeros((30, 7))

    # Jeśli przed sobą jest komórka (potencjalny protist) → jedz
    weights[5 + 0, 5] = 8.0  # sight[0] (komórka przed) → eat

    # Jeśli w innych kierunkach jest komórka → skręć
    for i in range(1, 8):
        weights[5 + i*2, 3] = 1.5  # sight[i*2] (czy jest komórka) → turn

    # Jeśli nigdzie nie widzi komórki → idź do przodu
    weights[2, 2] = 2.0  # health (motywuje do ruchu)
    weights[0, 6] = 0.5  # energy → sprint (np. sprintuj przy wysokiej energii)

    # Jeśli przed sobą jest feromon → mocniej idź do przodu
    weights[5 + 1, 2] = 2.0  # sight[1] (feromon przed) → move

    # Chemotaksja (jeśli dużo feromonów) → lekko preferuj ruch do przodu
    weights[3, 2] = 0.5  # chemotaxis → move

    # Rozmnażanie tylko przy wysokiej energii
    weights[0, 1] = 1.5  # energy → reproduce
    weights[4, 1] = -100.0  # cooldown > 0 → nie rozmnażaj się

    # Rozsyłanie feromonów tylko przy obecności innych komórek
    for i in range(8):
        weights[5 + i*2, 0] = 0.3  # sight komórki → pheromones

    # Dodatkowe zachowania dla bakterii roślinożernych
    weights[5 + 1, 3] = 2.0  # sight[1] (feromon po lewej) → turn_left
    weights[5 + 3, 4] = 2.0  # sight[3] (feromon po prawej) → turn_right

    weights[28, 5] = 10.0  # can_eat_ahead → eat
    weights[29, 1] = 10.0  # can_reproduce_ahead → reproduce

    # Jeśli nie może jeść ani się rozmnażać, preferuj ruch i skręt
    weights[2, 2] = 5.0   # health → move
    weights[2, 3] = 2.0   # health → turn_left
    weights[2, 4] = 2.0   # health → turn_right

    weights[:, 2] += 0.2  # lekka preferencja do ruchu

    nn = NeuralNetwork(input_size=30, output_size=7)
    nn.weights = weights
    return nn

def make_handcrafted_carnivore_bacteria_nn():
    weights = np.zeros((30, 7))

    # Jeśli przed sobą jest komórka (potencjalna ofiara) → jedz
    weights[5 + 0, 5] = 8.0  # sight[0] (czy jest komórka przed) → eat

    # Jeśli w innych kierunkach jest komórka → skręć w jej stronę
    for i in range(1, 8):
        if i < 4:
            weights[5 + i*2, 3] = 2.0  # sight[i*2] (komórka po lewej) → turn_left
        else:
            weights[5 + i*2, 4] = 2.0  # sight[i*2] (komórka po prawej) → turn_right

    # Jeśli nigdzie nie widzi komórki → idź do przodu lub sprintuj
    weights[2, 2] = 2.0  # health → move
    weights[0, 6] = 1.0  # energy → sprint (przy wysokiej energii)

    # Jeśli widzi feromon → lekko skręć w jego stronę
    for i in range(8):
        if i < 4:
            weights[5 + i*2 + 1, 3] = 0.5  # sight feromony po lewej → turn_left
        else:
            weights[5 + i*2 + 1, 4] = 0.5  # sight feromony po prawej → turn_right

    # Chemotaksja (wyczuwa dużo feromonów) → idź do przodu
    weights[3, 2] = 1.0  # chemotaxis → move

    # Rozmnażanie tylko przy bardzo wysokiej energii
    weights[0, 1] = 1.0  # energy → reproduce
    weights[4, 1] = -100.0  # cooldown > 0 → nie rozmnażaj się

    # Rozsyłanie feromonów tylko przy obecności ofiary
    for i in range(8):
        weights[5 + i*2, 0] = 0.2  # sight komórki → pheromones

    weights[28, 5] = 10.0  # can_eat_ahead → eat
    weights[29, 1] = 10.0  # can_reproduce_ahead → reproduce

    # Jeśli nie może jeść ani się rozmnażać, preferuj ruch i skręt
    weights[2, 2] = 5.0   # health → move
    weights[2, 3] = 2.0   # health → turn_left
    weights[2, 4] = 2.0   # health → turn_right

    weights[:, 2] += 0.2  # lekka preferencja do ruchu

    nn = NeuralNetwork(input_size=30, output_size=7)
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

    # Pełne genomy (22 geny + eating_strength na końcu)
    herb_genome = np.array([
        100, 100, 5, 20, 120, 50, 200, 50, 1, 10, 50, 2, 2, 5, 0, 1, 1, 240, 0, 0, 1, 2, 8  # eating_strength=8
    ], dtype=np.uint8)
    herb_genome = np.concatenate([herb_genome, np.random.randint(0, 256, 28*5, dtype=np.uint8)])

    carn_genome = np.array([
        100, 100, 5, 20, 80, 200, 50, 50, 1, 5, 50, 2, 2, 5, 0, 1, 1, 120, 255, 0, 1, 12, 20  # eating_strength=20
    ], dtype=np.uint8)
    carn_genome = np.concatenate([carn_genome, np.random.randint(0, 256, 28*5, dtype=np.uint8)])

    protist1_genome = np.array([50,100,5,20,100,255,0,0,1,2,80,2,2,5,0], dtype=np.uint8)  # photosynthesis_rate=2
    protist2_genome = np.array([60,80,8,15,120,0,255,0,2,3,60,4,1,8,0], dtype=np.uint8)

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