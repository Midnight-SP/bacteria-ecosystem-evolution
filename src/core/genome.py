import numpy as np

class Genome:
    GENE_MAP = {
        "initial_energy": 0,
        "max_age": 1,
        "color_red": 2,
        "color_green": 3,
        "color_blue": 4,
        "speed": 5,
    }

    def __init__(self, genes):
        self.genes = np.array(genes, dtype=np.uint8)

    @property
    def initial_energy(self):
        return int(50 + (self.genes[self.GENE_MAP["initial_energy"]] / 255) * 200)

    @property
    def max_age(self):
        return int(50 + (self.genes[self.GENE_MAP["max_age"]] / 255) * 150)

    @property
    def color(self):
        return (
            int(self.genes[self.GENE_MAP["color_red"]]),
            int(self.genes[self.GENE_MAP["color_green"]]),
            int(self.genes[self.GENE_MAP["color_blue"]])
        )

    @property
    def speed(self):
        return 1 + int((self.genes[self.GENE_MAP["speed"]] / 255) * 4)  # 1-5

    def mutate(self, rate=0.01):
        for i in range(len(self.genes)):
            if np.random.rand() < rate:
                self.genes[i] = np.random.randint(0, 256)

class BacteriaGenome:
    GENE_MAP = {
        "initial_energy": 0,
        "max_age": 1,
        "color_red": 2,
        "color_green": 3,
        "color_blue": 4,
        "speed": 5,
        "aggression": 6,
        "defense": 7,
    }
    def __init__(self, genes):
        self.genes = np.array(genes, dtype=np.uint8)

    @property
    def initial_energy(self):
        return int(50 + (self.genes[self.GENE_MAP["initial_energy"]] / 255) * 200)

    @property
    def max_age(self):
        return int(50 + (self.genes[self.GENE_MAP["max_age"]] / 255) * 150)

    @property
    def color(self):
        return (
            int(self.genes[self.GENE_MAP["color_red"]]),
            int(self.genes[self.GENE_MAP["color_green"]]),
            int(self.genes[self.GENE_MAP["color_blue"]])
        )

    @property
    def speed(self):
        return 1 + int((self.genes[self.GENE_MAP["speed"]] / 255) * 4)  # 1-5

    @property
    def aggression(self):
        return int(self.genes[self.GENE_MAP["aggression"]] / 255 * 100)  # 0-100

    @property
    def defense(self) -> float:
        # prawdopodobieństwo obrony [0.0–1.0]
        return int(self.genes[self.GENE_MAP["defense"]]) / 255.0

    def mutate(self, rate=0.01):
        for i in range(len(self.genes)):
            if np.random.rand() < rate:
                self.genes[i] = np.random.randint(0, 256)

class AlgaeGenome:
    GENE_MAP = {
        "initial_energy": 0,
        "max_age": 1,
        "color_red": 2,
        "color_green": 3,
        "color_blue": 4,
        "photosynthesis": 5,
        "defense": 6,
    }
    def __init__(self, genes):
        self.genes = np.array(genes, dtype=np.uint8)

    @property
    def initial_energy(self):
        return int(50 + (self.genes[self.GENE_MAP["initial_energy"]] / 255) * 200)

    @property
    def max_age(self):
        return int(50 + (self.genes[self.GENE_MAP["max_age"]] / 255) * 150)

    @property
    def color(self):
        return (
            int(self.genes[self.GENE_MAP["color_red"]]),
            int(self.genes[self.GENE_MAP["color_green"]]),
            int(self.genes[self.GENE_MAP["color_blue"]])
        )

    @property
    def photosynthesis(self):
        return int(self.genes[self.GENE_MAP["photosynthesis"]] / 255 * 100)  # 0-100
    
    @property
    def defense(self) -> float:
        return int(self.genes[self.GENE_MAP["defense"]]) / 255.0

    def mutate(self, rate=0.01):
        for i in range(len(self.genes)):
            if np.random.rand() < rate:
                self.genes[i] = np.random.randint(0, 256)

class FungiGenome:
    GENE_MAP = {
        "initial_energy": 0,
        "max_age": 1,
        "color_red": 2,
        "color_green": 3,
        "color_blue": 4,
        "spore_spread": 5,
        "defense": 6,
    }
    def __init__(self, genes):
        self.genes = np.array(genes, dtype=np.uint8)

    @property
    def initial_energy(self):
        return int(50 + (self.genes[self.GENE_MAP["initial_energy"]] / 255) * 200)

    @property
    def max_age(self):
        return int(50 + (self.genes[self.GENE_MAP["max_age"]] / 255) * 150)

    @property
    def color(self):
        return (
            int(self.genes[self.GENE_MAP["color_red"]]),
            int(self.genes[self.GENE_MAP["color_green"]]),
            int(self.genes[self.GENE_MAP["color_blue"]])
        )

    @property
    def spore_spread(self):
        return int(self.genes[self.GENE_MAP["spore_spread"]] / 255 * 100)  # 0-100
    
    @property
    def defense(self) -> float:
        return int(self.genes[self.GENE_MAP["defense"]]) / 255.0

    def mutate(self, rate=0.01):
        for i in range(len(self.genes)):
            if np.random.rand() < rate:
                self.genes[i] = np.random.randint(0, 256)


class ProtozoaGenome:
    GENE_MAP = {
        "initial_energy": 0,
        "max_age": 1,
        "color_red": 2,
        "color_green": 3,
        "color_blue": 4,
        "aggression": 5,
        "speed": 6,
        "defense": 7,
    }
    def __init__(self, genes):
        self.genes = np.array(genes, dtype=np.uint8)

    @property
    def initial_energy(self):
        return int(50 + (self.genes[self.GENE_MAP["initial_energy"]] / 255) * 200)

    @property
    def max_age(self):
        return int(50 + (self.genes[self.GENE_MAP["max_age"]] / 255) * 150)

    @property
    def color(self):
        return (
            int(self.genes[self.GENE_MAP["color_red"]]),
            int(self.genes[self.GENE_MAP["color_green"]]),
            int(self.genes[self.GENE_MAP["color_blue"]])
        )

    @property
    def aggression(self):
        return int(self.genes[self.GENE_MAP["aggression"]] / 255 * 100)

    @property
    def speed(self):
        return 1 + int((self.genes[self.GENE_MAP["speed"]] / 255) * 4)
    
    @property
    def defense(self) -> float:
        return int(self.genes[self.GENE_MAP["defense"]]) / 255.0

    def mutate(self, rate=0.01):
        for i in range(len(self.genes)):
            if np.random.rand() < rate:
                self.genes[i] = np.random.randint(0, 256)