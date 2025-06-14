import numpy as np
import random

class NeuralNetwork:
    def __init__(self, input_size, output_size, genome=None, mutation_rate=0.05):
        self.input_size = input_size
        self.output_size = output_size
        self.mutation_rate = mutation_rate

        if genome is not None:
            # Zamień uint8 na float w zakresie -1..1
            genome = np.array(genome, dtype=np.uint8)
            weights = (genome.astype(np.float32) / 255.0) * 2 - 1
            self.weights = weights.reshape((input_size, output_size))
        else:
            self.weights = np.random.uniform(-1, 1, (input_size, output_size))

    def forward(self, inputs):
        # inputs: numpy array o rozmiarze input_size
        x = np.dot(inputs, self.weights)
        # Softmax dla wyjścia akcji
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def get_genome(self):
        # Zwraca spłaszczony genom (listę wag)
        return self.weights.flatten().tolist()

    def mutate(self):
        # Mutuje wagi z zadanym prawdopodobieństwem
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                if random.random() < self.mutation_rate:
                    self.weights[i, j] += np.random.normal(0, 0.1)

    @classmethod
    def crossover(cls, nn1, nn2, mutation_rate=0.05):
        genome1 = nn1.get_genome()
        genome2 = nn2.get_genome()
        split = len(genome1) // 2
        new_genome = genome1[:split] + genome2[split:]
        # Mutacja genomu
        for i in range(len(new_genome)):
            if random.random() < mutation_rate:
                new_val = new_genome[i] + np.random.normal(0, 0.1)
            else:
                new_val = new_genome[i]
            # Wymuś zakres i typ int
            new_val = int(round(new_val))
            new_val = max(0, min(255, new_val))
            new_genome[i] = new_val
        # Teraz new_genome jest listą intów w zakresie 0–255
        return cls(nn1.input_size, nn1.output_size, genome=new_genome, mutation_rate=mutation_rate)