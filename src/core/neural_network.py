import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, output_size, weights=None):
        self.input_size = input_size
        self.output_size = output_size
        if weights is not None:
            self.weights = np.array(weights).reshape((input_size, output_size))
        else:
            self.weights = np.random.uniform(-1, 1, (input_size, output_size))

    def forward(self, inputs):
        x = np.dot(inputs, self.weights)
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def mutate(self, rate=0.01):
        mask = np.random.rand(*self.weights.shape) < rate
        self.weights[mask] += np.random.normal(0, 0.1, size=mask.sum())