import unittest
import numpy as np
from src.core.evolution import uniform_crossover, mutate_genome

class TestEvolution(unittest.TestCase):
    def test_uniform_crossover(self):
        g1 = np.array([1, 2, 3, 4], dtype=np.uint8)
        g2 = np.array([5, 6, 7, 8], dtype=np.uint8)
        child = uniform_crossover(g1, g2)
        self.assertEqual(child.shape, g1.shape)
        for i in range(len(child)):
            self.assertIn(child[i], [g1[i], g2[i]])

    def test_mutate_genome(self):
        g = np.array([1, 2, 3, 4], dtype=np.uint8)
        mutated = mutate_genome(g, rate=1.0)
        self.assertFalse(np.all(mutated == g))

if __name__ == "__main__":
    unittest.main()