import unittest
from agents.bacteria import Bacteria

class TestBacteria(unittest.TestCase):
    def test_initialization(self):
        b = Bacteria(0, 0)
        self.assertTrue(b.is_alive)
        self.assertGreaterEqual(b.energy, 0)