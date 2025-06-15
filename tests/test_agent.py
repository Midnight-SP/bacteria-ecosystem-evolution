import unittest
import numpy as np
from src.core.agent import Agent

class DummyAgent(Agent):
    def act(self, environment):
        return None

class TestAgent(unittest.TestCase):
    def test_agent_init(self):
        genome = np.array([1, 2, 3], dtype=np.uint8)
        class DummyNN:
            def __init__(self): pass
        agent = DummyAgent(genome, DummyNN(), (0, 0))
        self.assertEqual(agent.position, (0, 0))
        self.assertTrue(agent.is_alive)

    def test_agent_act(self):
        genome = np.array([1, 2, 3], dtype=np.uint8)
        class DummyNN:
            def __init__(self): pass
        agent = DummyAgent(genome, DummyNN(), (0, 0))
        self.assertIsNone(agent.act(None))

if __name__ == "__main__":
    unittest.main()