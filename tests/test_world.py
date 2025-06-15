import unittest
from src.environment.world import World

class TestWorld(unittest.TestCase):
    def test_world_init(self):
        world = World(10, 10)
        self.assertEqual(world.width, 10)
        self.assertEqual(world.height, 10)
        self.assertEqual(len(world.grid), 10)
        self.assertEqual(len(world.grid[0]), 10)

if __name__ == "__main__":
    unittest.main()