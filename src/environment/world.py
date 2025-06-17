from src.environment.resources import ResourceField
from src.environment.pheromone import PheromoneField

class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.resources = ResourceField(width, height)
        # Dodaj typy feromonów
        self.pheromones = PheromoneField(width, height, pheromone_types=['food', 'danger', 'predator', 'algae'])

    def update(self):
        self.resources.regenerate(rate=2)
        self.pheromones.evaporate()