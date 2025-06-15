import numpy as np

class ResourceField:
    """
    Reprezentuje pole zasobów (np. pożywienia, światła) w świecie.
    """
    def __init__(self, width, height, initial_value=200):
        self.width = width
        self.height = height
        self.grid = np.full((height, width), initial_value, dtype=float)

    def consume(self, x, y, amount):
        xi = x % self.width
        yi = y % self.height
        val = self.grid[yi, xi].item()
        consumed = min(val, amount)
        self.grid[yi, xi] -= consumed
        return consumed

    def regenerate(self, rate=1):
        self.grid += rate
        # Losowe plamy zasobów
        for _ in range(3):  # liczba plam na turę
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            self.grid[y, x] += 50  # wielkość plamy

    def get(self, x, y):
        xi = x % self.width
        yi = y % self.height
        return self.grid[yi, xi]