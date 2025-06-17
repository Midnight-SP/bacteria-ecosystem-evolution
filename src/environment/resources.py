import numpy as np

class ResourceField:
    """
    Reprezentuje pole zasobów (np. pożywienia, światła) w świecie.
    """
    def __init__(self, width, height, initial_value=400):
        self.width = width
        self.height = height
        # podniesione initial_value z 200 do 400
        self.grid = np.full((height, width), initial_value, dtype=float)

    def consume(self, x, y, amount):
        xi = x % self.width
        yi = y % self.height
        val = self.grid[yi, xi].item()
        consumed = min(val, amount)
        self.grid[yi, xi] -= consumed
        return consumed

    def regenerate(self, rate=2):
        # podwójne tempo regeneracji
        self.grid += rate
        # więcej i obficiej losowych plam
        for _ in range(5):       # zamiast 3, pięć plam
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            self.grid[y, x] += 100  # zamiast +50, +100

    def get(self, x, y):
        xi = x % self.width
        yi = y % self.height
        return self.grid[yi, xi]