class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(self.width)] for _ in range(self.height)]
    
    def update(self):
        pass # Placeholder for world update logic

    def consoleRender(self):
        output = ""
        for row in self.grid:
            output += " ".join(str(cell) if cell is not None else "." for cell in row) + "\n"
        return output.strip()