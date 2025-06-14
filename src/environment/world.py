from typing import Any, Optional

class World:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid: list[list[Optional[Any]]] = [[None for _ in range(self.width)] for _ in range(self.height)]
    
    def update(self):
        pass # Placeholder for world update logic

    def consoleRender(self):
        output = ""
        for row in self.grid:
            output += " ".join(str(cell) if cell is not None else "." for cell in row) + "\n"
        return output.strip()