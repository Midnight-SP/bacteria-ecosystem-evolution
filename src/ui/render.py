from PyQt5.QtGui import QImage, QColor

def render_world(world):
    """
    Renderuje świat jako QImage (do wyświetlenia w GUI).
    """
    width, height = world.width, world.height
    img = QImage(width, height, QImage.Format_RGB32)
    for y in range(height):
        for x in range(width):
            cell = world.grid[y][x]
            # Wyświetl tylko żywych agentów
            if cell is not None and hasattr(cell, "color") and getattr(cell, "is_alive", True):
                color = QColor(*cell.color)
            else:
                color = QColor(30, 30, 30)
            img.setPixel(x, y, color.rgb())
    return img