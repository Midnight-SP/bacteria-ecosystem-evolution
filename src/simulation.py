import random
from agents.bacteria import Bacteria
from agents.protist import Protist
from environment.world import World

def create_initial_population(world, num_bacteria=5, num_protists=3):
    # Dodaj bakterie w losowych miejscach
    for i in range(num_bacteria):
        x, y = random.randint(0, world.width-1), random.randint(0, world.height-1)
        b = Bacteria(x, y, name=f"Bac_{i}")
        world.grid[y][x] = b
    # Dodaj protisty w losowych miejscach
    for i in range(num_protists):
        x, y = random.randint(0, world.width-1), random.randint(0, world.height-1)
        # Unikaj nadpisania bakterii
        while world.grid[y][x] is not None:
            x, y = random.randint(0, world.width-1), random.randint(0, world.height-1)
        p = Protist(x, y, name=f"Prot_{i}")
        world.grid[y][x] = p

def get_environment_state(world):
    # Zwraca słownik {(x, y): info} dla wszystkich pól
    state = {}
    for y in range(world.height):
        for x in range(world.width):
            cell = world.grid[y][x]
            if cell is not None:
                state[(x, y)] = {"type": "cell", "object": cell}
            # Możesz dodać obsługę feromonów itp.
    return state

def step(world):
    # Każda komórka wykonuje akcję
    environment_state = get_environment_state(world)
    for y in range(world.height):
        for x in range(world.width):
            cell = world.grid[y][x]
            if cell is not None and hasattr(cell, "act"):
                result = cell.act(environment_state)
                # Obsłuż podział bakterii (dodaj dzieci do świata)
                if isinstance(result, list):
                    for child, pos in result:
                        cx, cy = pos
                        if 0 <= cx < world.width and 0 <= cy < world.height:
                            if world.grid[cy][cx] is None:
                                world.grid[cy][cx] = child

def main():
    width, height = 10, 6
    world = World(width, height)
    create_initial_population(world, num_bacteria=5, num_protists=3)

    for epoch in range(100):
        print(f"--- Turn {epoch} ---")
        print(world.consoleRender())
        step(world)

if __name__ == "__main__":
    main()