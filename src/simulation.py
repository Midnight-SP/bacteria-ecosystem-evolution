# Minimalny szkic, niepełny kod!
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QSlider, QFileDialog, QLabel, QGraphicsScene, QGraphicsView, QGraphicsRectItem
from PyQt5.QtCore import QTimer, QRectF, Qt
from PyQt5.QtGui import QColor, QBrush, QPen
import pickle
import numpy as np
import logging

from environment.world import World
from agents.bacteria import Bacteria
from agents.protist import Protist

CELL_SIZE = 8
WORLD_WIDTH = 100
WORLD_HEIGHT = 100

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def get_environment_state(world):
    """
    Zwraca słownik {(x, y): {'type': 'cell', 'object': cell, 'pheromone': ...}, ...}
    """
    state = {}
    for y in range(world.height):
        for x in range(world.width):
            cell = world.grid[y][x]
            if cell is not None:
                state[(x, y)] = {'type': 'cell', 'object': cell, 'pheromone': getattr(cell, 'pheromone_strength', 0)}
    return state

class SimulationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Symulacja bakterii")
        self.simulation_running = False
        self.simulation_speed = 200  # ms

        # Świat i populacja
        self.world = World(WORLD_WIDTH, WORLD_HEIGHT)
        # Ensure grid is a mutable list of lists that can hold any object
        if not hasattr(self.world, "grid") or not isinstance(self.world.grid, list):
            self.world.grid = [[None for _ in range(WORLD_WIDTH)] for _ in range(WORLD_HEIGHT)]
        self.init_population()

        # Przyciski i GUI
        self.start_btn = QPushButton("Start")
        self.pause_btn = QPushButton("Pauza")
        self.save_btn = QPushButton("Zapisz")
        self.load_btn = QPushButton("Wczytaj")
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(1000)
        self.speed_slider.setValue(self.simulation_speed)
        self.speed_label = QLabel("Szybkość")
        self.stats_label = QLabel()

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.speed_label)
        btn_layout.addWidget(self.speed_slider)

        # Plansza (QGraphicsView)
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setFixedSize(WORLD_WIDTH * CELL_SIZE + 2, WORLD_HEIGHT * CELL_SIZE + 2)

        main_layout = QVBoxLayout()
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.view)
        main_layout.addWidget(self.stats_label)
        self.setLayout(main_layout)

        # Timer do symulacji
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_step)

        # Obsługa przycisków
        self.start_btn.clicked.connect(self.start_simulation)
        self.pause_btn.clicked.connect(self.pause_simulation)
        self.save_btn.clicked.connect(self.save_simulation)
        self.load_btn.clicked.connect(self.load_simulation)
        self.speed_slider.valueChanged.connect(self.change_speed)

        self.render_world()

    def init_population(self):
        population_genomes = []
        for _ in range(5):
            x, y = np.random.randint(0, WORLD_WIDTH), np.random.randint(0, WORLD_HEIGHT)
            while self.world.grid[y][x] is not None:
                x, y = np.random.randint(0, WORLD_WIDTH), np.random.randint(0, WORLD_HEIGHT)
            # 22 geny + 28*5 wag NN = 162
            genome = np.random.randint(-128, 128, 22 + 28*5)
            population_genomes.append(genome)
            b = Bacteria(x, y, genome=genome, population_genomes=population_genomes)
            self.world.grid[y][x] = b
        for _ in range(3):
            x, y = np.random.randint(0, WORLD_WIDTH), np.random.randint(0, WORLD_HEIGHT)
            while self.world.grid[y][x] is not None:
                x, y = np.random.randint(0, WORLD_WIDTH), np.random.randint(0, WORLD_HEIGHT)
            genome = np.random.randint(-128, 128, 15)
            population_genomes.append(genome)
            p = Protist(x, y, genome=genome, population_genomes=population_genomes)
            self.world.grid[y][x] = p

    def simulation_step(self):
        new_cells = []
        to_remove = []
        for y in range(self.world.height):
            for x in range(self.world.width):
                cell = self.world.grid[y][x]
                if cell is not None:
                    cell.vitals()
                    if cell.is_alive:
                        environment_state = get_environment_state(self.world)
                        # --- poprawka tu ---
                        if isinstance(cell, Bacteria):
                            result = cell.act(environment_state, world=self.world)
                        else:
                            result = cell.act(environment_state)
                        # --- reszta bez zmian ---
                        logging.debug(f"Cell at ({x},{y}) acted: {type(cell).__name__}, energy={cell.energy}, age={cell.age}")
                        if (
                            isinstance(result, list)
                            and all(isinstance(r, tuple) and len(r) == 2 and hasattr(r[0], "x") and hasattr(r[0], "y") for r in result)
                        ):
                            for child, (cx, cy) in result:
                                child_name = getattr(child, "name", "unknown")
                                logging.info(f"Cell at ({x},{y}) divided, child at ({cx},{cy}) named {child_name}")
                                if 0 <= cx < self.world.width and 0 <= cy < self.world.height:
                                    new_cells.append((child, cx, cy))
                        elif not cell.is_alive or cell.energy <= 0 or cell.age > cell.max_age:
                            logging.info(f"Cell at ({x},{y}) died ({type(cell).__name__}, energy={cell.energy}, age={cell.age}, name={cell.name})")
                            to_remove.append((x, y))
        # Usuń martwe/zużyte komórki
        for x, y in to_remove:
            self.world.grid[y][x] = None
        # Dodaj nowe komórki
        for cell, x, y in new_cells:
            cell_name = getattr(cell, "name", "unknown")
            logging.info(f"New cell placed at ({x},{y}): {cell_name}")
            if self.world.grid[y][x] is None:
                self.world.grid[y][x] = cell
        self.render_world()

    def render_world(self):
        self.scene.clear()
        bacteria_count = 0
        protist_count = 0
        for y in range(self.world.height):
            for x in range(self.world.width):
                cell = self.world.grid[y][x]
                rect = QGraphicsRectItem(QRectF(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                rect.setPen(QPen(QColor(0, 0, 0)))
                if cell is not None:
                    # Kolor wg genomu lub typu
                    if hasattr(cell, "color"):
                        r, g, b = cell.color
                        rect.setBrush(QBrush(QColor(r, g, b)))
                    elif isinstance(cell, Bacteria):
                        rect.setBrush(QBrush(QColor(255, 255, 0)))  # żółty
                        bacteria_count += 1
                    elif isinstance(cell, Protist):
                        rect.setBrush(QBrush(QColor(0, 255, 255)))  # cyjan
                        protist_count += 1
                self.scene.addItem(rect)
        self.stats_label.setText(f"Bakterie: {bacteria_count} | Protisty: {protist_count}")

    def start_simulation(self):
        self.simulation_running = True
        self.timer.start(self.simulation_speed)

    def pause_simulation(self):
        self.simulation_running = False
        self.timer.stop()

    def save_simulation(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Zapisz symulację", "", "Pliki symulacji (*.sim)")
        if filename:
            with open(filename, "wb") as f:
                pickle.dump(self.world, f)

    def load_simulation(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Wczytaj symulację", "", "Pliki symulacji (*.sim)")
        if filename:
            with open(filename, "rb") as f:
                self.world = pickle.load(f)
            self.render_world()

    def change_speed(self, value):
        self.simulation_speed = value
        if self.simulation_running:
            self.timer.start(self.simulation_speed)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = SimulationWindow()
    window.show()
    sys.exit(app.exec_())