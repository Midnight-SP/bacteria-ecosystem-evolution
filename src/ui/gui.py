import sys
import pickle
import json
import numpy as np
import graphviz
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QHBoxLayout,
    QPushButton, QFileDialog, QSpinBox, QTreeWidget, QTreeWidgetItem
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap
from src.ui.render import render_world

class MainWindow(QMainWindow):
    def __init__(self, simulation_engine):
        super().__init__()
        self.simulation_engine = simulation_engine
        self.setWindowTitle("Bacteria Ecosystem Evolution")
        self.paused = False

        # Domyślne rozmiary okna symulacji
        self.display_width = 400
        self.display_height = 400

        # Layouty
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout()
        self.controls_layout = QHBoxLayout()

        # Kontrolki
        self.label = QLabel("Symulacja ekosystemu")
        self.main_layout.addWidget(self.label)

        # Przycisk pauzy/wznawiania
        self.pause_button = QPushButton("Pauza")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.controls_layout.addWidget(self.pause_button)

        # Przycisk zapisu
        self.save_button = QPushButton("Zapisz")
        self.save_button.clicked.connect(self.save_simulation)
        self.controls_layout.addWidget(self.save_button)

        # Przycisk wczytywania
        self.load_button = QPushButton("Wczytaj")
        self.load_button.clicked.connect(self.load_simulation)
        self.controls_layout.addWidget(self.load_button)

        # Zmiana rozmiaru okna symulacji
        self.display_width_spin = QSpinBox()
        self.display_width_spin.setRange(100, 2000)
        self.display_width_spin.setValue(self.display_width)
        self.controls_layout.addWidget(QLabel("Szerokość okna:"))
        self.controls_layout.addWidget(self.display_width_spin)

        self.display_height_spin = QSpinBox()
        self.display_height_spin.setRange(100, 2000)
        self.display_height_spin.setValue(self.display_height)
        self.controls_layout.addWidget(QLabel("Wysokość okna:"))
        self.controls_layout.addWidget(self.display_height_spin)

        self.resize_display_button = QPushButton("Zmień rozmiar okna")
        self.resize_display_button.clicked.connect(self.change_display_size)
        self.controls_layout.addWidget(self.resize_display_button)

        # Przycisk eksportu drzewa
        self.export_tree_button = QPushButton("Eksportuj drzewo (JSON)")
        self.export_tree_button.clicked.connect(self.export_genealogy_tree)
        self.controls_layout.addWidget(self.export_tree_button)

        # Przycisk eksportu drzewa jako obraz
        self.export_tree_img_button = QPushButton("Eksportuj drzewo (PNG)")
        self.export_tree_img_button.clicked.connect(self.export_genealogy_tree_image)
        self.controls_layout.addWidget(self.export_tree_img_button)

        # Przycisk eksportu drzewa agentów
        self.export_agent_tree_button = QPushButton("Eksportuj drzewo agentów (JSON)")
        self.export_agent_tree_button.clicked.connect(self.export_agent_genealogy_tree)
        self.controls_layout.addWidget(self.export_agent_tree_button)

        # Przycisk eksportu drzewa agentów jako obraz
        self.export_agent_tree_img_button = QPushButton("Eksportuj drzewo agentów (PNG)")
        self.export_agent_tree_img_button.clicked.connect(self.export_agent_genealogy_tree_image)
        self.controls_layout.addWidget(self.export_agent_tree_img_button)

        self.main_layout.addLayout(self.controls_layout)

        # Obraz symulacji
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.image_label)

        # Dodaj drzewo genealogiczne
        self.genealogy_tree = QTreeWidget()
        self.genealogy_tree.setHeaderLabel("Drzewo genealogiczne")
        self.main_layout.addWidget(self.genealogy_tree)

        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

        # Timer do symulacji
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(100)  # 10 FPS

    def update_simulation(self):
        if not self.paused:
            self.simulation_engine.step()
        img = render_world(self.simulation_engine.world)
        pixmap = QPixmap.fromImage(img)
        # Skalowanie do rozmiaru okna
        pixmap = pixmap.scaled(self.display_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        self.update_genealogy_tree()

    def update_genealogy_tree(self):
        self.genealogy_tree.clear()
        species_tree = self.simulation_engine.species_tree
        species_counts = self.simulation_engine.species_counts

        # Znajdź korzenie (gatunki bez rodziców)
        all_species = set(species_tree.keys())
        all_parents = set()
        for parents in species_tree.values():
            all_parents.update(parents)
        roots = list(all_species - all_parents)

        # Mapuj species_name na typ agenta (na podstawie żywych agentów)
        species_types = {}
        for agent in self.simulation_engine.agents:
            if agent.is_alive:
                population_genomes = [a.genome.genes for a in self.simulation_engine.agents if a.agent_type == agent.agent_type and a.is_alive]
                species_name = agent.get_species_name(population_genomes)
                species_types[species_name] = getattr(agent, "agent_type", type(agent).__name__)

        def add_species_node(species_name, parent_item=None):
            count = species_counts.get(species_name, 0)
            agent_type = species_types.get(species_name, "?")
            label = f"{species_name} ({count})"
            item = QTreeWidgetItem([label])
            if parent_item:
                parent_item.addChild(item)
            else:
                self.genealogy_tree.addTopLevelItem(item)
            # Dodaj dzieci
            children = [s for s, parents in species_tree.items() if species_name in parents]
            for child in children:
                add_species_node(child, item)

        for root in roots:
            add_species_node(root)

        self.genealogy_tree.expandAll()

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.setText("Wznów" if self.paused else "Pauza")

    def save_simulation(self):
        path, _ = QFileDialog.getSaveFileName(self, "Zapisz symulację", "", "Pliki symulacji (*.sim)")
        if path:
            with open(path, "wb") as f:
                pickle.dump(self.simulation_engine, f)

    def load_simulation(self):
        path, _ = QFileDialog.getOpenFileName(self, "Wczytaj symulację", "", "Pliki symulacji (*.sim)")
        if path:
            with open(path, "rb") as f:
                self.simulation_engine = pickle.load(f)
            self.update_simulation()

    def change_display_size(self):
        self.display_width = self.display_width_spin.value()
        self.display_height = self.display_height_spin.value()
        self.update_simulation()

    def export_genealogy_tree(self):
        path, _ = QFileDialog.getSaveFileName(self, "Zapisz drzewo genealogiczne", "", "Pliki JSON (*.json)")
        if path:
            # Przygotuj dane do eksportu
            data = {
                "species_tree": {k: list(v) for k, v in self.simulation_engine.species_tree.items()},
                "species_counts": self.simulation_engine.species_counts,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def export_genealogy_tree_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Zapisz drzewo genealogiczne", "", "Obrazy PNG (*.png)")
        if path:
            dot = graphviz.Digraph(comment="Genealogy Tree")
            species_tree = self.simulation_engine.species_tree
            species_counts = self.simulation_engine.species_counts
            # Dodaj węzły
            for species, parents in species_tree.items():
                label = f"{species}\n({species_counts.get(species, 0)})"
                dot.node(species, label)
                for parent in parents:
                    dot.edge(parent, species)
            dot.format = "png"
            dot.render(path, view=False, cleanup=True)

    def export_agent_genealogy_tree(self):
        path, _ = QFileDialog.getSaveFileName(self, "Zapisz drzewo agentów", "", "Pliki JSON (*.json)")
        if path:
            data = {}
            for agent in self.simulation_engine.agents:
                population_genomes = [a.genome.genes for a in self.simulation_engine.agents if a.agent_type == agent.agent_type and a.is_alive]
                data[agent.id] = {
                    "parent_ids": agent.parent_ids,
                    "species_name": agent.get_species_name(population_genomes),
                }
            genealogy = self.simulation_engine.genealogy
            export = {"agents": data, "genealogy": genealogy}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(export, f, ensure_ascii=False, indent=2)

    def export_agent_genealogy_tree_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Zapisz drzewo agentów", "", "Obrazy PNG (*.png)")
        if path:
            import graphviz
            dot = graphviz.Digraph(comment="Agent Genealogy Tree")
            agents = {a.id: a for a in self.simulation_engine.agents}
            genealogy = self.simulation_engine.genealogy
            # Dodaj węzły
            for agent_id, agent in agents.items():
                population_genomes = [a.genome.genes for a in agents.values() if a.agent_type == agent.agent_type and a.is_alive]
                label = f"{agent.get_species_name(population_genomes)}\nID:{agent_id}"
                dot.node(str(agent_id), label)
            # Dodaj krawędzie
            for child_id, parent_ids in genealogy.items():
                for pid in parent_ids:
                    if str(pid) in dot.body and str(child_id) in dot.body:
                        dot.edge(str(pid), str(child_id))
                    else:
                        dot.edge(str(pid), str(child_id))
            dot.format = "png"
            dot.render(path, view=False, cleanup=True)

def run_gui(simulation_engine):
    app = QApplication(sys.argv)
    window = MainWindow(simulation_engine)
    window.show()
    sys.exit(app.exec_())