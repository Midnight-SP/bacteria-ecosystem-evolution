import sys
import pickle
import json
import numpy as np
import graphviz
from collections import Counter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QHBoxLayout,
    QPushButton, QFileDialog, QSpinBox, QTreeWidget, QTreeWidgetItem,
    QMessageBox
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

        # Przycisk eksportu drzewa gatunków
        self.export_tree_button = QPushButton("Eksportuj drzewo (JSON)")
        self.export_tree_button.clicked.connect(self.export_genealogy_tree)
        self.controls_layout.addWidget(self.export_tree_button)

        # Przycisk eksportu drzewa gatunków jako obraz
        self.export_tree_img_button = QPushButton("Eksportuj drzewo (PNG)")
        self.export_tree_img_button.clicked.connect(self.export_genealogy_tree_image)
        self.controls_layout.addWidget(self.export_tree_img_button)

        # Przycisk eksportu genotypów founderów
        self.export_founders_button = QPushButton("Eksportuj genotypy founderów")
        self.export_founders_button.clicked.connect(self.export_founders_genotypes)
        self.controls_layout.addWidget(self.export_founders_button)

        # nowa etykieta ze statystykami
        self.stats_label = QLabel("Steps: 0 | Bacteria: 0 Algae: 0 Fungi: 0 Protozoa: 0")
        self.controls_layout.addWidget(self.stats_label)

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
        self.timer.start(50)

        self.tree_needs_update = True  # Dodaj flagę

    def update_simulation(self):
        if not self.paused:
            self.simulation_engine.step()
            self.tree_needs_update = True
        # zaktualizuj licznik
        self._update_stats_label()

        img = render_world(self.simulation_engine.world)
        pixmap = QPixmap.fromImage(img)
        pixmap = pixmap.scaled(self.display_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        if self.tree_needs_update:
            self.update_genealogy_tree()
            self.tree_needs_update = False

    def update_genealogy_tree(self):
        self.genealogy_tree.clear()
        families = {}
        for agent in self.simulation_engine.agents:
            families.setdefault(agent.founder_id, []).append(agent)

        founder_names = self.simulation_engine.founder_names
        founder_parents = self.simulation_engine.founder_parents

        roots = [fid for fid in families if founder_parents.get(fid) is None]

        def count_descendants(founder_id):
            """Zlicza wszystkich członków podrodzin (rekurencyjnie)."""
            children = [fid for fid, parent in founder_parents.items() if parent == founder_id and fid in families]
            total = 0
            for child_id in children:
                total += len(families[child_id])
                total += count_descendants(child_id)
            return total

        def add_family_node(founder_id, parent_item=None):
            count = len(families[founder_id])
            descendants = count_descendants(founder_id)
            if founder_id in founder_names:
                label = f"{founder_names[founder_id]} ({count}), dzieci: {descendants}"
            else:
                agent = families[founder_id][0]
                pop_genomes = self.simulation_engine.get_population_genomes(agent.agent_type)
                latin_name = agent.generate_founder_species_name(pop_genomes)
                label = f"{latin_name} ({count}), dzieci: {descendants}"
                founder_names[founder_id] = latin_name
            item = QTreeWidgetItem([label])
            if parent_item:
                parent_item.addChild(item)
            else:
                self.genealogy_tree.addTopLevelItem(item)
            children = [fid for fid, parent in founder_parents.items() if parent == founder_id and fid in families]
            for child_id in children:
                add_family_node(child_id, item)

        for root_id in roots:
            add_family_node(root_id)

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
            founder_names = self.simulation_engine.founder_names
            founder_parents = self.simulation_engine.founder_parents
            agents_by_founder = self.simulation_engine.agents_by_founder

            # Funkcja do liczenia wszystkich potomków (rekurencyjnie)
            def count_descendants(founder_id):
                children = [fid for fid, parent in founder_parents.items() if parent == founder_id and fid in agents_by_founder]
                total = 0
                for child_id in children:
                    total += len(agents_by_founder[child_id])
                    total += count_descendants(child_id)
                return total

            # Dodaj węzły
            for founder_id, name in founder_names.items():
                count = len(agents_by_founder.get(founder_id, []))
                descendants = count_descendants(founder_id)
                label = f"{name}\n({count} członków, dzieci: {descendants})"
                dot.node(str(founder_id), label)

            # Dodaj krawędzie
            for founder_id, parent_id in founder_parents.items():
                if parent_id is not None and founder_id in founder_names and parent_id in founder_names:
                    dot.edge(str(parent_id), str(founder_id))

            dot.format = "png"
            dot.render(path, view=False, cleanup=True)

    def export_founders_genotypes(self):
        path, _ = QFileDialog.getSaveFileName(self, "Zapisz genotypy founderów", "", "Pliki CSV (*.csv)")
        if not path:
            return

        # Zbierz founderów rodzin z żyjącymi członkami
        engine = self.simulation_engine
        living_founders = set(a.founder_id for a in engine.agents if a.is_alive)
        rows = []
        for founder_id in living_founders:
            genome = engine.founders.get(founder_id)
            name = engine.founder_names.get(founder_id, str(founder_id))
            if genome is not None:
                # Zamień genom na listę liczb
                genome_list = list(map(int, genome))
                rows.append({
                    "founder_id": founder_id,
                    "name": name,
                    "genome": genome_list
                })

        # Zapisz do CSV
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["founder_id", "name", "genome"])
            writer.writeheader()
            for row in rows:
                # Zapisz genom jako string
                row_out = row.copy()
                row_out["genome"] = ";".join(map(str, row["genome"]))
                writer.writerow(row_out)

        QMessageBox.information(self, "Eksport zakończony", f"Wyeksportowano {len(rows)} founderów.")

    def _update_stats_label(self):
        """Aktualizuje tekst w stats_label na podstawie engine.agents."""
        engine = self.simulation_engine
        step = engine.step_count
        alive = [a.agent_type for a in engine.agents if a.is_alive]
        cnt = Counter(alive)
        text = (f"Steps: {step} | "
                f"Bacteria: {cnt.get('Bacteria',0)} "
                f"Algae: {cnt.get('Algae',0)} "
                f"Fungi: {cnt.get('Fungi',0)} "
                f"Protozoa: {cnt.get('Protozoa',0)}")
        self.stats_label.setText(text)

def run_gui(simulation_engine):
    app = QApplication(sys.argv)
    window = MainWindow(simulation_engine)
    window.show()
    sys.exit(app.exec_())