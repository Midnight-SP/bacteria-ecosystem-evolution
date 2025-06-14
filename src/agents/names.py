import numpy as np

# Mapowanie cech na łacińskie rdzenie (przykładowe, możesz rozbudować)
LATIN_ROOTS = {
    "initial_energy": "Energo",
    "max_age": "Senecto",
    "mutation_rate": "Mutato",
    "pheromone_strength": "Pheromono",
    "max_health": "Sanito",
    "color_red": "Rubra",
    "color_green": "Virido",
    "color_blue": "Caeruleo",
    "metabolism_rate": "Metabolo",
    "photosynthesis_rate": "Photosyntho",
    "division_threshold": "Divisio",
    "chemotaxis_strength": "Chemotaxo",
    "chemotaxis_sensitivity": "Sensitivo",
    "reproduction_cooldown": "Reproducto",
    "lifespan_variation": "Variato",
    "speed": "Celer",
    "sight_range": "Visio",
    "sight_angle": "Angulo",
    "dietary_preference": "Dieta",
    "pheromone_preference": "Pheroprefero",
    "turning_speed": "Gyrato",
    "aggression_level": "Aggresso",
}

# Lista cech do analizy (indeksy zgodne z genome_map)
GENE_INDEX_TO_NAME = [
    "initial_energy",
    "max_age",
    "mutation_rate",
    "pheromone_strength",
    "max_health",
    "color_red",
    "color_green",
    "color_blue",
    "metabolism_rate",
    "photosynthesis_rate",
    "division_threshold",
    "chemotaxis_strength",
    "chemotaxis_sensitivity",
    "reproduction_cooldown",
    "lifespan_variation",
    "speed",
    "sight_range",
    "sight_angle",
    "dietary_preference",
    "pheromone_preference",
    "turning_speed",
    "aggression_level",
]

def get_species_name(genome: np.ndarray, population_genomes: list[np.ndarray]) -> str:
    """
    Nadaje dwuczłonową nazwę łacińską na podstawie najbardziej wyróżniających się cech genomu.
    """
    gene_len = len(GENE_INDEX_TO_NAME)
    # Funkcja pomocnicza: przytnij lub dopełnij genom do gene_len
    def fix_length(g):
        g = np.array(g)
        if len(g) < gene_len:
            return np.pad(g, (0, gene_len - len(g)), 'constant')
        else:
            return g[:gene_len]
    # Ujednolicenie długości genomów w populacji
    pop_matrix = np.stack([fix_length(g) for g in population_genomes])
    # Ujednolicenie długości bieżącego genomu
    genome_fixed = fix_length(genome)
    medians = np.median(pop_matrix, axis=0)
    diffs = np.abs(fix_length(genome) - medians)
    top2 = np.argsort(diffs)[-2:][::-1]
    first, second = top2
    genus = LATIN_ROOTS.get(GENE_INDEX_TO_NAME[first], f"Gen{first}")
    species = LATIN_ROOTS.get(GENE_INDEX_TO_NAME[second], f"Spec{second}")
    genus = genus.capitalize()
    species = species.lower()
    return f"{genus} {species}"

# Przykład użycia:
if __name__ == "__main__":
    # Przykładowa populacja genomów (np. 10 komórek)
    pop = [np.random.randint(0, 256, 22, dtype=np.uint8) for _ in range(10)]
    # Genom nowej komórki
    genome = np.random.randint(0, 256, 22, dtype=np.uint8)
    print(get_species_name(genome, pop))