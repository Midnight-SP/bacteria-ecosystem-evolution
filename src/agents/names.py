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
    "eating_strength": "Edendo",
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
    "eating_strength",
]

def get_species_name(genome: np.ndarray, population_genomes: list[np.ndarray]) -> str:
    """
    Nadaje dwuczłonową nazwę łacińską na podstawie najbardziej wyróżniających się cech genomu,
    ale tylko z tych genów, które faktycznie opisują cechy (nie wagi NN).
    """
    gene_len = min(len(GENE_INDEX_TO_NAME), len(genome))
    # Bierzemy tylko genomy o tej samej długości co bieżący genom
    filtered = [np.array(g[:gene_len]) for g in population_genomes if len(g) == gene_len]
    if not filtered:
        pop_matrix = np.stack([genome[:gene_len]])
    else:
        pop_matrix = np.stack(filtered)
    genome_fixed = np.array(genome[:gene_len])
    medians = np.median(pop_matrix, axis=0)
    diffs = np.abs(genome_fixed - medians)
    if gene_len < 2:
        genus = "Genus"
        species = "species"
    else:
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