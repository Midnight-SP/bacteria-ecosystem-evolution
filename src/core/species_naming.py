import numpy as np

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
    "photosynthesis": "Photosyntho",
    "photosynthesis_rate": "Photosyntho",
    "division_threshold": "Divisio",
    "chemotaxis_strength": "Chemotaxo",
    "chemotaxis_sensitivity": "Sensitivo",
    "reproduction_cooldown": "Reproducto",
    "lifespan_variation": "Variato",
    "speed": "Celer",
    "spore_spread": "Sporulo",
    "aggression": "Aggresso",
    "infectivity": "Infecto",
    "resistance": "Resisto",
    "replication_rate": "Multiplico",
}

# Mapowanie: typ genomu -> lista nazw genów (kolejność zgodna z GENE_MAP)
GENOME_GENE_NAMES = {
    "BacteriaGenome": [
        "initial_energy", "max_age", "color_red", "color_green", "color_blue",
        "speed", "aggression"
    ],
    "AlgaeGenome": [
        "initial_energy", "max_age", "color_red", "color_green", "color_blue",
        "photosynthesis"
    ],
    "FungiGenome": [
        "initial_energy", "max_age", "color_red", "color_green", "color_blue",
        "spore_spread"
    ],
    "ProtozoaGenome": [
        "initial_energy", "max_age", "color_red", "color_green", "color_blue",
        "aggression", "speed"
    ],
    "Genome": [
        "initial_energy", "max_age", "color_red", "color_green", "color_blue", "speed"
    ]
}

def get_species_name(genome_obj, population_genomes, threshold=0):
    """
    Nadaje dwuczłonową nazwę łacińską na podstawie najbardziej wyróżniających się cech genomu.
    Wersja: wybiera cechy najbardziej różniące się od średniej populacji lub najbliższego sąsiada.
    """
    genome_type = type(genome_obj).__name__
    gene_names = GENOME_GENE_NAMES.get(genome_type, [])
    genome = genome_obj.genes
    gene_len = min(len(gene_names), len(genome))
    filtered = [np.array(g[:gene_len]) for g in population_genomes if len(g) == gene_len]
    genome_fixed = np.array(genome[:gene_len])

    if not filtered:
        pop_matrix = np.stack([genome_fixed])
    else:
        pop_matrix = np.stack(filtered)

    # Wariant 1: różnica od średniej populacji
    means = np.mean(pop_matrix, axis=0)
    diffs_mean = np.abs(genome_fixed - means)

    # Wariant 2: różnica od najbliższego sąsiada (inne genomy)
    if len(pop_matrix) > 1:
        # Oblicz dystanse do wszystkich innych genomów
        dists = np.linalg.norm(pop_matrix - genome_fixed, axis=1)
        # Pomijamy siebie (dystans 0)
        nonzero = np.where(dists > 1e-8)[0]
        if len(nonzero) > 0:
            nearest_idx = nonzero[np.argmin(dists[nonzero])]
            nearest = pop_matrix[nearest_idx]
            diffs_neighbor = np.abs(genome_fixed - nearest)
        else:
            diffs_neighbor = diffs_mean
    else:
        diffs_neighbor = diffs_mean

    # Możesz wybrać jedną z metod:
    diffs = diffs_mean  # lub diffs_neighbor

    if gene_len < 2:
        genus = "Genus"
        species = "species"
    else:
        top2 = np.argsort(diffs)[-2:][::-1]
        first, second = top2
        if diffs[first] < threshold and diffs[second] < threshold:
            genus = "Genus"
            species = "species"
        else:
            genus = LATIN_ROOTS.get(gene_names[first], f"Gen{first}")
            species = LATIN_ROOTS.get(gene_names[second], f"Spec{second}")
            genus = genus.capitalize()
            species = species.lower()
    agent_type = type(genome_obj).__name__.replace("Genome", "")
    return f"{genus} {species} [{agent_type}]"