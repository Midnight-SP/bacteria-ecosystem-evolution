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

def get_species_name(genome_obj, population_genomes=None, threshold=0):
    """
    Nadaje dwuczłonową nazwę łacińską na podstawie własnych genów (nie porównuje do populacji).
    """
    genome_type = type(genome_obj).__name__
    gene_names = GENOME_GENE_NAMES.get(genome_type, [])
    genome = genome_obj.genes
    gene_len = min(len(gene_names), len(genome))

    # Wybierz dwie najbardziej skrajne cechy (największa i najmniejsza wartość genu)
    if gene_len < 2:
        genus = "Genus"
        species = "species"
    else:
        # Indeks największego i najmniejszego genu
        max_idx = int(np.argmax(genome[:gene_len]))
        min_idx = int(np.argmin(genome[:gene_len]))
        genus = LATIN_ROOTS.get(gene_names[max_idx], f"Gen{max_idx}")
        species = LATIN_ROOTS.get(gene_names[min_idx], f"Spec{min_idx}")
        genus = genus.capitalize()
        species = species.lower()
    agent_type = type(genome_obj).__name__.replace("Genome", "")
    return f"{genus} {species} [{agent_type}]"