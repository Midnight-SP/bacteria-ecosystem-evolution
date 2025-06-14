genome_map = {
    0: "initial_energy",         # Początkowa energia komórki
    1: "max_age",                # Maksymalny wiek komórki
    2: "mutation_rate",          # Prawdopodobieństwo mutacji
    3: "pheromone_strength",     # Siła feromonów
    4: "max_health",                # Zdrowie komórki
    5: "color_red",            # Składnik koloru czerwonego (0-255)
    6: "color_green",          # Składnik koloru zielonego (0-255)
    7: "color_blue",           # Składnik koloru niebieskiego (0-255)
    8: "metabolism_rate",  # Tempo metabolizmu (jak szybko zużywa energię)
    9: "photosynthesis_rate",  # Tempo fotosyntezy (jak szybko regeneruje energię)
    10: "division_threshold",  # Próg podziału komórki
    11: "chemotaxis_strength",  # Siła chemotaksji (wrażliwość na feromony)
    12: "chemotaxis_sensitivity",  # Czułość chemotaksji (jak szybko reaguje na feromony)
    13: "reproduction_cooldown",  # Czas odnowienia po reprodukcji
    14: "lifespan_variation",  # Zmienność długości życia
    # Dodatkowe cechy komórki:
    15: "speed",  # Prędkość ruchu komórki
    16: "sight_range",  # Zasięg widzenia komórki
    17: "sight_angle",  # Kąt widzenia komórki
    18: "dietary_preference",  # Preferencje dietetyczne (0-64 dla protistów, 65-192 dla wszystkiego, 193-256 dla bakterii)
    19: "pheromone_preference",  # Preferencje feromonów
    20: "turning_speed",  # Prędkość obrotu (jak szybko może zmienić kierunek)
    21: "aggression_level",  # Poziom agresji (jak bardzo atakuje inne komórki)
    
    # Ostatnie input_size * output_size genów:
    # Wagi sieci neuronowej (nn_weights)
    # Przykład: jeśli input_size=18, output_size=5, to geny [-90:] to nn_weights
    "nn_weights": "Ostatnie input_size * output_size genów: wagi sieci neuronowej"
}