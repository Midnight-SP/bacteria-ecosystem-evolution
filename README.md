# Bacteria Ecosystem Evolution

Symulacja ewolucji mikroorganizmów w dynamicznym ekosystemie.

## Główne cechy

- Ewolucja bakterii i protistów z własnym genomem i siecią neuronową
- Modularna architektura (łatwa rozbudowa)
- Konfigurowalne parametry symulacji
- Wizualizacja świata i populacji
- Testy jednostkowe

## Inspiracje

- [neat-python](https://github.com/CodeReclaimers/neat-python) – biblioteka do ewolucji sieci neuronowych metodą NEAT.
- [FAVITES](https://github.com/niemasd/FAVITES) – narzędzie do symulacji i analizy rozprzestrzeniania się patogenów z modularną architekturą.
- [cax](https://github.com/maxencefaldor/cax) – framework do tworzenia i wizualizacji automatów komórkowych.
- [PyLife2](https://github.com/steph-koopmanschap/PyLife2) – prosty symulator sztucznego życia oparty na automatach komórkowych.

## Struktura projektu

```
src/
  core/
  environment/
  simulation/
  ui/
  utils/
tests/
```

## Szybki start

1. `pip install -r requirements.txt`
2. `python src/ui/gui.py`

## Licencja

MIT