# Evolutionary Computation - TSP with Genetic Algorithm

A Genetic Algorithm (GA) implementation for solving the Traveling Salesman Problem (TSP), featuring standard ordered crossover and two custom "natural" crossover methods.

Dongseo University - Prof. 강대기

## Crossover Methods

1. **Ordered Crossover (OX)** (`-c 1`): Standard crossover that picks a random sub-segment from parent1 and fills remaining positions from parent2 in order.

2. **Natural Crossover** (`-c 2`): Custom group-based crossover that partitions cities into groups, then alternates between parents (even groups from parent1, odd from parent2), preserving contiguous sub-tours.

3. **Semi-Natural Crossover** (`-c 3`): Selective crossover that gives 75% of cities to the better-performing parent and splits the remaining cities into smaller groups. A variant tests multiple random sub-sequences from both parents and picks the one with the shortest sub-route distance as the seed segment.

## Project Structure

```
├── TSP_GA.py              # Main GA script (selection, breeding, mutation, evolution loop)
├── crossover.py           # Standard ordered crossover (OX)
├── natural_crossover.py   # Custom natural & semi-natural crossover methods
├── dataset.py             # City class, random city generation, TSPLIB file loader
└── dataset/
    ├── 1d_i/
    │   └── att48.txt      # 48 cities (TSPLIB att48)
    └── 1d_ii/
        ├── att532.txt     # 532 cities
        ├── u724.txt       # 724 cities (default)
        └── dsj1000.txt    # 1000 cities
```

## Requirements

```
numpy
matplotlib
```

## Usage

```bash
# Standard ordered crossover
python TSP_GA.py -c 1

# Natural crossover (random groups)
python TSP_GA.py -c 2

# Natural crossover with 5 groups
python TSP_GA.py -c 2 -g 5

# Semi-natural (selective) crossover
python TSP_GA.py -c 3

# Semi-natural with 2 groups (breed2 variant)
python TSP_GA.py -c 3 -g 2
```

## GA Parameters

| Parameter | Value |
|-----------|-------|
| Population size | 100 |
| Generations | 500 |
| Elite size | 20 |
| Mutation rate | 0.01 |
| Mutation type | Swap |
| Selection | Fitness-proportionate |
| Default dataset | u724 (724 cities) |

Results (progress plot and description) are saved to a timestamped `results/` subdirectory.
