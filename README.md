# Snake [WIP]

### Play

```bash
uv run python -m snake
```
- upto 3 players simultaneously

![play](doc/play.png)

### GA Train

```bash
uv run --group ga train_ga.py
```
- 400 genomes
- 8x4 matrix controlled
- 8 features: danger (rel. distance to wall or body) in each direction, rel. distance to apple in each direction
- things one can tweak:
  - genome features, controller matrix
  - selection distribution, mutation and crossover settings
  - fitness function - rewards and penalties
  - apple positions - order of features learned
- problem of GA in general:

  1. **Large number of possible combinations:**

    If we consider only binary values (-1, 1) for 8x4 feature array, there is 2^32=4,294,967,296 possible combinations. (And the input can be non-binary, so even larger.)

    Feature arrays are adjusted only via mutations or crossovers. Most of the adjuted arrays are not correctly connecting the features with the direction controls, because of the large number of possible combinations and random nature (mentioned in 2).

  2. **Feature focus:**

    Adjusted features are selected randomly. There is no mechanism of keeping good features, that might be very useful later.

#### Apple positions decision

  1. Random apple
    Random apple positions - different for every generation, different for every game
    Issue: Uncomparable fitness / noisy selection - prefers genomes, that had the apples randomly generated in front of them, instead of learning any features.

  2. Deterministic apple
    Predefined apple positions - same for every generation, same for every game
    This solves the random apple issue, because all genomes train on the same apple positions per generation.
    Issue: Being stuck for quite a long time learning a specific feature, bcs the select population is less fit.
    Minor issue: Some mutations / crossovers might have valuable insights about a certain features, but get eliminated, cause they are not useful at the moment.

  3. Pseudo random apple
    Random apple positions - set of training positions plus one random for every generation, but same for every game
    This solves the deterministic apple issue, because genomes are constantly learning both seen features and arenas as well as random nes.

### GA Play - watch how your trained genome performs on random apple positions

```bash
uv run play_ga.py
```

https://github.com/user-attachments/assets/9097a2af-d041-4078-a492-54cf8ffe0297
