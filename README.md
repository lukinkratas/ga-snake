# Snake [WIP]

### Play

```bash
uv run play.py
```
- upto 3 players simultaneously

![play](doc/play.png)

### GA Train

```bash
uv run --group ga train_ga.py
```
- 160 genomes
- 8x4 matrix controlled
- 8 features: danger (rel. distance to wall or body) in each direction, rel. distance to apple in each direction

https://github.com/user-attachments/assets/9097a2af-d041-4078-a492-54cf8ffe0297
