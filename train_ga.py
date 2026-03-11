import datetime
import logging
import random
from pathlib import Path

import numpy as np
import pygame
from numpy import linalg
from pygame_screen_record import ScreenRecorder

from snake.engine import GAController, GAGame, Player
from snake.renderer import Renderer
from snake.state import DeterministicApple, Snake
from snake.utils import get_random_color, get_squared_wall

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
rng = np.random.default_rng(seed=42)

NCOLS = 30
NROWS = 20
GRID_SIZE = 15
WIDTH = NCOLS * GRID_SIZE
HEIGHT = NROWS * GRID_SIZE

FPS = 60
POP_SIZE = 160
NGENS = 1000
# limit number of steps per game
# NSTEPS = 300

VECS_TO_APPLE = np.array(DeterministicApple._COORDS) - np.array(
    [Snake.INIT_HEAD_COORDS, *DeterministicApple._COORDS[:-1]]
)


def init_genomes(size: int) -> list[np.ndarray]:
    """Initialize genomes / population.

    Args:
        size: length of returned list of genomes

    Returns: list of genomes
    """
    population = []

    # add prev best genome, fi exists
    best_genome_path = Path("best_genome.npy")
    if best_genome_path.exists():
        logger.debug("Last best genome found.")
        population.append(np.load(best_genome_path))
        remaining_size = size - 1
    else:
        remaining_size = size

    population.extend(rng.uniform(-1.0, 1.0, size=(remaining_size, 8, 4)))
    return population


def init_games(population: list[np.ndarray]) -> list[GAGame]:
    """Initialize games with it's assets - player, controller, wall, snake and apple.

    Args:
        population: list of genomes

    Returns: list of games
    """
    # common wall for all games
    wall = get_squared_wall(NCOLS, NROWS)
    games = []
    for idx, genome in enumerate(population, start=1):
        color = get_random_color()
        controller = GAController(NCOLS, NROWS, genome)
        player = Player(color, controller, name=f"G{idx}")
        snake = Snake(color)
        apple = DeterministicApple(color)
        game = GAGame(NCOLS, NROWS, player, wall, snake, apple)
        games.append(game)

    return games


def reset_games(games: list[GAGame], population: list[np.ndarray]) -> None:
    """Reset games from list and set new GA controllers from list.

    Args:
        games: list of games
        population: list of genomes (arrays)
    """
    for genome, game in zip(population, games):
        game.reset()
        game.player.controller = GAController(NCOLS, NROWS, genome)


def eval_fitness(game: GAGame, max_steps: int) -> float:
    """Evaluate fitness of game.

    Args:
        game: game instance
        max_steps: max number of steps of given generation

    Returns: fitness per game
    """
    score_factor = game.player.score
    game_over_penalty = game.is_over
    fitness = 10 * score_factor - 10 * game_over_penalty

    # coords stepped penalty: 0 if lasted till max steps, otherwise linearly increasing
    steps_penalty = 1 - game.steps / max_steps
    fitness -= 10 * steps_penalty

    # cycling penalty: 0 if all steps were unique, otherwise linearly increasing
    cycling_penalty = (
        (1 - np.unique(game.coords_stepped, axis=0).shape[0] / len(game.coords_stepped))
        if len(game.coords_stepped) > 0
        else 1
    )
    fitness -= 10 * cycling_penalty

    # apple_dist_penalty: 1 if distance from apple to snake's head is is max distance (diagonal), otherwise linearly decreasing
    apple_dist_penalty = linalg.norm(
        game.apple.coords - game.snake.head_coords, 2
    ) / linalg.norm([NCOLS, NROWS], 2)
    fitness -= 10 * apple_dist_penalty

    # apple dir penalty: 0 if all applied directions in the current apple hunt are are the same as vector between current and previous apple.
    vec_to_apple = VECS_TO_APPLE[game.apple.idx].copy()
    # select nonzero items
    nz_idxs = np.nonzero(vec_to_apple)
    # normalize nonzero items to 1
    vec_to_apple[nz_idxs] = vec_to_apple[nz_idxs] / np.abs(vec_to_apple[nz_idxs])

    # apple_dir can be: [1, 0], [0, 1], or even [1, 1]
    if np.count_nonzero(vec_to_apple) == 2:
        apple_dir_x, apple_dir_y = vec_to_apple
        apple_dirs = [np.array([apple_dir_x, 0]), np.array([0, apple_dir_y])]
    else:
        apple_dirs = [vec_to_apple]

    # [2 or 1 or 0, ...]
    dir_matches = np.count_nonzero(
        np.isin(game.dirs_from_last_apple, apple_dirs), axis=1
    )
    apple_dir_penalty = (
        (
            np.extract(dir_matches != 2, dir_matches).shape[0]
            / len(game.dirs_from_last_apple)
        )
        if len(game.dirs_from_last_apple) > 0
        else 1
    )
    fitness -= 10 * apple_dir_penalty

    logger.debug(
        f"{game.player.name} fitness: {fitness}"
        f", {score_factor=}"
        f", {game_over_penalty=}"
        f", {steps_penalty=}"
        f", {cycling_penalty=}"
        f", {apple_dist_penalty=}"
        f", {apple_dir_penalty=}"
    )

    return fitness


def sort_games_by_fitness(
    games: list[GAGame], fitness: list[float]
) -> list[np.ndarray]:
    """Sort games by fitness in descencing order.

    Args:
        games: list of games
        fitness: list of fitnesses

    Returns: list of genomes sorted by fitness
    """
    # sort by fitness
    elite_idxs = np.argsort(fitness)[::-1]
    # select top 5 as elite
    return [games[idx] for idx in elite_idxs]


def mutate(genome: np.ndarray, gen: int) -> np.ndarray:
    """Mutate genome.

    Args:
        genome: genome (array)
        gen: current generation number, used for balancing exploration and exploitation

    Return: Mutated genome (array)
    """
    logger.debug("Mutating genome.")
    mutation_rate = 0.2
    mutation_scale = 0.4

    # progress = gen / NGENS
    # mutation_rate = max(0.05, 0.2 * (1 - progress))
    # mutation_scale = max(0.05, 0.4 * (1 - progress))
    # Cosine annealing: oscillates to escape local optima
    # mutation_rate = 0.05 + 0.15 * (1 + np.cos(np.pi * progress)) / 2
    # mutation_scale = 0.05 + 0.35 * (1 + np.cos(np.pi * progress)) / 2

    mask = rng.uniform(0, 1, genome.shape) < mutation_rate
    noise = rng.uniform(-1, 1, genome.shape) * mutation_scale
    new_arr = genome.copy()
    new_arr[mask] += noise[mask]
    return new_arr


def get_mutated_genomes(
    genomes_choice: list[np.ndarray], size: int, gen: int
) -> list[np.ndarray]:
    """Mutate genomes randomly chosen from list.

    Args:
        genomes_choice: choice list of genomes (arrays)
        size: length of returned list of crossovered genomes
        gen: current generation number, used for balancing exploration and exploitation

    Returns: list of crossovered genomes (arrays)
    """
    genomes = []
    for _ in range(size):
        parent = random.choice(genomes_choice)
        child_genome = mutate(parent, gen)
        genomes.append(child_genome)
    return genomes


def crossover(genome_a: np.ndarray, genome_b: np.ndarray) -> np.ndarray:
    """Crossover genomes.

    Args:
        genome_a: genome (array)
        genome_b: genome (array)

    Return: Crossovered genome (array)
    """
    logger.debug("Crossovering genomes.")
    mask = rng.uniform(0, 1, genome_a.shape) < 0.5
    return np.where(mask, genome_a, genome_b)


def get_crossover_genomes(
    genomes_a_choice: list[np.ndarray], genomes_b_choice: list[np.ndarray], size: int
) -> list[np.ndarray]:
    """Crossover genomes randomly chosen from lists.

    Args:
        genomes_a_choice: choice list of genomes (arrays)
        genomes_b_choice: choice list of genomes (arrays)
        size: length of returned list of crossovered genomes

    Returns: list of crossovered genomes (arrays)
    """
    genomes = []
    for _ in range(size):
        parent_a = random.choice(genomes_a_choice)
        parent_b = random.choice(genomes_b_choice)
        child_genome = crossover(parent_a, parent_b)
        genomes.append(child_genome)
    return genomes


def get_next_population(
    sorted_population: list[np.ndarray], gen: int
) -> list[np.ndarray]:
    """Get next generation population by mutations and crossovers.

    Args:
        sorted_population: list of genomes sorted by fitness
        gen: current generation number, used for balancing exploration and exploitation

    Returns: next population
    """
    # 15%, at least 3
    nelites = max(3, round(0.15 * POP_SIZE))

    elites = sorted_population[:nelites]
    rest = sorted_population[nelites:]
    top_half = sorted_population[nelites : int(0.5 * POP_SIZE)]

    # keep elites unchanged
    next_population = elites.copy()

    # inject random immigrants
    nimmigrants = max(2, round(0.05 * POP_SIZE))
    next_population.extend(rng.uniform(-1.0, 1.0, size=(nimmigrants, 8, 4)))

    mutated_genomes = get_mutated_genomes(elites, size=int(0.15 * POP_SIZE), gen=gen)
    next_population.extend(mutated_genomes)

    mutated_genomes = get_mutated_genomes(top_half, size=int(0.15 * POP_SIZE), gen=gen)
    next_population.extend(mutated_genomes)

    crossover_genomes = get_crossover_genomes(elites, elites, size=int(0.3 * POP_SIZE))
    next_population.extend(crossover_genomes)

    crossover_genomes = get_crossover_genomes(elites, rest, size=int(0.2 * POP_SIZE))
    next_population.extend(crossover_genomes)

    return next_population[:POP_SIZE]


def main() -> None:
    """Main GA training function.

    Inits genomes and games.
    Renders frames of all games of all generations.
    Handles the genomes mutations and crossovers.
    Plots the best fitness.
    Records a video per generation.
    """
    pygame.init()

    scoreboard_row_size = 14
    score_width = 800
    plot_height = 305

    win = pygame.display.set_mode(size=(WIDTH + score_width, HEIGHT + plot_height))
    pygame.display.set_caption("Snake - GA Training")

    clock = pygame.time.Clock()

    game_rect = pygame.Rect(0, 0, WIDTH, HEIGHT)
    game_surf = win.subsurface(game_rect)

    score_rect = pygame.Rect(WIDTH, 0, score_width, HEIGHT + plot_height)
    score_surf = win.subsurface(score_rect)

    plot_rect = pygame.Rect(0, HEIGHT, WIDTH, plot_height)
    plot_surf = win.subsurface(plot_rect)

    population = init_genomes(POP_SIZE)
    games = init_games(population)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_dir = Path("doc").joinpath(f"train_ga_{ts}")
    video_dir.mkdir()
    best_fitness_history = []
    avg_fitness_history = []
    gen = 1

    renderer = Renderer(
        game_surf,
        score_surf,
        NCOLS,
        NROWS,
        GRID_SIZE,
        rect_radius=int(GRID_SIZE / 4),
        line_width=1,
        font_size=10,
        scoreboard_row_size=scoreboard_row_size,
        plot_surf=plot_surf,
    )
    renderer.render_games(games)
    renderer.render_scoreboard(games)
    pygame.display.update()

    pygame.time.delay(1000)

    is_running = True
    # is_restarted = False
    while True:
        logger.info(f"New gen {gen}")

        recorder = ScreenRecorder(FPS).start_rec()
        max_steps = max(100, gen * 10)
        # gen loop
        for step_idx in range(max_steps + POP_SIZE):
            clock.tick(FPS)

            # start games with 1 fram delay from each other
            for game_idx, game in enumerate(games):
                if game_idx == step_idx:
                    game.has_started = True

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False

                    if event.key == pygame.K_q:
                        is_running = False

            if is_running:
                for game in games:
                    if game.has_started and not game.is_over and game.steps < max_steps:
                        game.step()

                fitness_desc = [eval_fitness(game, max_steps) for game in games]
                sorted_games_desc = sort_games_by_fitness(games, fitness_desc)
                renderer.render_games(sorted_games_desc[::-1], gen=gen)
                renderer.render_scoreboard(sorted_games_desc, gen=gen)

                pygame.display.update()

            if is_running is False or all(
                game.has_started and game.is_over and game.steps >= max_steps
                for game in games
            ):
                break

        recorder.stop_rec()
        recorder.save_recording(video_dir.joinpath(f"gen{gen}.mp4"))

        if is_running is False:
            # break before the mutation and crossover
            break

        fitness = [eval_fitness(game, max_steps) for game in games]
        best_fitness_history.append(np.max(fitness))
        avg_fitness_history.append(np.mean(fitness))
        sorted_games_desc = sort_games_by_fitness(games, fitness)
        sorted_population_desc = [g.player.controller.genome for g in sorted_games_desc]
        logger.debug(f"best genome: {sorted_population_desc[0]}")
        np.save("best_genome.npy", sorted_population_desc[0])
        next_population = get_next_population(sorted_population_desc, gen)

        pygame.time.delay(1000)
        reset_games(games, next_population)

        renderer.render_plot(best_fitness_history, avg_fitness_history)
        pygame.display.update()

        if gen == NGENS:
            break

        gen += 1

    pygame.quit()


if __name__ == "__main__":
    main()
