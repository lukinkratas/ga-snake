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

DEBUG = True

NCOLS = 30
NROWS = 20
GRID_SIZE = 15
WIDTH = NCOLS * GRID_SIZE
HEIGHT = NROWS * GRID_SIZE

FPS = 120
POP_SIZE = 120
NGENS = 1000
# limit number of steps per game
# NSTEPS = 300


def init_genomes(size: int) -> list[np.ndarray]:
    """Initialize genomes.

    Args:
        size: length of returned list of genomes

    Returns: list of genomes
    """
    best_genome_path = Path("best_genome.npy")
    if best_genome_path.exists():
        logger.debug("Last best genome found.")
        last_best_genome = np.load(best_genome_path)
        genomes = [last_best_genome]
        rest_size = size - 1
    else:
        genomes = []
        rest_size = size

    rest_genomes = [rng.choice([-0.1, 0.1], (8, 4)) for _ in range(rest_size)]
    genomes.extend(rest_genomes)
    return genomes


def init_games(genomes: list[np.ndarray]) -> list[GAGame]:
    """Initialize games with it's assets - player, controller, wall, snake and apple.

    Args:
        genomes: list of genomes

    Returns: list of games
    """
    # common wall for all games
    wall = get_squared_wall(NCOLS, NROWS)
    games = []
    for idx, genome in enumerate(genomes, start=1):
        color = get_random_color()
        controller = GAController(NCOLS, NROWS, genome)
        player = Player(color, controller, name=f"G{idx}")
        snake = Snake(color)
        apple = DeterministicApple(color)
        game = GAGame(NCOLS, NROWS, player, wall, snake, apple)
        games.append(game)

    return games


def start_games(games: list[GAGame]) -> None:
    """Start games from list.

    Args:
        games: list of games
    """
    for game in games:
        game.has_started = True


def reset_games(games: list[GAGame], genomes: list[np.ndarray]) -> None:
    """Reset games from list and set new GA controllers from list.

    Args:
        games: list of games
        genomes: list of genomes (arrays)
    """
    for genome, game in zip(genomes, games):
        game.reset()
        game.player.controller = GAController(NCOLS, NROWS, genome)


def eval_fitness(game: GAGame, max_steps: int) -> float:
    """Evaluate fitness of game.

    Args:
        game: game instance
        max_steps: max number of steps of given generation

    Returns: fitness per game
    """
    # STEPS_THRESHOLD = 100
    score_factor = 10 * game.player.score
    alive_factor = -5 * game.is_over
    fitness = score_factor + alive_factor

    # last_coords_stepped = game.coords_stepped[:STEPS_THRESHOLD]
    # unique steps repeating == stuck only over a few cells
    num_unique_coords_stepped = np.unique(game.coords_stepped, axis=0).shape[0]
    # reward exploration
    unique_coords_factor = num_unique_coords_stepped / max_steps
    fitness += unique_coords_factor

    # reward distance to last apple
    apple_dist_factor = 1 - linalg.norm(
        game.apple.coords - game.snake.head_coords, 2
    ) / linalg.norm([NCOLS, NROWS], 2)

    fitness += apple_dist_factor

    # reward directions towards apple
    prev_apple = (
        game.apple._COORDS[game.apple.idx - 1]
        if game.apple.idx > 0
        else game.snake.INIT_HEAD_COORDS
    )
    vec_to_apple = game.apple.coords - prev_apple
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

    # /2 bcs extract automatically flattens
    dir_matches = np.count_nonzero(np.isin(game.dirs_to_apple, apple_dirs), axis=1)
    apple_dir_factor = np.extract(dir_matches == 2, dir_matches).shape[0] / len(
        game.dirs_to_apple
    )
    fitness += apple_dir_factor

    logger.debug(
        f"{game.player.name} fitness: {fitness}"
        f", {score_factor=}"
        f", {alive_factor=}"
        f", {unique_coords_factor=}"
        f", {apple_dist_factor=}"
        f", {apple_dir_factor=}"
    )

    return fitness


def sort_genomes_by_fitness(
    games: list[GAGame], fitness: list[float]
) -> list[np.ndarray]:
    """Sort genomes by fitness.

    Args:
        games: list of games
        fitness: list of fitnesses

    Returns: list of genomes sorted by fitness
    """
    genomes = [game.player.controller.genome for game in games]
    # sort by fitness
    elite_idxs = np.argsort(fitness)[::-1]
    # select top 5 as elite
    return [genomes[idx] for idx in elite_idxs]


def mutate(genome: np.ndarray, gen: int) -> np.ndarray:
    """Mutate genome.

    Args:
        genome: genome (array)
        gen: current generation number, used for balancing exploration and exploitation

    Return: Mutated genome (array)
    """
    logger.debug("Mutating genome.")
    mutation_rate = max(0.05, 0.2 * (1 - gen / NGENS))
    mutation_scale = max(0.05, 0.4 * (1 - gen / NGENS))
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
    score_width = 700
    plot_height = 305

    win = pygame.display.set_mode(size=(WIDTH + score_width, HEIGHT + plot_height))
    pygame.display.set_caption("Snake")

    clock = pygame.time.Clock()

    game_rect = pygame.Rect(0, 0, WIDTH, HEIGHT)
    game_surf = win.subsurface(game_rect)

    score_rect = pygame.Rect(WIDTH, 0, score_width, HEIGHT + plot_height)
    score_surf = win.subsurface(score_rect)

    plot_rect = pygame.Rect(0, HEIGHT, WIDTH, plot_height)
    plot_surf = win.subsurface(plot_rect)

    genomes = init_genomes(POP_SIZE)
    games = init_games(genomes)
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

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_dir = Path("doc").joinpath(f"train_ga_{ts}")
    video_dir.mkdir()
    is_running = True
    is_paused = False
    best_fitness_history = []
    for gen in range(1, NGENS + 1):
        logger.info(f"New gen {gen}")

        recorder = ScreenRecorder(FPS).start_rec()

        max_steps = max(100, gen * 10)
        start_games(games)
        # gen loop
        while True:
            clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        is_paused = not is_paused

                    if event.key == pygame.K_q:
                        is_running = False

            if not is_paused and is_running:
                for game in games:
                    if game.is_over or game.steps == max_steps:
                        continue

                    game.step()

                renderer.render_games(games, gen=gen)
                renderer.render_scoreboard(games, gen=gen)

            if is_paused and DEBUG:
                renderer.render_coords()
            if is_paused:
                renderer.render_paused()
            pygame.display.update()

            if (
                all(game.is_over or game.steps == max_steps for game in games)
                or is_running is False
            ):
                break

        recorder.stop_rec()
        recorder.save_recording(video_dir.joinpath(f"gen{gen}.mp4"))

        if is_running is False:
            break

        fitness = [eval_fitness(game, max_steps) for game in games]
        best_fitness_history.append(np.max(fitness))
        sorted_genomes = sort_genomes_by_fitness(games, fitness)
        logger.info(f"best genome: {sorted_genomes[0]}")
        np.save("best_genome.npy", sorted_genomes[0])
        nelites = max(3, round(0.15 * POP_SIZE))  # 15%, at least 3
        elites = sorted_genomes[:nelites]
        rest = sorted_genomes[nelites:]
        top_half = sorted_genomes[nelites : int(0.5 * POP_SIZE)]

        # keep elites unchanged
        next_gen_genomes = elites.copy()

        if len(next_gen_genomes) < POP_SIZE:
            mutated_genomes = get_mutated_genomes(
                elites, size=int(0.15 * POP_SIZE), gen=gen
            )
            next_gen_genomes.extend(mutated_genomes)

        if len(next_gen_genomes) < POP_SIZE:
            mutated_genomes = get_mutated_genomes(
                top_half, size=int(0.15 * POP_SIZE), gen=gen
            )
            next_gen_genomes.extend(mutated_genomes)

        if len(next_gen_genomes) < POP_SIZE:
            crossover_genomes = get_crossover_genomes(
                elites, elites, size=int(0.3 * POP_SIZE)
            )
            next_gen_genomes.extend(crossover_genomes)

        if len(next_gen_genomes) < POP_SIZE:
            crossover_genomes = get_crossover_genomes(
                elites, rest, size=POP_SIZE - len(next_gen_genomes)
            )
            next_gen_genomes.extend(crossover_genomes)

        pygame.time.delay(1000)
        reset_games(games, next_gen_genomes)
        genomes = next_gen_genomes

        renderer.render_plot(best_fitness_history)
        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()
