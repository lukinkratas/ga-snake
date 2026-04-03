import datetime
import logging

# from itertools import product
from pathlib import Path

import numpy as np
import pygame
from numpy import linalg
from pygame_screen_record import ScreenRecorder

from snake.const import DIRECTIONS
from snake.engine import GAController, GAGame, Player
from snake.renderer import Renderer
from snake.state import Apple, Snake
from snake.utils import (
    get_exclude_coords,
    get_free_coords,
    get_random_color,
    get_squared_wall,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
rng = np.random.default_rng(seed=42)

NCOLS = 30
NROWS = 20
GRID_SIZE = 15
WIDTH = NCOLS * GRID_SIZE
HEIGHT = NROWS * GRID_SIZE

FPS = 60
NGENOMES = 600
NGENS = 1000
NSTEPS = 600

BEST_GENOMES_DIR = Path("best_genomes")

SHAPE = (len(GAController.FEATURE_NAMES), len(DIRECTIONS))
TRAINING_SETS = [
    [
        # 0: mid of wall, clockwise
        np.array([15, 18]),
        np.array([1, 10]),
        np.array([15, 1]),
        np.array([28, 10]),
        np.array([15, 15]),
        np.array([10, 10]),
        np.array([15, 5]),
        np.array([20, 10]),
        np.array([15, 11]),
        np.array([14, 10]),
        np.array([15, 9]),
        np.array([16, 10]),
    ],
    [
        # 1: every corner, anti-clockwise
        np.array([28, 1]),
        np.array([1, 1]),
        np.array([1, 18]),
        np.array([28, 18]),
        np.array([20, 5]),
        np.array([10, 5]),
        np.array([10, 15]),
        np.array([20, 15]),
        np.array([16, 9]),
        np.array([14, 9]),
        np.array([14, 11]),
        np.array([16, 11]),
    ],
    [
        # 2: mid of wall, back to mid, anti-clockwise
        np.array([15, 1]),
        np.array([15, 10]),
        np.array([1, 10]),
        np.array([15, 10]),
        np.array([15, 18]),
        np.array([15, 10]),
        np.array([28, 10]),
        np.array([15, 10]),
    ],
    [
        # 3: every corner, back to mid, clockwise
        np.array([28, 18]),
        np.array([15, 10]),
        np.array([1, 18]),
        np.array([15, 10]),
        np.array([1, 1]),
        np.array([15, 10]),
        np.array([28, 1]),
        np.array([15, 10]),
    ],
    [
        # 4: eight
        np.array([28, 18]),
        np.array([15, 18]),
        np.array([15, 10]),
        np.array([15, 1]),
        np.array([1, 1]),
        np.array([1, 10]),
        np.array([15, 10]),
        np.array([15, 1]),
        np.array([28, 1]),
        np.array([28, 10]),
        np.array([15, 10]),
        np.array([1, 10]),
        np.array([1, 18]),
        np.array([15, 18]),
        np.array([15, 10]),
    ],
    [
        # 5: triangles
        np.array([28, 18]),
        np.array([1, 18]),
        np.array([1, 1]),
        np.array([28, 18]),
        np.array([28, 1]),
        np.array([1, 1]),
        np.array([28, 18]),
        np.array([1, 18]),
        np.array([28, 1]),
        np.array([28, 18]),
        np.array([1, 18]),
        np.array([28, 1]),
        np.array([1, 1]),
        np.array([1, 18]),
    ],
    [
        # 6: zig zags
        np.array([28, 18]),
        np.array([1, 10]),
        np.array([28, 10]),
        np.array([1, 1]),
        np.array([28, 1]),
        np.array([15, 18]),
        np.array([15, 1]),
        np.array([1, 18]),
        np.array([1, 1]),
    ],
    [
        # 7: side to side
        np.array([1, 18]),
        np.array([1, 1]),
        np.array([2, 18]),
        np.array([2, 1]),
        np.array([3, 18]),
        np.array([3, 1]),
        np.array([4, 18]),
        np.array([4, 1]),
        np.array([5, 18]),
        np.array([5, 1]),
        np.array([6, 18]),
        np.array([6, 1]),
        np.array([28, 1]),
        np.array([1, 2]),
        np.array([28, 2]),
        np.array([1, 3]),
        np.array([28, 3]),
        np.array([1, 4]),
        np.array([28, 4]),
        np.array([1, 5]),
        np.array([28, 5]),
        np.array([1, 6]),
        np.array([28, 6]),
    ],
]
# Number of rounds per generation, if larger than training rounds -> extra rounds are random
NROUNDS = len(TRAINING_SETS) + 1
# selection momentum - number of gens after which selection occurs
MOMENTUM = NROUNDS


def init_population(size: int) -> list[np.ndarray]:
    """Initialize population / genomes.

    Args:
        size: length of returned list of genomes

    Returns: population - list of genomes(arrays)
    """
    return [rng.choice([0.0, 1.0, -1.0], size=SHAPE) for _ in range(size)]


def init_games(population: list[np.ndarray]) -> list[GAGame]:
    """Initialize games with it's assets - player, controller, wall, snake and apple.

    Args:
        population: list of genomes(arrays)

    Returns: list of games
    """
    # common wall for all games
    wall = get_squared_wall(NCOLS, NROWS)

    def init_game(genome: np.ndarray, player_name: str | None = None) -> GAGame:
        color = get_random_color()
        controller = GAController(NCOLS, NROWS, genome)
        player = Player(color, controller, player_name)
        snake = Snake()
        apple = Apple()
        return GAGame(NCOLS, NROWS, player, wall, snake, apple)

    return [
        init_game(genome, f"G{gidx}") for gidx, genome in enumerate(population, start=1)
    ]


def reset_games(games: list[GAGame]) -> None:
    """Reset games from list and set new GA controllers from list.

    Args:
        games: list of games
        population: list of genomes(arrays)
    """
    # separate for loop in case lens are not equal
    for game in games:
        game.reset()


def set_population(games: list[GAGame], population: list[np.ndarray]) -> None:
    for game, genome in zip(games, population):
        game.player.controller = GAController(NCOLS, NROWS, genome)


def eval_fitness(game: GAGame) -> float:
    """Evaluate fitness of game.

    Args:
        game: game instance

    Returns: fitness per game
    """
    score_factor = game.player.score
    game_over_penalty = int(game.is_over)
    fitness = 10 * score_factor - 2 * game_over_penalty
    info = {"score_factor": score_factor, "game_over_penalty": game_over_penalty}

    # # steps penalty: 0 if lasted till max steps, otherwise linearly increasing
    # # commented out -> I do not want snake to stay as many steps as possible
    # # 1. that leads to cycling or pseudo-cycling
    # # I rather want it to be as efficient as possible
    # steps_penalty = 1 - game.steps / max_steps
    # fitness -= steps_penalty
    # info["steps_penalty"] = steps_penalty

    # # steps penalty: 0 if the most efficient path, otherwise linearly increasing
    # steps_penalty = max(0, game.snake.steps / np.sum(game.apple._min_steps_needed) - 1)
    # fitness -= 2 * steps_penalty
    # info["steps_penalty"] = steps_penalty

    # cycling penalty: 0 if all last steps were unique, otherwise linearly increasing
    last_steps = game.snake.coords_history[-100:]
    cycling_penalty = 1 - np.unique(last_steps, axis=0).shape[0] / len(last_steps)
    fitness -= 2 * cycling_penalty
    info["cycling_penalty"] = cycling_penalty

    # apple_dist_penalty: 1 if distance from apple to snake's head is is max distance
    # (diagonal), otherwise linearly decreasing
    apple_dist_penalty = linalg.norm(
        game.apple.coords - game.snake.head_coords, 2
    ) / linalg.norm([NCOLS, NROWS], 2)
    fitness -= apple_dist_penalty
    info["apple_dist_penalty"] = apple_dist_penalty

    logger.debug(f"{game.player.name} fitness: {fitness}, {info}")

    return fitness


def mutate(genome: np.ndarray, progress: float) -> np.ndarray:
    """Mutate genome.

    Args:
        genome: genome (array)
        progress: float 0-1, indicating how far throught the generations training is

    Return: Mutated genome (array)
    """
    logger.debug("Mutating genome.")

    # mutation_rate = 0.5
    mutation_rate = 0.1 + 0.2 * (1 - progress)
    mask = rng.random(genome.shape) < mutation_rate

    # mutation_scale = 1.0
    mutation_scale = 0.2 + 0.4 * (1 - progress)
    noise = rng.uniform(-1.0, 1.0, size=genome.shape) * mutation_scale

    new_arr = genome.copy()
    new_arr[mask] += noise[mask]

    return np.clip(new_arr, -1, 1)


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


def get_next_gen(
    elites: list[np.ndarray], population: list[np.ndarray], progress: float
) -> list[np.ndarray]:
    """Get next generation population by mutations and crossovers."""

    def get_mutated_genomes(
        genomes_choice: list[np.ndarray], size: int, progress: float
    ) -> list[np.ndarray]:
        """Mutate genomes randomly chosen from list.

        Args:
            genomes_choice: choice list of genomes (arrays)
            size: length of returned list of crossovered genomes
            progress: float 0-1, indicating how far throught the generations training is

        Returns: list of crossovered genomes (arrays)
        """
        return [mutate(genome, progress) for genome in rng.choice(genomes_choice, size)]

    def get_crossover_genomes(
        genomes_a_choice: list[np.ndarray],
        genomes_b_choice: list[np.ndarray],
        size: int,
    ) -> list[np.ndarray]:
        """Crossover genomes randomly chosen from lists.

        Args:
            genomes_a_choice: choice list of genomes (arrays)
            genomes_b_choice: choice list of genomes (arrays)
            size: length of returned list of crossovered genomes

        Returns: list of crossovered genomes (arrays)
        """
        return [
            crossover(parent_a, parent_b)
            for parent_a, parent_b in zip(
                rng.choice(genomes_a_choice, size), rng.choice(genomes_b_choice, size)
            )
        ]

    ngenomes = len(population)

    # keep elites unchanged
    next_gen = elites.copy()

    mutated_genomes = get_mutated_genomes(
        elites, size=int(0.30 * ngenomes), progress=progress
    )
    next_gen.extend(mutated_genomes)

    mutated_genomes = get_mutated_genomes(
        population, size=int(0.10 * ngenomes), progress=progress
    )
    next_gen.extend(mutated_genomes)

    crossover_genomes = get_crossover_genomes(elites, elites, size=int(0.30 * ngenomes))
    next_gen.extend(crossover_genomes)

    crossover_genomes = get_crossover_genomes(
        elites, population, size=int(0.10 * ngenomes)
    )
    next_gen.extend(crossover_genomes)

    # inject random immigrants
    next_gen.extend(init_population(size=int(0.10 * ngenomes)))

    return next_gen


def _get_alphas(pop_size: int) -> np.ndarray:
    alphas = 63 * np.ones(pop_size)
    alphas[-1] = 255
    return alphas


# def get_apple_sequence(gen: int, length: int = 100) -> list[np.ndarray]:
#     gen_rng = np.random.default_rng(seed=gen)  # deterministic per gen
#     coords = []
#     for _ in range(length):
#         x = gen_rng.integers(0, NCOLS)
#         y = gen_rng.integers(0, NROWS)
#         # CAN BE IN WALL !!
#         coords.append(np.array([x, y]))
#     return coords


def select_elites(
    fitness_history: list[list[float]], population: list[np.ndarray], size: int
):
    momentum = len(fitness_history)
    elites = []
    for fitness in fitness_history:
        order_desc = np.argsort(fitness)[::-1]
        elites.extend([population[idx] for idx in order_desc[: int(size / momentum)]])

    return elites


def main() -> None:
    """Main GA training function.

    Inits genomes and games.
    Renders frames of all games of all generations.
    Handles the genomes mutations and crossovers.
    Plots the best fitness.
    Records a video per generation.
    """
    pygame.init()

    score_width = 800
    history_plot_height = 155
    genome_plot_height = 155

    win = pygame.display.set_mode(
        size=(WIDTH + score_width, HEIGHT + history_plot_height + genome_plot_height)
    )
    pygame.display.set_caption("Snake - GA Training")

    clock = pygame.time.Clock()

    game_rect = pygame.Rect(0, 0, WIDTH, HEIGHT)
    game_surf = win.subsurface(game_rect)

    score_rect = pygame.Rect(
        WIDTH, 0, score_width, HEIGHT + history_plot_height + genome_plot_height
    )
    score_surf = win.subsurface(score_rect)

    history_plot_rect = pygame.Rect(0, HEIGHT, WIDTH, history_plot_height)
    history_plot_surf = win.subsurface(history_plot_rect)

    genome_plot_rect = pygame.Rect(
        0, HEIGHT + history_plot_height, WIDTH, genome_plot_height
    )
    genome_plot_surf = win.subsurface(genome_plot_rect)

    population = init_population(NGENOMES)
    # game_pools and games flatten copy?
    games = init_games(population)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_dir = Path("doc").joinpath(f"train_ga_{ts}")
    video_dir.mkdir()
    fitness_history = []

    renderer = Renderer(
        game_surf,
        score_surf,
        NCOLS,
        NROWS,
        GRID_SIZE,
        rect_radius=int(GRID_SIZE / 4),
        line_width=1,
        font_size=10,
        history_plot_surf=history_plot_surf,
        genome_plot_surf=genome_plot_surf,
    )

    is_running = True
    for gen in range(NGENS):
        logger.info(f"New gen {gen}")

        training_idx = gen % NROUNDS
        training_set = (
            TRAINING_SETS[training_idx] if training_idx < len(TRAINING_SETS) else []
        )

        renderer.render_games(games[::-1], alphas=_get_alphas(NGENOMES))
        renderer.render_scoreboard(games, gen)
        pygame.display.update()

        pygame.time.delay(1000)

        recorder = ScreenRecorder(FPS).start_rec()

        # gen loop
        for _ in range(NSTEPS):
            clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        is_running = False

            if is_running is False:
                break

            for game in games:
                if not game.is_over:
                    apple_eaten = game.step()

                    if apple_eaten:
                        # infinite loop - if apple coords, exceeded, start from first idx
                        coords = training_set[game.apple.idx % len(training_set)]
                        game.apple.move(coords)

                    # last training round (random), append apples on the fly
                    if training_idx >= len(TRAINING_SETS) and game.apple.idx >= len(
                        training_set
                    ):
                        xrange = np.arange(NCOLS)
                        yrange = np.arange(NROWS)
                        exclude = get_exclude_coords(games)
                        coords = get_free_coords(xrange, yrange, exclude)
                        training_set.append(coords)

            # render games based on orig order
            renderer.render_games(games[::-1], alphas=_get_alphas(NGENOMES))

            # render scoreboard sorted
            fitness = [eval_fitness(game) for game in games]
            order_desc = np.argsort(fitness)[::-1]
            sorted_games_desc = [games[idx] for idx in order_desc]
            renderer.render_scoreboard(sorted_games_desc, gen=gen)
            pygame.display.update()

            # if all game are over prior to finishing all NSTEPS
            if all(game.is_over for game in games):
                break

        recorder.stop_rec()
        recorder.save_recording(video_dir.joinpath(f"gen{gen}.mp4"))

        if is_running is False:
            # break before the mutation and crossover
            break

        # eval
        fitness = [eval_fitness(game) for game in games]
        fitness_history.append(fitness.copy())
        order_desc = np.argsort(fitness)[::-1]
        best_idx = order_desc[0]

        # render plots
        renderer.render_history_plot(fitness_history, MOMENTUM)
        renderer.render_genome_plot(
            population[best_idx],
            color=np.array(games[best_idx].player.color) / 255,
            name=games[best_idx].player.name,
            fitness=fitness[best_idx],
        )
        pygame.display.update()

        if gen % MOMENTUM == 0 and gen > 0:
            # eval
            fitness = np.mean(np.transpose(fitness_history[-MOMENTUM:]), axis=1)
            order_desc = np.argsort(fitness)[::-1]
            best_idx = order_desc[0]

            # save best genome
            np.save(
                BEST_GENOMES_DIR.joinpath(f"best_genome_{ts}.npy"),
                population[best_idx],
            )

            # set next gen
            elites = select_elites(
                fitness_history[-MOMENTUM:], population, size=int(0.10 * NGENOMES)
            )
            population = get_next_gen(elites, population, progress=gen / NGENS)
            set_population(games, population)

        reset_games(games)

        pygame.time.delay(1000)

    pygame.quit()


if __name__ == "__main__":
    main()
