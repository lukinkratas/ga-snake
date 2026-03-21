import datetime
import logging
import random
from pathlib import Path

import numpy as np
import pygame
from numpy import linalg
from pygame_screen_record import ScreenRecorder

from snake.const import DIRECTIONS
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
POP_SIZE = 400
NGENS = 500
NSTEPS = 300

BEST_GENOMES_DIR = Path("best_genomes")

SHAPE = (len(GAController.FEATURE_NAMES), len(DIRECTIONS))
# VECS_TO_APPLE = np.array(DeterministicApple._COORDS) - np.array(
#     [Snake.INIT_HEAD_COORDS, *DeterministicApple._COORDS[:-1]]
# )


def init_population(size: int) -> list[np.ndarray]:
    """Initialize population / genomes.

    Args:
        size: length of returned list of genomes

    Returns: population - list of genomes(arrays)
    """
    return [rng.uniform(-1.0, 1.0, size=SHAPE) for _ in range(size)]


def init_games(population: list[np.ndarray]) -> list[GAGame]:
    """Initialize games with it's assets - player, controller, wall, snake and apple.

    Args:
        population - list of genomes(arrays)

    Returns: list of games
    """
    # common wall for all games
    wall = get_squared_wall(NCOLS, NROWS)

    def init_game(genome: np.ndarray, player_name: str | None = None):
        color = get_random_color()
        controller = GAController(NCOLS, NROWS, genome)
        player = Player(color, controller, player_name)
        snake = Snake()
        apple = DeterministicApple()
        return GAGame(NCOLS, NROWS, player, wall, snake, apple)

    return [
        init_game(genome, f"G{gidx}") for gidx, genome in enumerate(population, start=1)
    ]


def reset_games(games: list[GAGame], population: list[np.ndarray]) -> None:
    """Reset games from list and set new GA controllers from list.

    Args:
        games - list of games
        population - list of genomes(arrays)
    """
    # separate for loop in case lens are not equal
    for game in games:
        game.reset()

    for game, genome in zip(games, population):
        game.player.controller = GAController(NCOLS, NROWS, genome)


def eval_fitness(game: GAGame, max_steps: int) -> float:
    """Evaluate fitness of game.

    Args:
        game: game instance
        max_steps: max number of steps of given generation

    Returns: fitness per game
    """
    score_factor = game.player.score
    game_over_penalty = int(game.is_over)
    fitness = 10 * score_factor - 5 * game_over_penalty
    info = {"score_factor": score_factor, "game_over_penalty": game_over_penalty}

    # # coords stepped penalty: 0 if lasted till max steps, otherwise linearly increasing
    # steps_penalty = 1 - game.steps / max_steps
    # fitness -= steps_penalty
    # info["steps_penalty"] = steps_penalty

    # if game.steps > 0:
    #     # cycling penalty: 0 if all steps were unique, otherwise linearly increasing
    #     cycling_penalty = (
    #         1 - np.unique(game.coords_stepped, axis=0).shape[0] / game.steps
    #     )
    #     fitness -= 2 * cycling_penalty
    #     info["cycling_penalty"] = cycling_penalty

    # apple_dist_penalty: 1 if distance from apple to snake's head is is max distance (diagonal), otherwise linearly decreasing
    apple_dist_penalty = linalg.norm(
        game.apple.coords - game.snake.head_coords, 2
    ) / linalg.norm([NCOLS, NROWS], 2)
    fitness -= 0.1 * apple_dist_penalty
    info["apple_dist_penalty"] = apple_dist_penalty

    # # apple dir penalty: 0 if all applied directions in the current apple hunt are are the same as vector between current and previous apple.
    # vec_to_apple = VECS_TO_APPLE[game.apple.idx].copy()
    # # select nonzero items
    # nz_idxs = np.nonzero(vec_to_apple)
    # # normalize nonzero items to 1
    # vec_to_apple[nz_idxs] = vec_to_apple[nz_idxs] / np.abs(vec_to_apple[nz_idxs])
    #
    # # apple_dir can be: [1, 0], [0, 1], or even [1, 1]
    # if np.count_nonzero(vec_to_apple) == 2:
    #     apple_dir_x, apple_dir_y = vec_to_apple
    #     apple_dirs = [np.array([apple_dir_x, 0]), np.array([0, apple_dir_y])]
    # else:
    #     apple_dirs = [vec_to_apple]
    #
    # # [2 or 1 or 0, ...]
    # dir_matches = np.count_nonzero(
    #     np.isin(game.dirs_from_last_apple, apple_dirs), axis=1
    # )
    # apple_dir_penalty = (
    #     (
    #         np.extract(dir_matches != 2, dir_matches).shape[0]
    #         / len(game.dirs_from_last_apple)
    #     )
    #     if len(game.dirs_from_last_apple) > 0
    #     else 1
    # )
    # fitness -= apple_dir_penalty
    # info["apple_dir_penalty"] = apple_dir_penalty

    logger.debug(f"{game.player.name} fitness: {fitness}, {info}")

    return fitness


# def sort_games_by_fitness(
#     games: list[GAGame], fitness: list[float]
# ) -> list[np.ndarray]:
#     """Sort games by fitness in descencing order.
#
#     Args:
#         games: list of games
#         fitness: list of fitnesses
#
#     Returns: list of genomes sorted by fitness
#     """
#     # sort by fitness
#     elite_idxs = np.argsort(fitness)[::-1]
#     # select top 5 as elite
#     return [games[idx] for idx in elite_idxs]
#


def mutate(genome: np.ndarray, progress: float) -> np.ndarray:
    """Mutate genome.

    Args:
        genome: genome (array)
        gen: current generation number, used for balancing exploration and exploitation

    Return: Mutated genome (array)
    """
    logger.debug("Mutating genome.")
    # mutation_rate = 0.2
    # mutation_scale = 0.4

    # mutation_rate = 0.2 + 0.2 * (1 - progress)
    # mask = rng.uniform(0, 1, genome.shape) < mutation_rate
    # mask = rng.random(genome.shape) < mutation_rate
    mask = rng.choice([True, False], size=SHAPE, p=[1 - progress, progress])

    mutation_scale = 0.1 + 0.4 * (1 - progress)
    # noise = rng.uniform(-1, 1, genome.shape) * mutation_scale
    noise = rng.choice([-1, 1], size=genome.shape) * mutation_scale

    new_arr = genome.copy()
    new_arr[mask] += noise[mask]

    return np.clip(new_arr, -1, 1)


def get_mutated_genomes(
    genomes_choice: list[np.ndarray], size: int, progress: float
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
        child_genome = mutate(parent, progress)
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


def get_next_gen(
    sorted_population: list[np.ndarray], progress: float
) -> list[np.ndarray]:
    """Get next generation population by mutations and crossovers."""
    # 10%
    nelites = int(0.10 * POP_SIZE)
    nimmigrants = int(0.10 * POP_SIZE)

    elites = sorted_population[:nelites]
    rest = sorted_population[nelites:]
    top_half = sorted_population[nelites : int(0.50 * POP_SIZE)]

    # keep elites unchanged
    next_gen = elites.copy()

    # inject random immigrants
    next_gen.extend(rng.uniform(-1.0, 1.0, size=(nimmigrants, *SHAPE)))

    mutated_genomes = get_mutated_genomes(
        elites, size=int(0.15 * POP_SIZE), progress=progress
    )
    next_gen.extend(mutated_genomes)

    mutated_genomes = get_mutated_genomes(
        top_half, size=int(0.15 * POP_SIZE), progress=progress
    )
    next_gen.extend(mutated_genomes)

    crossover_genomes = get_crossover_genomes(elites, elites, size=int(0.3 * POP_SIZE))
    next_gen.extend(crossover_genomes)

    crossover_genomes = get_crossover_genomes(elites, rest, size=int(0.2 * POP_SIZE))
    next_gen.extend(crossover_genomes)

    return next_gen


def _get_alphas(pop_size: int) -> np.ndarray:
    alphas = 63 * np.ones(pop_size)
    alphas[-1] = 255
    return alphas


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

    population = init_population(POP_SIZE)
    # game_pools and games flatten copy?
    games = init_games(population)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_dir = Path("doc").joinpath(f"train_ga_{ts}")
    video_dir.mkdir()
    best_fitness_history = []
    avg_fitness_history = []

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
        history_plot_surf=history_plot_surf,
        genome_plot_surf=genome_plot_surf,
    )

    is_running = True
    for gen in range(1, NGENS + 1):
        logger.info(f"New gen {gen}")

        renderer.render_games(games[::-1], alphas=_get_alphas(POP_SIZE))
        renderer.render_scoreboard(games, gen)
        pygame.display.update()
        pygame.time.delay(1000)
        recorder = ScreenRecorder(FPS).start_rec()

        # gen loop
        for step in range(NSTEPS):
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
                    game.step()

            # render games based on orig order
            renderer.render_games(games[::-1], alphas=_get_alphas(POP_SIZE))

            # render scoreboard sorted
            fitness = [eval_fitness(game, NSTEPS) for game in games]
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
        fitness = [eval_fitness(game, NSTEPS) for game in games]
        order_desc = np.argsort(fitness)[::-1]

        # sort in place
        games = [games[idx] for idx in order_desc]
        population = [population[idx] for idx in order_desc]

        # save best genome
        best_genome = population[0]
        logger.debug(f"best genome: {best_genome}")
        np.save(
            BEST_GENOMES_DIR.joinpath(f"best_genome_{ts}.npy"),
            best_genome,
        )

        # render plots
        best_fitness_history.append(np.max(fitness))
        avg_fitness_history.append(np.mean(fitness))
        best_game = games[0]
        renderer.render_history_plot(best_fitness_history, avg_fitness_history)
        renderer.render_genome_plot(
            best_genome,
            color=np.array(best_game.player.color) / 255,
            name=best_game.player.name,
        )
        pygame.display.update()

        # set next gen
        population = get_next_gen(population, progress=gen / NGENS)
        reset_games(games, population)
        pygame.time.delay(1000)

    pygame.quit()


if __name__ == "__main__":
    main()
