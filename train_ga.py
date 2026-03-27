import datetime
import logging
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
NSTEPS = 400

BEST_GENOMES_DIR = Path("best_genomes")

SHAPE = (len(GAController.FEATURE_NAMES), len(DIRECTIONS))


def init_population(size: int) -> list[np.ndarray]:
    """Initialize population / genomes.

    Args:
        size: length of returned list of genomes

    Returns: population - list of genomes(arrays)
    """
    return [rng.choice([1.0, 0.0, -1.0, 0.5, -0.5], size=SHAPE) for _ in range(size)]


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

    # # steps penalty: 0 if the most efficient path
    # # otherwise linearly increasing
    # steps_penalty = max(0, game.steps / np.sum(game.apple.min_nsteps_needed) - 1)
    # fitness -= steps_penalty
    # info["steps_penalty"] = steps_penalty

    # cycling penalty: 0 if all last steps were unique, otherwise linearly increasing
    last_steps = game.coords_stepped[-100:]
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
    sorted_population: list[np.ndarray], progress: float
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

    ngenomes = len(sorted_population)
    nelites = int(0.05 * ngenomes)

    elites = sorted_population[:nelites]
    rest = sorted_population[nelites:]
    top_half = sorted_population[nelites : int(0.50 * ngenomes)]

    # keep elites unchanged
    next_gen = elites.copy()

    mutated_genomes = get_mutated_genomes(
        elites, size=int(0.10 * ngenomes), progress=progress
    )
    next_gen.extend(mutated_genomes)

    mutated_genomes = get_mutated_genomes(
        top_half, size=int(0.30 * ngenomes), progress=progress
    )
    next_gen.extend(mutated_genomes)

    crossover_genomes = get_crossover_genomes(elites, elites, size=int(0.10 * ngenomes))
    next_gen.extend(crossover_genomes)

    crossover_genomes = get_crossover_genomes(elites, rest, size=int(0.35 * ngenomes))
    next_gen.extend(crossover_genomes)

    # inject random immigrants
    next_gen.extend(init_population(size=int(0.10 * ngenomes)))

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
    for gen in range(1, NGENS + 1):
        logger.info(f"New gen {gen}")

        renderer.render_games(games[::-1], alphas=_get_alphas(NGENOMES))
        renderer.render_scoreboard(games, gen)
        pygame.display.update()

        pygame.time.delay(1000)

        recorder = ScreenRecorder(FPS).start_rec()
        apple_coords_generated = []

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
                        coords = apple_coords_generated[game.apple.idx]
                        game.apple.move(coords)

                    if game.apple.idx >= len(apple_coords_generated):
                        xrange = np.arange(NCOLS)
                        yrange = np.arange(NROWS)
                        exclude = get_exclude_coords(games)
                        new_apple_coords = get_free_coords(xrange, yrange, exclude)
                        apple_coords_generated.append(new_apple_coords)

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
        renderer.render_history_plot(
            np.max(fitness_history, axis=1), np.mean(fitness_history, axis=1)
        )

        renderer.render_genome_plot(
            population[best_idx],
            color=np.array(games[best_idx].player.color) / 255,
            name=games[best_idx].player.name,
            fitness=fitness[best_idx],
        )
        pygame.display.update()

        if gen % 20 == 0:
            # eval and sort in place
            avg_fitness = np.mean(np.transpose(fitness_history[-5:]), axis=1)
            order_desc = np.argsort(avg_fitness)[::-1]
            games = [games[idx] for idx in order_desc]
            population = [population[idx] for idx in order_desc]

            # save best genome
            best_genome = population[0]
            logger.debug(f"best genome: {best_genome}")
            np.save(
                BEST_GENOMES_DIR.joinpath(f"best_genome_{ts}.npy"),
                best_genome,
            )

            # set next gen
            population = get_next_gen(population, progress=gen / NGENS)
            set_population(games, population)

        reset_games(games)

        pygame.time.delay(1000)

    pygame.quit()


if __name__ == "__main__":
    main()
