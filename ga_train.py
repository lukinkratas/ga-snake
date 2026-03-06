import logging
import random

import matplotlib
import numpy as np
import pygame
from numpy import linalg as LA

from snake.engine import GAController, GAGame, Game, Player
from snake.renderer import Renderer
from snake.state import DeterministicApple, Snake
from snake.utils import get_random_color, get_squared_wall

matplotlib.use("pygame")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

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
    return [np.random.uniform(-0.1, 0.1, (8, 4)) for _ in range(size)]


def init_games(genomes: list[np.ndarray]) -> list[Game]:
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


def reset(games: list[Game], genomes: list[np.ndarray]) -> None:
    for genome, game in zip(genomes, games):
        game.reset()
        game.player.controller = GAController(NCOLS, NROWS, genome)


def eval_fitness(game: Game, max_steps: int) -> np.ndarray:
    logger.debug(f"Evaluating {game.player.name} fitness")
    # STEPS_THRESHOLD = 100
    score_factor = 10 * game.player.score
    active_factor = -5 * (game.active is False)
    fitness = score_factor + active_factor

    # last_coords_stepped = game.coords_stepped[:STEPS_THRESHOLD]
    # unique steps repeating == stuck only over a few cells
    num_unique_coords_stepped = np.unique(game.coords_stepped, axis=0).shape[0]
    # reward exploration
    unique_coords_factor = num_unique_coords_stepped / max_steps
    fitness += unique_coords_factor

    # reward distance to last apple
    apple_dist_factor = 1 - LA.norm(
        game.apple.coords - game.snake.head_coords, 2
    ) / LA.norm([NCOLS, NROWS], 2)

    fitness += apple_dist_factor

    # reward directions towards apple
    prev_apple = (
        game.apple._COORDS[game.apple.idx - 1]
        if game.apple.idx > 0
        else game.snake.INIT_HEAD_COORDS
    )
    vec_to_apple = game.apple.coords - prev_apple
    # normalize to 1 (div by len)
    apple_dir = np.where(
        vec_to_apple != 0, vec_to_apple / np.abs(vec_to_apple), vec_to_apple
    )
    # apple_dir can be: [1, 0], [0, 1], or even [1, 1]
    if np.abs(apple_dir).sum() == 2:
        apple_dirs = apple_dir
    else:
        apple_dir_x, apple_dir_y = apple_dir
        apple_dirs = [np.array([apple_dir_x, 0]), np.array([0, apple_dir_y])]
    print(f"{apple_dirs=}")
    print(f"{game.dirs_to_apple}")
    # /2 bcs extract automatically flattens
    dir_matches = np.count_nonzero(np.isin(game.dirs_to_apple, apple_dirs), axis=1)
    apple_dir_factor = np.extract(dir_matches == 2, dir_matches).shape[0] / len(
        game.dirs_to_apple
    )

    logger.debug(
        f"{fitness=}\n"
        f"  {score_factor=}\n"
        f"  {active_factor=}\n"
        f"  {unique_coords_factor=}\n"
        f"  {apple_dist_factor=}\n"
        f"  {apple_dir_factor=}"
    )

    return fitness


def sort_genomes_by_fitness(
    games: list[Game], fitness: list[np.ndarray]
) -> list[np.ndarray]:
    genomes = [game.player.controller.genome for game in games]
    # sort by fitness
    elite_idxs = np.argsort(fitness)[::-1]
    # select top 5 as elite
    return [genomes[idx] for idx in elite_idxs]


def mutate(genome: np.ndarray) -> np.ndarray:
    # TODO exponential decay exploration -> exploitation
    # MUTATION_SCALE = max(0.05, 0.2 * (1 - gen / NGENS))
    logger.debug("Mutating genome.")
    # MUTATION_RATE = 0.1
    # MUTATION_SCALE = 0.2
    MUTATION_RATE = 0.2
    MUTATION_SCALE = 0.4
    mask = np.random.uniform(0, 1, genome.shape) < MUTATION_RATE
    noise = np.random.uniform(-1, 1, genome.shape) * MUTATION_SCALE
    new_arr = genome.copy()
    new_arr[mask] += noise[mask]
    return new_arr


def get_mutated_genomes(
    genomes_choice: list[np.ndarray], size: int
) -> list[np.ndarray]:
    genomes = []
    for _ in range(size):
        parent = random.choice(genomes_choice)
        child_genome = mutate(parent)
        genomes.append(child_genome)
    return genomes


def crossover(genome_a: np.ndarray, genome_b: np.ndarray) -> np.ndarray:
    logger.debug("Crossovering genomes.")
    mask = np.random.uniform(0, 1, genome_a.shape) < 0.5
    return np.where(mask, genome_a, genome_b)


def get_crossover_genomes(
    genomes_a_choice: list[np.ndarray], genomes_b_choice: list[np.ndarray], size: int
) -> list[np.ndarray]:
    genomes = []
    for _ in range(size):
        parent_a = random.choice(genomes_a_choice)
        parent_b = random.choice(genomes_b_choice)
        child_genome = crossover(parent_a, parent_b)
        genomes.append(child_genome)
    return genomes


def main():
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
    if DEBUG:
        renderer.render_coords()
    pygame.display.update()

    while True:
        event = pygame.event.wait()

        if event.type in [pygame.QUIT]:
            pygame.quit()

        if event.type in [pygame.KEYDOWN]:
            break

    # pygame.time.delay(1000)

    best_fitness_history = []
    for gen in range(1, NGENS + 1):
        logger.info(f"New gen {gen}")

        max_steps = gen * 10
        # game loop
        for _ in range(max_steps):
            clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            for game in games:
                if game.active is False:
                    continue

                game.step()

            renderer.render_games(games, gen)
            renderer.render_scoreboard(games, gen)
            pygame.display.update()

            if all(game.active is False for game in games):
                break

        fitness = [eval_fitness(game, max_steps) for game in games]
        best_fitness_history.append(np.max(fitness))
        sorted_genomes = sort_genomes_by_fitness(games, fitness)
        nelites = max(3, round(0.15 * POP_SIZE))  # 15%, at least 3
        elites = sorted_genomes[:nelites]
        rest = sorted_genomes[nelites:]
        top_half = sorted_genomes[nelites : int(0.5 * POP_SIZE)]

        # keep elites unchanged
        next_gen_genomes = elites.copy()

        if len(next_gen_genomes) < POP_SIZE:
            mutated_genomes = get_mutated_genomes(elites, size=int(0.15 * POP_SIZE))
            next_gen_genomes.extend(mutated_genomes)

        if len(next_gen_genomes) < POP_SIZE:
            mutated_genomes = get_mutated_genomes(top_half, size=int(0.15 * POP_SIZE))
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
        reset(games, next_gen_genomes)
        genomes = next_gen_genomes

        if DEBUG:
            renderer.render_plot(best_fitness_history)
            pygame.display.update()

    while True:
        event = pygame.event.wait()

        if event.type in [pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN]:
            break

    pygame.quit()


if __name__ == "__main__":
    main()
