import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import pygame
from numpy import linalg as LA

from snake.const import DOWN, LEFT, RIGHT, UP
from snake.engine import Game, Player
from snake.renderer import Renderer
from snake.state import Apple, DeterministicApple, Snake, Wall
from snake.utils import get_random_color, get_squared_wall

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

DEBUG = True

NCOLS = 30
NROWS = 20
GRID_SIZE = 15
WIDTH = NCOLS * GRID_SIZE
GAME_HEIGHT = NROWS * GRID_SIZE

FPS = 120
POP_SIZE = 22
NGENS = 1000
# limit number of steps per game
# NSTEPS = 300

SCORE_HEIGHT = 350


class GAController:
    def __init__(self, genome: np.ndarray):
        self.genome = genome

    def set_dir(self, features: np.ndarray) -> np.ndarray:
        scores = features @ self.genome  # linear combination
        move_idx = np.argmax(scores)

        directions = [UP, DOWN, LEFT, RIGHT]
        return directions[move_idx]

    def eval_state(self, snake: Snake, apple: Apple, wall: Wall) -> np.ndarray:
        AVAILABLE_NCOLS = NCOLS - 3
        AVAILABLE_NROWS = NROWS - 3

        head_x, head_y = snake.head_coords

        # Distances to walls
        wall_right_rects_dists = (
            np.array([c[0] for c in wall.coords if c[1] == head_y and c[0] > head_x])
            - head_x
            - 1
        )
        # No bodies rects found on the right
        wall_safety_right = (
            np.min(wall_right_rects_dists) / AVAILABLE_NCOLS
            if wall_right_rects_dists.size != 0
            else np.float64(1)
        )

        # Distances to snake body
        body_right_rects_dists = (
            np.array(
                [c[0] for c in snake.body_coords if c[1] == head_y and c[0] > head_x]
            )
            - head_x
            - 1
        )
        body_safety_right = (
            np.min(body_right_rects_dists) / AVAILABLE_NCOLS
            if body_right_rects_dists.size != 0
            else np.float64(1)
        )

        wall_left_rects_dists = (
            head_x
            - 1
            - np.array([c[0] for c in wall.coords if c[1] == head_y and c[0] < head_x])
        )

        wall_safety_left = (
            np.min(wall_left_rects_dists) / AVAILABLE_NCOLS
            if wall_left_rects_dists.size != 0
            else np.float64(1)
        )

        body_left_rects_dists = (
            head_x
            - 1
            - np.array(
                [c[0] for c in snake.body_coords if c[1] == head_y and c[0] < head_x]
            )
        )

        body_safety_left = (
            np.min(body_left_rects_dists) / AVAILABLE_NCOLS
            if body_left_rects_dists.size != 0
            else np.float64(1)
        )

        wall_up_rects_dists = (
            head_y
            - 1
            - np.array([c[1] for c in wall.coords if c[0] == head_x and c[1] < head_y])
        )

        wall_safety_up = (
            np.min(wall_up_rects_dists) / AVAILABLE_NROWS
            if wall_up_rects_dists.size != 0
            else np.float64(1)
        )

        body_up_rects_dists = (
            head_y
            - 1
            - np.array(
                [c[1] for c in snake.body_coords if c[0] == head_x and c[1] < head_y]
            )
        )

        body_safety_up = (
            np.min(body_up_rects_dists) / AVAILABLE_NROWS
            if body_up_rects_dists.size != 0
            else np.float64(1)
        )

        wall_down_rects_dists = (
            np.array([c[1] for c in wall.coords if c[0] == head_x and c[1] > head_y])
            - head_y
            - 1
        )

        wall_safety_down = (
            np.min(wall_down_rects_dists) / AVAILABLE_NROWS
            if wall_down_rects_dists.size != 0
            else np.float64(1)
        )

        body_down_rects_dists = (
            np.array(
                [c[1] for c in snake.body_coords if c[0] == head_x and c[1] > head_y]
            )
            - head_y
            - 1
        )

        body_safety_down = (
            np.min(body_down_rects_dists) / AVAILABLE_NROWS
            if body_down_rects_dists.size != 0
            else np.float64(1)
        )

        safety_right = np.min([wall_safety_right, body_safety_right])
        safety_left = np.min([wall_safety_left, body_safety_left])
        safety_up = np.min([wall_safety_up, body_safety_up])
        safety_down = np.min([wall_safety_down, body_safety_down])

        # Distances to apple
        apple_dx = (apple.coords[0] - head_x) / AVAILABLE_NCOLS
        apple_dy = (apple.coords[1] - head_y) / AVAILABLE_NROWS

        apple_right = np.max((0, apple_dx))
        apple_left = np.max((0, -apple_dx))
        apple_up = np.max((0, -apple_dy))
        apple_down = np.max((0, apple_dy))

        features = np.array(
            [
                safety_left,
                safety_right,
                safety_up,
                safety_down,
                apple_right,
                apple_left,
                apple_up,
                apple_down,
            ]
        )

        logger.debug(
            f"features: {wall_safety_right=} {wall_safety_left=} {wall_safety_down=} {wall_safety_up=} {body_safety_right=} {body_safety_left=} {body_safety_down=} {body_safety_up=} {safety_right=} {safety_left=} {safety_down=} {safety_up=} {apple_right=} {apple_left=} {apple_up=} {apple_down=}"
        )
        return features


def init_genomes(size: int) -> list[np.ndarray]:
    return [np.random.uniform(-1, 1, (8, 4)) for _ in range(size)]


def init_games(genomes: list[np.ndarray]) -> list[Game]:
    # common wall for all games
    wall = get_squared_wall(NCOLS, NROWS)
    games = []
    for idx, genome in enumerate(genomes, start=1):
        color = get_random_color()
        controller = GAController(genome)
        player = Player(color, controller, name=str(idx))
        snake = Snake(color)
        apple = DeterministicApple(color)
        game = Game(NCOLS, NROWS, player, wall, snake, apple)
        games.append(game)

    return games


def reset(games: list[Game], genomes: list[np.ndarray]) -> None:
    for genome, game in zip(genomes, games):
        game.reset()
        game.player.controller = GAController(genome)


def eval_fitness(game: Game, max_steps: int) -> np.ndarray:
    logger.debug(f"Evaluating {game.player.name} fitness")
    STEPS_THRESHOLD = 100
    logger.debug(f"{game.score=}")
    logger.debug(f"{game.active=}")
    fitness = 10 * game.score - 5 * (game.active is False)

    if max_steps > STEPS_THRESHOLD:
        last_coords_stepped = game.coords_stepped[:STEPS_THRESHOLD]
        # unique steps repeating == stuck only over a few cells
        num_unique_coords_stepped = np.unique(last_coords_stepped, axis=0).shape[0]
        logger.debug(
            f"Snake stepped unique {num_unique_coords_stepped} steps out of {STEPS_THRESHOLD}."
        )
        # reward exploration
        fitness += num_unique_coords_stepped / STEPS_THRESHOLD

    # reward distance to apple
    apple_diff = 1 - LA.norm(game.apple.coords - game.snake.head_coords, 2) / LA.norm(
        np.array([NCOLS, NROWS]), 2
    )
    fitness += apple_diff
    logger.debug(f"{apple_diff=}")
    logger.debug(f"{fitness=}")

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
    MUTATION_RATE = 0.1
    MUTATION_SCALE = 0.2
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

    win = pygame.display.set_mode(size=(WIDTH, GAME_HEIGHT + SCORE_HEIGHT))
    pygame.display.set_caption("Snake")

    clock = pygame.time.Clock()

    game_rect = pygame.Rect(0, 0, WIDTH, GAME_HEIGHT)
    game_surf = win.subsurface(game_rect)

    score_rect = pygame.Rect(0, GAME_HEIGHT, WIDTH, SCORE_HEIGHT)
    score_surf = win.subsurface(score_rect)

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
    )
    renderer.render_frame(games, debug=DEBUG)

    if DEBUG:
        plt.ion()
        fig, ax = plt.subplots()

    while True:
        event = pygame.event.wait()

        if event.type in [pygame.QUIT]:
            pygame.quit()

        if event.type in [pygame.KEYDOWN]:
            break

    # pygame.time.delay(1000)

    fitness_history = []
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

                features = game.player.controller.eval_state(
                    game.snake, game.apple, game.wall
                )
                direction = game.player.controller.set_dir(features)
                game.move(direction)
                game.eval_state()

            renderer.render_frame(games, gen)

            if all(game.active is False for game in games):
                break

        fitness = [eval_fitness(game, max_steps) for game in games]
        fitness_history.append(np.max(fitness))
        sorted_genomes = sort_genomes_by_fitness(games, fitness)
        nelites = max(3, round(0.15 * POP_SIZE))
        elites = sorted_genomes[:nelites]
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
                elites, top_half, size=POP_SIZE - len(next_gen_genomes)
            )
            next_gen_genomes.extend(crossover_genomes)

        pygame.time.delay(1000)
        reset(games, next_gen_genomes)
        genomes = next_gen_genomes

        if DEBUG:
            ax.clear()
            ax.plot(fitness_history)
            ax.set_title("Best Fitness per Generation")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Fitness")

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

    while True:
        event = pygame.event.wait()

        if event.type in [pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN]:
            break

    pygame.quit()


if __name__ == "__main__":
    main()
