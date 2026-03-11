import logging
from pathlib import Path

import numpy as np
import pygame

from snake.engine import GAController, GAGame, Player
from snake.renderer import Renderer
from snake.state import RandomApple, Snake
from snake.utils import get_random_color, get_squared_wall

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

DEBUG = True

NCOLS = 30
NROWS = 20
GRID_SIZE = 20
WIDTH = NCOLS * GRID_SIZE
HEIGHT = NROWS * GRID_SIZE

FPS = 15
NPLAYERS = 3


def init_genomes() -> list[np.ndarray]:
    """Initialize genomes.

    Args:
        size: length of returned list of genomes

    Returns: list of genomes
    """
    custom_genome = np.concatenate([0.8 * np.eye(4), 0.5 * np.eye(4)])
    best_genome_path = Path("best_genome.npy")
    last_best_genome = np.load(best_genome_path)
    return [custom_genome, last_best_genome]


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
        apple = RandomApple(color)
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


def reset_games(games: list[GAGame]) -> None:
    """Reset games from list and set new GA controllers from list.

    Args:
        games: list of games
        genomes: list of genomes (arrays)
    """
    for game in games:
        game.reset()


def main() -> None:
    """Main human play function.

    Inits games.
    Renders frames of all games of all generations.
    """
    pygame.init()

    scoreboard_row_size = 20
    score_height = (NPLAYERS + 3) * scoreboard_row_size
    win = pygame.display.set_mode(size=(WIDTH, HEIGHT + score_height))
    pygame.display.set_caption("Snake")

    clock = pygame.time.Clock()

    game_rect = pygame.Rect(0, 0, WIDTH, HEIGHT)
    game_surf = win.subsurface(game_rect)

    score_rect = pygame.Rect(0, HEIGHT, WIDTH, score_height)
    score_surf = win.subsurface(score_rect)

    genomes = init_genomes()
    games = init_games(genomes)
    renderer = Renderer(
        game_surf,
        score_surf,
        NCOLS,
        NROWS,
        GRID_SIZE,
        rect_radius=int(GRID_SIZE / 4),
        line_width=2,
        font_size=14,
        scoreboard_row_size=scoreboard_row_size,
    )

    renderer.render_games(games)
    renderer.render_scoreboard(games)
    pygame.display.update()

    pygame.time.delay(1000)

    # game loop
    is_running = True
    is_paused = False
    while is_running:
        clock.tick(FPS)

        start_games(games)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    is_paused = not is_paused

                if event.key == pygame.K_q:
                    is_running = False

                if event.key == pygame.K_r and all(game.is_over for game in games):
                    reset_games(games)

        if not is_paused:
            for game in games:
                if game.has_started is False or game.is_over:
                    continue

                game.step()

            renderer.render_games(games)
            renderer.render_scoreboard(games)

        if is_paused and DEBUG:
            renderer.render_coords()
        if is_paused:
            renderer.render_paused()
        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()
