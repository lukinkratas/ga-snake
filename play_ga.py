import logging
from pathlib import Path

import numpy as np
import pygame

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

DEBUG = True

NCOLS = 30
NROWS = 20
GRID_SIZE = 20
WIDTH = NCOLS * GRID_SIZE
HEIGHT = NROWS * GRID_SIZE

FPS = 15
NPLAYERS = 3

BEST_GENOMES_DIR = Path("best_genomes")


def init_genomes() -> list[np.ndarray]:
    """Initialize genomes.

    Args:
        size: length of returned list of genomes

    Returns: list of genomes
    """
    custom_genome = np.clip(
        np.array(
            [
                [-0.3, 0.1, 0.1, 0.1],
                [0.1, -0.3, 0.1, 0.1],
                [0.1, 0.1, -0.3, 0.1],
                [0.1, 0.1, 0.1, -0.3],
                [1.0, 0, 0, 0],
                [0, 1.0, 0, 0],
                [0, 0, 1.0, 0],
                [0, 0, 0, 1.0],
            ]
        )
        + rng.uniform(-0.1, 0.1, size=(8, 4)),
        -1,
        1,
    )
    best_genomes = [
        np.load(genome_npy)
        for genome_npy in BEST_GENOMES_DIR.iterdir()
        if str(genome_npy).endswith(".npy")
    ]
    return [custom_genome] + best_genomes


def init_games(genomes: list[np.ndarray]) -> list[GAGame]:
    """Initialize games with it's assets - player, controller, wall, snake and apple.

    Args:
        genomes: list of genomes

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
        init_game(genome, f"G{gidx}") for gidx, genome in enumerate(genomes, start=1)
    ]


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
        font_size=13,
    )

    renderer.render_games(games)
    renderer.render_scoreboard(games)
    pygame.display.update()

    pygame.time.delay(1000)

    # game loop
    is_running = True
    is_paused = False
    apple_coords_generated = []
    while is_running:
        clock.tick(FPS)

        for game in games:
            game.has_started = True

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
                if game.has_started is True and not game.is_over:
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

            sorted_games_desc = sorted(
                games,
                key=lambda g: (not g.is_over, g.player.score),
                reverse=True,
            )
            renderer.render_games(sorted_games_desc[::-1])
            renderer.render_scoreboard(sorted_games_desc)

        if is_paused and DEBUG:
            renderer.render_coords()
        if is_paused:
            renderer.render_paused()
        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()
