import logging

import numpy as np
import pygame

from snake.const import DOWN, LEFT, RIGHT, UP
from snake.engine import Game, Player
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
GAME_HEIGHT = NROWS * GRID_SIZE

FPS = 15
NPLAYERS = 3

SCORE_HEIGHT = 100


class HumanController:
    def set_dir(self) -> np.ndarray | None:
        keymap = {
            pygame.K_LEFT: LEFT,
            pygame.K_h: LEFT,
            pygame.K_a: LEFT,
            pygame.K_RIGHT: RIGHT,
            pygame.K_l: RIGHT,
            pygame.K_d: RIGHT,
            pygame.K_UP: UP,
            pygame.K_k: UP,
            pygame.K_w: UP,
            pygame.K_DOWN: DOWN,
            pygame.K_j: DOWN,
            pygame.K_s: DOWN,
        }

        keys = pygame.key.get_pressed()

        for key, direction in keymap.items():
            if keys[key]:
                return direction


def init_games() -> Game:
    wall = get_squared_wall(NCOLS, NROWS)
    games = []
    for idx in range(NPLAYERS):
        color = get_random_color()
        controller = HumanController()
        player = Player(color, controller, name=str(idx + 1))
        snake = Snake(color)
        apple = RandomApple(color)
        game = Game(NCOLS, NROWS, player, wall, snake, apple)
        games.append(game)

    return games


def main():
    pygame.init()

    win = pygame.display.set_mode(size=(WIDTH, GAME_HEIGHT + SCORE_HEIGHT))
    pygame.display.set_caption("Snake")

    clock = pygame.time.Clock()

    game_rect = pygame.Rect(0, 0, WIDTH, GAME_HEIGHT)
    game_surf = win.subsurface(game_rect)

    score_rect = pygame.Rect(0, GAME_HEIGHT, WIDTH, SCORE_HEIGHT)
    score_surf = win.subsurface(score_rect)

    games = init_games()
    renderer = Renderer(
        game_surf,
        score_surf,
        NCOLS,
        NROWS,
        GRID_SIZE,
        rect_radius=int(GRID_SIZE / 4),
        line_width=2,
        font_size=14,
    )
    renderer.render_frame(games, debug=DEBUG)

    while True:
        event = pygame.event.wait()

        if event.type in [pygame.QUIT]:
            pygame.quit()

        if event.type in [pygame.KEYDOWN]:
            break

    # game loop
    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        for game in games:
            direction = game.player.controller.set_dir()
            game.move(direction)
            game.eval_state()

        renderer.render_frame(games)

        if all(game.active is False for game in games):
            break

    while True:
        event = pygame.event.wait()

        if event.type in [pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN]:
            break

    pygame.quit()


if __name__ == "__main__":
    main()
