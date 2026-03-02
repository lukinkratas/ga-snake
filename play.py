import numpy as np
import pygame

from snake.const import DOWN, GRID_SIZE, LEFT, RIGHT, UP
from snake.engine import Game, Player
from snake.renderer import Renderer
from snake.utils import get_random_color, get_squared_wall

COLS = 30
GAME_ROWS = 20
WIDTH = COLS * GRID_SIZE
GAME_HEIGHT = GAME_ROWS * GRID_SIZE

FPS = 15
NPLAYERS = 1

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


def init_game(width: int, height: int) -> Game:
    color = get_random_color()
    controller = HumanController()
    player = Player(color, controller)
    wall = get_squared_wall(width, height, GRID_SIZE)
    return Game(width, height, player, wall, color)


def main():
    pygame.init()

    win = pygame.display.set_mode(size=(WIDTH, GAME_HEIGHT + SCORE_HEIGHT))
    pygame.display.set_caption("Snake")

    clock = pygame.time.Clock()

    game_rect = pygame.Rect(0, 0, WIDTH, GAME_HEIGHT)
    game_surf = win.subsurface(game_rect)

    score_rect = pygame.Rect(0, GAME_HEIGHT, WIDTH, SCORE_HEIGHT)
    score_surf = win.subsurface(score_rect)

    game = init_game(WIDTH, GAME_HEIGHT)
    renderer = Renderer(
        game_surf,
        score_surf,
        rect_radius=int(GRID_SIZE / 4),
        line_width=1,
        font_size=10,
    )
    renderer.render_frame([game], debug=True)

    while True:
        event = pygame.event.wait()

        if event.type in [pygame.KEYDOWN]:
            break

    # game loop
    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        direction = game.player.controller.set_dir()
        game.move(direction)
        game.eval_state()

        renderer.render_frame([game])
        running = game.active

    while True:
        event = pygame.event.wait()

        if event.type in [pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN]:
            break

    pygame.quit()


if __name__ == "__main__":
    main()
