import numpy as np
import pygame

from snake.const import DOWN, GAME_HEIGHT, LEFT, RIGHT, SCORE_HEIGHT, UP, WIDTH
from snake.engine import Game, Player
from snake.renderer import render_frame
from snake.utils import get_random_color


class HumanPlayer(Player):
    def __init__(self, color: tuple[int]):
        super().__init__(color)

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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        keys = pygame.key.get_pressed()

        for key, direction in keymap.items():
            if keys[key]:
                return direction


def main():
    pygame.init()

    win = pygame.display.set_mode(size=(WIDTH, GAME_HEIGHT + SCORE_HEIGHT))

    game_rect = pygame.Rect(0, 0, WIDTH, GAME_HEIGHT)
    game_surf = win.subsurface(game_rect)

    score_rect = pygame.Rect(0, GAME_HEIGHT, WIDTH, SCORE_HEIGHT)
    score_surf = win.subsurface(score_rect)

    color = get_random_color()
    player = HumanPlayer(color)
    game = Game(player, color)

    pygame.display.set_caption("Snake")
    clock = pygame.time.Clock()

    run = True
    while run:
        # game loop
        pygame.time.delay(50)
        clock.tick(30)

        render_frame(game_surf, score_surf, game)
        direction = player.set_dir()
        run = game.step(direction)

    while True:
        event = pygame.event.wait()

        if event.type in [pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN]:
            break

    pygame.quit()


if __name__ == "__main__":
    main()
