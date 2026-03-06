import logging

import pygame

from snake.engine import Game, HumanController, HumanGame, Player
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

FPS = 1
NPLAYERS = 3


def init_games() -> Game:
    wall = get_squared_wall(NCOLS, NROWS)
    games = []
    for idx in range(NPLAYERS):
        color = get_random_color()
        controller = HumanController()
        player = Player(color, controller, name=str(idx + 1))
        snake = Snake(color)
        apple = RandomApple(color)
        game = HumanGame(NCOLS, NROWS, player, wall, snake, apple)
        games.append(game)

    return games


def main():
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
        scoreboard_row_size=scoreboard_row_size,
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

    # game loop
    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        for game in games:
            if game.active is False:
                continue

            game.step()

        renderer.render_games(games)
        renderer.render_scoreboard(games)
        pygame.display.update()

        if all(game.active is False for game in games):
            break

    while True:
        event = pygame.event.wait()

        if event.type in [pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN]:
            break

    pygame.quit()


if __name__ == "__main__":
    main()
