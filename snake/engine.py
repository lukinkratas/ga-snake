from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

from .state import Apple, Snake, Wall
from .utils import get_random_color


@lru_cache
def _get_wall():
    """Lazy init the same wall object for multiple snakes and apples."""
    return Wall()


class Player(ABC):
    def __init__(self, color: tuple[int]):
        self.color = color
        self.score = 0
        self.snake = Snake(color)

    @abstractmethod
    def set_dir(self) -> np.ndarray | None:
        pass


class Game:
    def __init__(
        self,
        player: Player,
        color: tuple[int, int, int] | None = None,
        apple: Apple | None = None,
    ):
        self.player = player
        self.color = color or get_random_color()
        self.apple = apple or Apple(self.color)
        self.wall = _get_wall()

    def eval_state(self) -> bool:

        snake = self.player.snake

        # if snake.alive is False:
        #     continue

        # wall collision
        if snake.head_rect.collidelist(self.wall.rects) != -1:
            snake.alive = False
            print("Wall collision.")

        # self collision
        if snake.head_rect.collidelist(
            snake.body_rects
        ) != -1 or snake.head_rect.colliderect(snake.tail_rect):
            snake.alive = False
            print("Self collision.")

        if snake.head_rect.colliderect(self.apple.rect):
            self.player.score += 1
            snake.extend()
            self.apple.move(exclude=snake.rects)
            print("Apple eaten.")

        if snake.alive is False:
            return False

        return True

    def step(self, direction: np.ndarray) -> bool:
        run = self.eval_state()

        if self.player.snake.alive:
            self.player.snake.move(direction)
            print("Snake moving.")

        return run
