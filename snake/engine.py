import logging
from itertools import product

import numpy as np

from .state import Apple, DeterministicApple, RandomApple, Snake, Wall

logger = logging.getLogger(__name__)


class Player:
    def __init__(self, color: tuple[int], controller, name: str | None = None):
        self.color = color
        self.controller = controller
        self.name = name


class Game:
    def __init__(
        self,
        ncols: int,
        nrows: int,
        player: Player,
        wall: Wall,
        snake: Snake,
        apple: Apple,
    ):
        self.ncols = ncols
        self.nrows = nrows
        self.player = player
        self.wall = wall
        self.snake = snake
        self.apple = apple
        self.reset()

    def reset(self) -> None:
        self.active = True
        self.score = 0
        self.steps = 0
        self.coords_stepped = []
        self.snake.reset()
        self.apple.reset()

    def _get_coords_for_random_apple(self):
        xs = np.arange(int(self.ncols))
        ys = np.arange(int(self.nrows))
        exclude_coords = self.wall.coords + self.snake.coords

        coords = []
        for c in product(xs, ys):
            arr = np.array(c)
            if not np.all(exclude_coords == arr, axis=1).any():
                coords.append(arr)

        return coords

    def eval_state(self) -> bool:

        # wall collision
        if np.all(self.wall.coords == self.snake.head_coords, axis=1).any():
            self.active = False
            logger.debug("Wall collision.")

        # self collision
        if np.all(self.snake.body_coords == self.snake.head_coords, axis=1).any():
            self.active = False
            logger.debug("Self collision.")

        if np.all(self.snake.head_coords == self.apple.coords):
            self.score += 1
            self.snake.extend()
            if isinstance(self.apple, RandomApple):
                self.apple.move(coords_choice=self._get_coords_for_random_apple())
            elif isinstance(self.apple, DeterministicApple):
                self.apple.move(idx=self.score)
            else:
                raise Exception
            logger.debug("Apple eaten.")

        return self.active

    def move(self, direction: np.ndarray) -> None:
        if self.active:
            self.snake.move(direction)
            logger.debug(f"Snake moving in {direction}.")
            self.steps += 1
            self.coords_stepped.append(self.snake.head_coords.copy())
