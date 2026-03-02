from itertools import product

import numpy as np
from names_generator import generate_name

from .const import GRID_SIZE
from .state import Apple, Snake, Wall
from .utils import get_random_color


class Player:
    def __init__(self, color: tuple[int], controller, name: str | None = None):
        self.controller = controller
        self.name = name or generate_name(style="capital")


class Game:
    def __init__(
        self,
        width: int,
        height: int,
        player: Player,
        wall: Wall,
        color: tuple[int, int, int] | None = None,
        snake: Snake | None = None,
        apple: Apple | None = None,
    ):
        self.width = width
        self.height = height
        self.player = player
        self.wall = wall
        self.color = color or get_random_color()
        self.snake = snake or Snake(self.color)
        self.apple = apple or Apple(
            self.color,
            available_coords=self._get_available_apple_coords(),
            exclude_coords=self.wall.coords + self.snake.coords,
        )
        self.reset()

    def reset(self) -> None:
        self.snake.reset()
        self.apple.reset(exclude_coords=self.wall.coords + self.snake.coords)
        self.score = 0
        self.steps = 0
        self.active = True

    def _get_available_apple_coords(self):
        xs = GRID_SIZE * np.arange(int(self.width / GRID_SIZE))
        ys = GRID_SIZE * np.arange(int(self.height / GRID_SIZE))
        return [np.array(c) for c in product(xs, ys)]

    def eval_state(self) -> bool:

        # wall collision
        if any(np.array_equal(self.snake.head_coords, wc) for wc in self.wall.coords):
            self.active = False
            print("Wall collision.")

        # self collision
        if any(
            np.array_equal(self.snake.head_coords, bc) for bc in self.snake.body_coords
        ):
            self.active = False
            print("Self collision.")

        if np.array_equal(self.snake.head_coords, self.apple.coords):
            self.score += 1
            self.snake.extend()
            self.apple.move(exclude_coords=self.wall.coords + self.snake.coords)
            print("Apple eaten.")

        return self.active

    def move(self, direction: np.ndarray) -> None:
        if self.active:
            self.snake.move(direction)
            print("Snake moving.")
            self.steps += 1
