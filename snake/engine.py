import logging
from abc import ABC, abstractmethod
from itertools import product

import numpy as np
import pygame

from .const import DOWN, LEFT, RIGHT, UP
from .state import Apple, DeterministicApple, RandomApple, Snake, Wall

logger = logging.getLogger(__name__)


class HumanController:
    _KEYMAP_IDX = 0
    KEYMAPS = [
        {
            "name": "arrows",
            "keymap_dict": {
                pygame.K_LEFT: LEFT,
                pygame.K_RIGHT: RIGHT,
                pygame.K_UP: UP,
                pygame.K_DOWN: DOWN,
            },
        },
        {
            "name": "asdw",
            "keymap_dict": {
                pygame.K_a: LEFT,
                pygame.K_d: RIGHT,
                pygame.K_w: UP,
                pygame.K_s: DOWN,
            },
        },
        {
            "name": "hjkl",
            "keymap_dict": {
                pygame.K_h: LEFT,
                pygame.K_l: RIGHT,
                pygame.K_k: UP,
                pygame.K_j: DOWN,
            },
        },
    ]

    def __init__(self):
        self.keymap_idx = self._KEYMAP_IDX
        self.keymap = self.KEYMAPS[self.keymap_idx]["keymap_dict"]
        self.keymap_name = self.KEYMAPS[self.keymap_idx]["name"]
        self.__class__._KEYMAP_IDX += 1

    def set_dir(self) -> np.ndarray | None:
        keys = pygame.key.get_pressed()

        for key, direction in self.keymap.items():
            if keys[key]:
                return direction


class GAController:
    def __init__(self, ncols: int, nrows: int, genome: np.ndarray):
        self.ncols = ncols
        self.nrows = nrows
        self.genome = genome

    def set_dir(self, features: np.ndarray) -> np.ndarray:
        scores = features @ self.genome  # linear combination
        move_idx = np.argmax(scores)

        directions = [UP, DOWN, LEFT, RIGHT]
        return directions[move_idx]

    def eval_state(self, snake: Snake, apple: Apple, wall: Wall) -> np.ndarray:
        AVAILABLE_NCOLS = self.ncols - 3
        AVAILABLE_NROWS = self.nrows - 3

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
        apple_dx, apple_dy = (apple.coords - snake.head_coords) / np.array(
            [AVAILABLE_NCOLS, AVAILABLE_NROWS]
        )

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
            "features:\n"
            f"  {wall_safety_right=}\n"
            f"  {wall_safety_left=}\n"
            f"  {wall_safety_down=}\n"
            f"  {wall_safety_up=}\n"
            f"  {body_safety_right=}\n"
            f"  {body_safety_left=}\n"
            f"  {body_safety_down=}\n"
            f"  {body_safety_up=}\n"
            f"  {safety_right=}\n"
            f"  {safety_left=}\n"
            f"  {safety_down=}\n"
            f"  {safety_up=}\n"
            f"  {apple_right=}\n"
            f"  {apple_left=}\n"
            f"  {apple_up=}\n"
            f"  {apple_down=}"
        )
        return features


class Player:
    _IDX = 0

    def __init__(self, color: tuple[int], controller, name: str | None = None):
        self.color = color
        self.controller = controller
        self.name = name or str(self._IDX)
        self.__class__._IDX += 1
        self.reset()

    def reset(self) -> None:
        self.score = 0


class Game(ABC):
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
        self.steps = 0
        self.coords_stepped = []
        self.dirs_to_apple = []
        self.player.reset()
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
            self.player.score += 1
            self.snake.extend()
            self.dirs_to_apple = []
            if isinstance(self.apple, RandomApple):
                self.apple.move(coords_choice=self._get_coords_for_random_apple())
            elif isinstance(self.apple, DeterministicApple):
                self.apple.move()
            else:
                raise Exception
            logger.debug("Apple eaten.")

        return self.active

    def move(self, direction: np.ndarray) -> None:
        if self.active:
            self.snake.move(direction)
            logger.debug(f"Snake moving in {direction}.")

    @abstractmethod
    def step(self) -> None:
        self.coords_stepped.append(self.snake.head_coords.copy())
        self.dirs_to_apple.append(self.snake.head_dir)
        self.steps += 1


class HumanGame(Game):
    def __init__(
        self,
        ncols: int,
        nrows: int,
        player: Player,
        wall: Wall,
        snake: Snake,
        apple: Apple,
    ):
        super().__init__(ncols, nrows, player, wall, snake, apple)

    def step(self) -> None:
        direction = self.player.controller.set_dir()
        self.move(direction)
        self.eval_state()
        super().step()


class GAGame(Game):
    def __init__(
        self,
        ncols: int,
        nrows: int,
        player: Player,
        wall: Wall,
        snake: Snake,
        apple: Apple,
    ):
        super().__init__(ncols, nrows, player, wall, snake, apple)

    def step(self) -> None:
        features = self.player.controller.eval_state(self.snake, self.apple, self.wall)
        direction = self.player.controller.set_dir(features)
        self.move(direction)
        self.eval_state()
        super().step()
