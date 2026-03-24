import logging
from abc import ABC, abstractmethod
from itertools import product

import numpy as np
import pygame

from .const import DIRECTIONS, DOWN, LEFT, RIGHT, UP
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
            "name": "A|S|D|W",
            "keymap_dict": {
                pygame.K_a: LEFT,
                pygame.K_d: RIGHT,
                pygame.K_w: UP,
                pygame.K_s: DOWN,
            },
        },
        {
            "name": "vim-like H|J|K|L",
            "keymap_dict": {
                pygame.K_h: LEFT,
                pygame.K_l: RIGHT,
                pygame.K_k: UP,
                pygame.K_j: DOWN,
            },
        },
    ]

    def __init__(self) -> None:
        self.keymap_idx = self._KEYMAP_IDX
        self.keymap = self.KEYMAPS[self.keymap_idx]["keymap_dict"]
        self.keymap_name = self.KEYMAPS[self.keymap_idx]["name"]
        self.__class__._KEYMAP_IDX += 1

    def set_dir(self) -> np.ndarray | None:
        """Set direction for snake from keys pressed.

        Returns: direction (array) or None
        """
        keys = pygame.key.get_pressed()

        for key, direction in self.keymap.items():
            if keys[key]:
                return direction

        return None


class GAController:
    FEATURE_NAMES = [
        "danger_right",
        "danger_left",
        "danger_down",
        "danger_up",
        "apple_right",
        "apple_left",
        "apple_down",
        "apple_up",
    ]

    def __init__(self, ncols: int, nrows: int, genome: np.ndarray) -> None:
        self.ncols = ncols
        self.nrows = nrows
        self.genome = genome

    def set_dir(self, features: np.ndarray) -> np.ndarray:
        """Set direction for snake from features.

        Args:
            features: features to guide direction decision (array)

        Returns: direction (array) or None
        """
        scores = features @ self.genome  # linear combination
        move_idx = np.argmax(scores)

        return list(DIRECTIONS.values())[move_idx]

    def eval_state(self, snake: Snake, apple: Apple, wall: Wall) -> np.ndarray:
        """Evaluate the state of snake, apple and wall.

        Args:
            snake: snake
            apple: apple
            wall: wall

        Returns: features (array)
        """
        available_ncols = self.ncols - 3
        available_nrows = self.nrows - 3

        head_x, head_y = snake.head_coords

        # Distances to walls
        wall_right_rects_xs = np.array(
            [c[0] for c in wall.coords if c[1] == head_y and c[0] > head_x]
        )
        # danger: 1 if wall is next to head, linearly decreasing
        if wall_right_rects_xs.size != 0:
            # No bodies rects found on the right
            wall_danger_right = (
                1 - np.min(wall_right_rects_xs - head_x - 1) / available_ncols
            )
        else:
            wall_danger_right = np.float64(0)

        # Distances to snake body
        body_right_rects_xs = np.array(
            [c[0] for c in snake.body_coords if c[1] == head_y and c[0] > head_x]
        )
        if body_right_rects_xs.size != 0:
            body_danger_right = (
                1 - np.min(body_right_rects_xs - head_x - 1) / available_ncols
            )
        else:
            body_danger_right = np.float64(0)

        wall_left_rects_xs = np.array(
            [c[0] for c in wall.coords if c[1] == head_y and c[0] < head_x]
        )
        if wall_left_rects_xs.size != 0:
            wall_danger_left = (
                1 - np.min(head_x - 1 - wall_left_rects_xs) / available_ncols
            )
        else:
            wall_danger_left = np.float64(0)

        body_left_rects_xs = np.array(
            [c[0] for c in snake.body_coords if c[1] == head_y and c[0] < head_x]
        )
        if body_left_rects_xs.size != 0:
            body_danger_left = (
                1 - np.min(head_x - 1 - body_left_rects_xs) / available_ncols
            )
        else:
            body_danger_left = np.float64(0)

        wall_upper_rects_ys = np.array(
            [c[1] for c in wall.coords if c[0] == head_x and c[1] < head_y]
        )
        if wall_upper_rects_ys.size != 0:
            wall_danger_up = (
                1 - np.min(head_y - 1 - wall_upper_rects_ys) / available_nrows
            )
        else:
            wall_danger_up = np.float64(0)

        body_upper_rects_ys = np.array(
            [c[1] for c in snake.body_coords if c[0] == head_x and c[1] < head_y]
        )
        if body_upper_rects_ys.size != 0:
            body_danger_up = (
                1 - np.min(head_y - 1 - body_upper_rects_ys) / available_nrows
            )
        else:
            body_danger_up = np.float64(0)

        wall_bottom_rects_ys = np.array(
            [c[1] for c in wall.coords if c[0] == head_x and c[1] > head_y]
        )
        if wall_bottom_rects_ys.size != 0:
            wall_danger_down = (
                1 - np.min(wall_bottom_rects_ys - head_y - 1) / available_nrows
            )
        else:
            wall_danger_down = np.float64(0)

        body_bottom_rects_ys = np.array(
            [c[1] for c in snake.body_coords if c[0] == head_x and c[1] > head_y]
        )
        if body_bottom_rects_ys.size != 0:
            body_danger_down = (
                1 - np.min(body_bottom_rects_ys - head_y - 1) / available_nrows
            )
        else:
            body_danger_down = np.float64(0)

        danger_right = np.max([wall_danger_right, body_danger_right])
        danger_left = np.max([wall_danger_left, body_danger_left])
        danger_up = np.max([wall_danger_up, body_danger_up])
        danger_down = np.max([wall_danger_down, body_danger_down])

        # Distances to apple
        apple_dx, apple_dy = (apple.coords - snake.head_coords) / np.array(
            [
                available_ncols,
                available_nrows,
            ]
        )

        apple_right = np.max((0, apple_dx))
        apple_left = np.max((0, -apple_dx))
        apple_up = np.max((0, -apple_dy))
        apple_down = np.max((0, apple_dy))

        features = np.array(
            [
                danger_right,
                danger_left,
                danger_down,
                danger_up,
                apple_right,
                apple_left,
                apple_down,
                apple_up,
            ]
        )
        logger.debug(f"features: {dict(zip(self.FEATURE_NAMES, features))}")

        return features


class Player:
    _IDX = 0

    def __init__(
        self, color: tuple[int, int, int], controller, name: str | None = None
    ) -> None:
        self.color = color
        self.controller = controller
        self.name = name or str(self._IDX)
        self.__class__._IDX += 1
        self.reset()

    def reset(self) -> None:
        """Reset the player to default state."""
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
    ) -> None:
        self.ncols = ncols
        self.nrows = nrows
        self.player = player
        self.wall = wall
        self.snake = snake
        self.apple = apple
        self.reset()

    @property
    def is_over(self) -> bool:
        """Whether the game is over or not."""
        return not self.snake.is_alive

    def reset(self) -> None:
        """Reset the game and corresponding assets - player, snake, apple to default state."""  # noqa E501
        self.player.reset()
        self.snake.reset()
        self.apple.reset()

    def _get_coords_for_random_apple(self) -> list[np.ndarray]:
        """Helper method used for setting free coordinates for apple placement."""
        xs = np.arange(int(self.ncols))
        ys = np.arange(int(self.nrows))
        exclude_coords = self.wall.coords + self.snake.coords

        coords = []
        for c in product(xs, ys):
            arr = np.array(c)
            if not np.all(exclude_coords == arr, axis=1).any():
                coords.append(arr)

        return coords

    def eval_state(self) -> None:
        """Evaluate a game state - check wall and self collisions of snake, check if snake ate an apple."""  # noqa E501
        # wall collision
        if np.all(self.wall.coords == self.snake.head_coords, axis=1).any():
            self.snake.is_alive = False
            logger.debug("Wall collision.")

        # self collision
        if np.all(self.snake.body_coords == self.snake.head_coords, axis=1).any():
            self.snake.is_alive = False
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

    @abstractmethod
    def step(self) -> None:
        """Do a game step - player controller sets direction, snake moves, evaluate the state."""  # noqa E501
        pass


class HumanGame(Game):
    def __init__(
        self,
        ncols: int,
        nrows: int,
        player: Player,
        wall: Wall,
        snake: Snake,
        apple: Apple,
    ) -> None:
        super().__init__(ncols, nrows, player, wall, snake, apple)

    def reset(self) -> None:
        """Reset the game and corresponding assets - player, snake, apple to default state."""  # noqa E501
        self.has_started = False
        super().reset()

    def step(self) -> None:
        """Do a game step - player controller sets direction, snake moves, evaluate the state."""  # noqa E501
        direction = self.player.controller.set_dir()
        self.snake.move(direction)
        self.eval_state()


class GAGame(Game):
    def __init__(
        self,
        ncols: int,
        nrows: int,
        player: Player,
        wall: Wall,
        snake: Snake,
        apple: Apple,
    ) -> None:
        super().__init__(ncols, nrows, player, wall, snake, apple)

    @property
    def steps(self) -> int:
        return len(self.coords_stepped)

    def reset(self) -> None:
        """Reset the game and corresponding assets - player, snake, apple to default state."""  # noqa E501
        self.coords_stepped = []
        # self.dirs_from_last_apple = [self.snake.head_dir]
        super().reset()

    def step(self) -> None:
        """Do a game step - player controller sets direction, snake moves, evaluate the state."""  # noqa E501
        features = self.player.controller.eval_state(self.snake, self.apple, self.wall)
        direction = self.player.controller.set_dir(features)
        self.snake.move(direction)
        self.eval_state()
        self.coords_stepped.append(self.snake.head_coords.copy())
        # self.dirs_from_last_apple.append(self.snake.head_dir)
