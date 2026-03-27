import logging
from collections import deque

import numpy as np

from .const import LEFT, RIGHT

logger = logging.getLogger(__name__)
rng = np.random.default_rng(seed=42)


class Apple:
    INIT_COORDS = np.array([20, 10])

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset the apple to default position."""
        self.idx = 0
        self.coords = self.INIT_COORDS.copy()
        self.coords_list = [self.coords]

    def move(self, coords: np.ndarray) -> None:
        """Move apple to new position."""
        self.coords = coords
        self.coords_list.append(self.coords)
        self.idx += 1

    # @property
    # def vecs(self) -> np.ndarray:
    #     return np.diff(
    #         np.concatenate(([Snake.INIT_HEAD_COORDS], self.coords_list)), axis=0
    #     )
    #
    # @property
    # def min_nsteps_needed(self) -> np.ndarray:
    #     return np.sum(np.abs(self.vecs), axis=1)


class Snake:
    INIT_HEAD_COORDS = np.array([10, 10])
    INIT_COORDS = [
        INIT_HEAD_COORDS,
        INIT_HEAD_COORDS + LEFT,
        INIT_HEAD_COORDS + 2 * LEFT,
    ]

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset snake to default position and direction."""
        self.coords = [c.copy() for c in self.INIT_COORDS]
        self.head_dir = RIGHT
        self.dirs_q = deque([self.head_dir, RIGHT, RIGHT])
        self.is_alive = True

    @property
    def head_coords(self) -> np.ndarray:
        """Coordinates of snake's head. Used for collision detection and rendering."""
        return self.coords[0]

    @property
    def body_coords(self) -> list[np.ndarray]:
        """Coordinates of snake's body. Used for collision detection and rendering."""
        return self.coords[1:]

    @property
    def tail_coords(self) -> np.ndarray:
        """Coordinates of snake's tail. Used for extending."""
        return self.coords[-1]

    @property
    def tail_dir(self) -> np.ndarray:
        """Direction of snake's tail. Used for extending."""
        return self.dirs_q[-1]

    def move(self, direction: np.ndarray | None) -> None:
        """Move the snake in direction.

        Args:
            direction: direction (array)
        """
        if direction is not None:
            self.head_dir = direction

        self.dirs_q.appendleft(self.head_dir)
        self.dirs_q.pop()

        for c, d in zip(self.coords, self.dirs_q):
            c += d

    def extend(self) -> None:
        """Extend the snake."""
        self.coords.append(self.tail_coords.copy() - self.tail_dir)
        self.dirs_q.append(self.tail_dir)


class Wall:
    def __init__(self, coords: list[np.ndarray]) -> None:
        self.color = tuple(50 * np.ones(3, dtype=int))
        self.coords = coords
