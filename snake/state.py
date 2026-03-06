import logging
import random
from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from .const import LEFT, RIGHT

logger = logging.getLogger(__name__)


class Apple(ABC):
    def __init__(self, color: tuple[int]):
        self.color = color

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def move(self):
        pass


class RandomApple(Apple):
    INIT_COORDS = np.array([20, 10])

    def __init__(self, color: tuple[int]):
        super().__init__(color)
        self.reset()

    def reset(self):
        self.coords = self.INIT_COORDS.copy()

    def move(self, coords_choice: list[np.ndarray]):
        self.coords = random.choice(coords_choice)


class DeterministicApple(Apple):
    _COORDS = [
        np.array([20, 10]),
        np.array([20, 15]),
        np.array([10, 15]),
        np.array([10, 5]),
        np.array([20, 5]),
        np.array([27, 17]),
        np.array([2, 17]),
        np.array([2, 2]),
        np.array([27, 2]),
        np.array([28, 18]),
        np.array([1, 18]),
        np.array([1, 1]),
        np.array([28, 1]),
    ]

    def __init__(self, color: tuple[int]):
        super().__init__(color)
        self.reset()

    def reset(self):
        self.idx = 0

    def move(self):
        self.idx += 1

    @property
    def coords(self):
        return self._COORDS[self.idx]


class Snake:
    INIT_HEAD_COORDS = np.array([10, 10])
    INIT_COORDS = [
        INIT_HEAD_COORDS,
        INIT_HEAD_COORDS + LEFT,
        INIT_HEAD_COORDS + 2 * LEFT,
    ]

    def __init__(self, color: tuple[int]):
        self.color = color
        self.reset()

    def reset(self) -> None:
        self.coords = [c.copy() for c in self.INIT_COORDS]
        self.head_dir = RIGHT
        self.dirs_q = deque([self.head_dir, RIGHT, RIGHT])

    @property
    def head_coords(self) -> np.ndarray:
        return self.coords[0]

    @property
    def body_coords(self) -> np.ndarray:
        return self.coords[1:]

    @property
    def tail_coords(self):
        # Used for extending
        return self.coords[-1]

    @property
    def tail_dir(self):
        # Used for extending
        return self.dirs_q[-1]

    def move(self, direction: np.ndarray | None):

        if direction is not None:
            self.head_dir = direction

        self.dirs_q.appendleft(self.head_dir)
        self.dirs_q.pop()

        for c, direction in zip(self.coords, self.dirs_q):
            c += direction

    def extend(self):
        self.coords.append(self.tail_coords.copy() - self.tail_dir)
        self.dirs_q.append(self.tail_dir)


class Wall:
    def __init__(self, coords: list[np.ndarray]):
        self.color = tuple(50 * np.ones(3))
        self.coords = coords
