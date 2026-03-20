import logging
import random
from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from .const import LEFT, RIGHT

logger = logging.getLogger(__name__)


class AppleBase(ABC):
    def __init__(self, color: tuple[int, int, int]) -> None:
        self.color = color

    @abstractmethod
    def reset(self) -> None:
        """Reset the apple to default position."""
        self.idx = 0

    @abstractmethod
    def move(self) -> None:
        """Move apple to new position."""
        self.idx += 1


class RandomApple(AppleBase):
    INIT_COORDS = np.array([20, 10])

    def __init__(self, color: tuple[int, int, int]) -> None:
        super().__init__(color)
        self.reset()

    def reset(self) -> None:
        """Reset the apple to default position."""
        self.coords = self.INIT_COORDS.copy()
        super().reset()

    def move(self, coords_choice: list[np.ndarray]) -> None:
        """Move apple to new position."""
        self.coords = random.choice(coords_choice)
        super().move()


class DeterministicApple(AppleBase):
    _COORDS = [
        np.array([20, 10]),
        # 2 - 8: dist 4, clockwise
        np.array([20, 15]),
        np.array([15, 15]),
        np.array([10, 15]),
        np.array([10, 10]),
        np.array([10, 5]),
        np.array([15, 5]),
        np.array([20, 5]),
        # 9 - 12: dist 1, mid of wall, clockwise
        np.array([27, 10]),
        np.array([15, 17]),
        np.array([2, 10]),
        np.array([15, 2]),
        # 13 - 16: dist 1, every corner, clockwise
        np.array([27, 2]),
        np.array([27, 17]),
        np.array([2, 17]),
        np.array([2, 2]),
        # 17 - 20: dist 0, mid of wall, anti-clockwise
        np.array([15, 1]),
        np.array([1, 10]),
        np.array([15, 18]),
        np.array([28, 10]),
        # 21 - 24: dist 0, every corver, anti-clockwise
        np.array([28, 1]),
        np.array([1, 1]),
        np.array([1, 18]),
        np.array([28, 18]),
        # 25-27: mid
        np.array([15, 10]),
        np.array([14, 10]),
        np.array([16, 10]),
    ]

    def __init__(self, color: tuple[int, int, int]) -> None:
        super().__init__(color)
        self.reset()

    def reset(self) -> None:
        """Reset the apple to default position."""
        super().reset()

    def move(self) -> None:
        """Move apple to new position."""
        super().move()

    @property
    def coords(self) -> np.ndarray:
        """Coordinates of an apple."""
        return self._COORDS[self.idx]


class Snake:
    INIT_HEAD_COORDS = np.array([10, 10])
    INIT_COORDS = [
        INIT_HEAD_COORDS,
        INIT_HEAD_COORDS + LEFT,
        INIT_HEAD_COORDS + 2 * LEFT,
    ]

    def __init__(self, color: tuple[int, int, int]) -> None:
        self.color = color
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
    def body_coords(self) -> np.ndarray:
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

        for c, direction in zip(self.coords, self.dirs_q):
            c += direction

    def extend(self) -> None:
        """Extend the snake."""
        self.coords.append(self.tail_coords.copy() - self.tail_dir)
        self.dirs_q.append(self.tail_dir)


class Wall:
    def __init__(self, coords: list[np.ndarray]) -> None:
        self.color = tuple(50 * np.ones(3))
        self.coords = coords
