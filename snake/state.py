import random
from collections import deque

import numpy as np
import pygame

from .const import GRID_SIZE, LEFT, RIGHT


def get_rect(coords: np.ndarray):
    return pygame.Rect(*coords, GRID_SIZE, GRID_SIZE)


class Apple:
    def __init__(
        self,
        color: tuple[int],
        available_coords: list[np.ndarray],
        exclude_coords: list[np.ndarray],
    ):
        self.color = color
        self.available_coords = available_coords
        self.reset(exclude_coords)

    def reset(self, exclude_coords: list[np.ndarray]):
        self.move(exclude_coords)

    def move(self, exclude_coords: list[np.ndarray]):

        while True:
            new_coords = random.choice(self.available_coords)
            if not any(np.array_equal(new_coords, c) for c in exclude_coords):
                break

            print("Re-arranging apple.")

        self.coords = new_coords

    @property
    def rect(self):
        return get_rect(self.coords)


class Snake:
    INIT_HEAD_COORDS = GRID_SIZE * np.array([10, 10])
    INIT_COORDS = [
        INIT_HEAD_COORDS,
        INIT_HEAD_COORDS + GRID_SIZE * LEFT,
        INIT_HEAD_COORDS + 2 * GRID_SIZE * LEFT,
    ]

    def __init__(self, color: tuple[int]):
        self.color = color
        self.reset()

    def reset(self) -> None:
        self.coords = [c.copy() for c in self.INIT_COORDS]
        self.head_dir = RIGHT
        self.dirs_q = deque([self.head_dir, RIGHT, RIGHT])

    @property
    def rects(self):
        return [get_rect(c) for c in self.coords]

    # @property
    # def head_pos(self):
    #     return self.head_rect.topleft

    @property
    def head_rect(self):
        # Used for rendering
        return self.rects[0]

    @property
    def body_rects(self):
        # Used for rendering
        return self.rects[1:]

    @property
    def head_coords(self) -> np.ndarray:
        return self.coords[0]

    @property
    def body_coords(self) -> np.ndarray:
        return self.coords[1:]

    @property
    def tail_coords(self):
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
            c += direction * GRID_SIZE

    def extend(self):
        self.coords.append(self.tail_coords.copy() - self.tail_dir * GRID_SIZE)
        self.dirs_q.append(self.tail_dir)


class Wall:
    def __init__(self, coords: list[np.ndarray]):
        self.color = tuple(50 * np.ones(3))
        self.coords = coords

    @property
    def rects(self):
        return [get_rect(c) for c in self.coords]
