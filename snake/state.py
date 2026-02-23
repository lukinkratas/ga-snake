from collections import deque

import numpy as np
import pygame

from .const import (
    COLS,
    GAME_HEIGHT,
    GAME_ROWS,
    GRID_SIZE,
    LEFT,
    RIGHT,
    WIDTH,
)


def get_rect(x, y):
    return pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)


class Apple:
    def __init__(self, color: tuple[int]):
        self.color = color
        self.pos = GRID_SIZE * np.array([int(3 * COLS / 4), int(3 * GAME_ROWS / 4)])

    @property
    def rect(self):
        return get_rect(*self.pos)

    def move(self, exclude: list[pygame.Rect]):
        while True:
            # exclude border by default
            new_x = np.random.randint(1, COLS - 1, size=1)
            new_y = np.random.randint(1, GAME_ROWS - 1, size=1)
            self.pos = GRID_SIZE * np.concatenate((new_x, new_y))
            # TODO FIX
            if self.rect.collidelist(exclude) == -1:
                break
            print("re-regenarating apple")


class Snake:
    def __init__(self, color: tuple[int]):
        self.color = color
        self.head_pos = GRID_SIZE * np.array([int(COLS / 2), int(GAME_ROWS / 2)])
        self.rects = [
            get_rect(*self.head_pos),
            get_rect(*(self.head_pos + GRID_SIZE * LEFT)),
            get_rect(*(self.head_pos + 2 * GRID_SIZE * LEFT)),
        ]
        self.head_dir = RIGHT
        self.dirs_q = deque([self.head_dir, RIGHT, RIGHT])
        self.alive = True

    @property
    def head_rect(self):
        return self.rects[0]

    @property
    def body_rects(self):
        return self.rects[1:-1]

    @property
    def tail_rect(self):
        return self.rects[-1]

    @property
    def tail_dir(self):
        return self.dirs_q[-1]

    def move(self, direction: np.ndarray | None):

        if direction is not None:
            self.head_dir = direction

        self.dirs_q.appendleft(self.head_dir)
        self.dirs_q.pop()

        for rect, direction in zip(self.rects, self.dirs_q):
            rect.move_ip(*(direction * GRID_SIZE))

    def extend(self):
        self.dirs_q.append(self.tail_dir)
        self.rects.append(self.tail_rect.move(*(-self.tail_dir * GRID_SIZE)))


class Wall:
    def __init__(self):
        self.color = tuple(50 * np.ones(3))
        self.rects = self.get_rects()

    def get_rects(self):
        rects = []

        rects += [get_rect(x, y=0) for x in np.arange(0, WIDTH - 1, GRID_SIZE)]

        rects += [
            get_rect(x, y=GAME_HEIGHT - GRID_SIZE)
            for x in np.arange(0, WIDTH - 1, GRID_SIZE)
        ]

        rects += [
            get_rect(0, y)
            for y in np.arange(GRID_SIZE, GAME_HEIGHT - GRID_SIZE - 1, GRID_SIZE)
        ]

        rects += [
            get_rect(WIDTH - GRID_SIZE, y)
            for y in np.arange(GRID_SIZE, GAME_HEIGHT - GRID_SIZE - 1, GRID_SIZE)
        ]

        return rects
