from itertools import product

import numpy as np

from .engine import Game
from .state import Wall

rng = np.random.default_rng(seed=42)


def get_random_color() -> np.ndarray:
    """Get random color.

    Returns: RGB tuple
    """
    return rng.integers(50, 200, size=3)


def get_squared_wall(ncols: int, nrows: int) -> Wall:
    """Get wall instance of square shape.

    ncols: number of columns
    nrows: number of rows

    Returns: Wall instance
    """
    x1 = np.arange(0, ncols)
    y1 = np.zeros(x1.shape)

    x2 = np.arange(0, ncols)
    y2 = (nrows - 1) * np.ones(x2.shape)

    y3 = np.arange(1, nrows - 1)
    x3 = np.zeros(y3.shape)

    y4 = np.arange(1, nrows - 1)
    x4 = (ncols - 1) * np.ones(y4.shape)

    xs = np.concatenate((x1, x2, x3, x4), axis=0)
    ys = np.concatenate((y1, y2, y3, y4), axis=0)

    coords = [np.array(coords) for coords in zip(xs, ys)]
    return Wall(coords)


def get_exclude_coords(games: list[Game]) -> list[np.ndarray]:
    exclude = []

    uniq_walls = {game.wall for game in games}
    for wall in uniq_walls:
        exclude.extend(wall.coords)

    for game in games:
        exclude.extend(game.snake.coords)

    return exclude


def get_free_coords(
    xrange: np.ndarray, yrange: np.ndarray, exclude: list[np.ndarray]
) -> list[np.ndarray]:
    """Helper method used for setting free coordinates for apple placement."""
    all_coords = np.array(list(product(xrange, yrange)))  # shape: (N, 2)
    exclude_arr = np.array(exclude)  # shape: (M, 2)

    # For each point in all_coords, check if it matches ANY point in exclude_arr.
    # all_coords[:, None] broadcasts to (N, M, 2), enabling pairwise comparison.
    is_excluded = np.any(
        np.all(all_coords[:, None] == exclude_arr[None, :], axis=2), axis=1
    )
    return rng.choice(all_coords[~is_excluded])
