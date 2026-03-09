import numpy as np

from .state import Wall

rng = np.random.default_rng(seed=42)


def get_random_color() -> tuple[int, int, int]:
    """Get random color.

    Returns: RGB tuple
    """
    return tuple(rng.integers(50, 200, size=3))


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
