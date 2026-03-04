import numpy as np

from .state import Wall


def get_random_color() -> tuple[int, int, int]:
    return tuple(np.random.randint(50, 200, size=3))


def get_squared_wall(ncols: int, nrows: int):

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
