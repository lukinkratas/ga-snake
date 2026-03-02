import numpy as np

from .state import Wall


def get_random_color() -> tuple[int, int, int]:
    return tuple(np.random.randint(50, 200, size=3))


def get_squared_wall(width: int, height: int, step: int):

    x1 = np.arange(0, width - 1, step)
    y1 = np.zeros(x1.shape)

    x2 = np.arange(0, width - 1, step)
    y2 = (height - step) * np.ones(x2.shape)

    y3 = np.arange(step, height - 1, step)
    x3 = np.zeros(y3.shape)

    y4 = np.arange(step, height - 1, step)
    x4 = (width - step) * np.ones(y4.shape)

    xs = np.concatenate((x1, x2, x3, x4), axis=0)
    ys = np.concatenate((y1, y2, y3, y4), axis=0)

    xs = np.expand_dims(xs, axis=1)
    ys = np.expand_dims(ys, axis=1)

    coords = np.concatenate((xs, ys), axis=1).tolist()
    return Wall(coords)
