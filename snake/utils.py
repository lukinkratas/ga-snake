import numpy as np


def get_random_color() -> tuple[int]:
    return tuple(np.random.randint(50, 200, size=3))
