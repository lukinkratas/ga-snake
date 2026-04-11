import numpy as np

RIGHT = np.array([1, 0])
LEFT = np.array([-1, 0])
UP = np.array([0, -1])
DOWN = np.array([0, 1])

DIRECTIONS = {"right": RIGHT, "left": LEFT, "down": DOWN, "up": UP}
TRAINING_SETS = [
    [
        # 0: mid of wall, clockwise
        np.array([15, 18]),
        np.array([1, 10]),
        np.array([15, 1]),
        np.array([28, 10]),
        np.array([15, 15]),
        np.array([10, 10]),
        np.array([15, 5]),
        np.array([20, 10]),
        np.array([15, 11]),
        np.array([14, 10]),
        np.array([15, 9]),
        np.array([16, 10]),
    ],
    [
        # 1: every corner, anti-clockwise
        np.array([28, 1]),
        np.array([1, 1]),
        np.array([1, 18]),
        np.array([28, 18]),
        np.array([20, 5]),
        np.array([10, 5]),
        np.array([10, 15]),
        np.array([20, 15]),
        np.array([16, 9]),
        np.array([14, 9]),
        np.array([14, 11]),
        np.array([16, 11]),
    ],
    [
        # 2: mid of wall, back to mid, anti-clockwise
        np.array([15, 1]),
        np.array([15, 10]),
        np.array([1, 10]),
        np.array([15, 10]),
        np.array([15, 18]),
        np.array([15, 10]),
        np.array([28, 10]),
        np.array([15, 10]),
    ],
    [
        # 3: every corner, back to mid, clockwise
        np.array([28, 18]),
        np.array([15, 10]),
        np.array([1, 18]),
        np.array([15, 10]),
        np.array([1, 1]),
        np.array([15, 10]),
        np.array([28, 1]),
        np.array([15, 10]),
    ],
    [
        # 4: eight
        np.array([28, 18]),
        np.array([15, 18]),
        np.array([15, 10]),
        np.array([15, 1]),
        np.array([1, 1]),
        np.array([1, 10]),
        np.array([15, 10]),
        np.array([15, 1]),
        np.array([28, 1]),
        np.array([28, 10]),
        np.array([15, 10]),
        np.array([1, 10]),
        np.array([1, 18]),
        np.array([15, 18]),
        np.array([15, 10]),
    ],
    [
        # 5: triangles
        np.array([28, 18]),
        np.array([1, 18]),
        np.array([1, 1]),
        np.array([28, 18]),
        np.array([28, 1]),
        np.array([1, 1]),
        np.array([28, 18]),
        np.array([1, 18]),
        np.array([28, 1]),
        np.array([28, 18]),
        np.array([1, 18]),
        np.array([28, 1]),
        np.array([1, 1]),
        np.array([1, 18]),
    ],
    [
        # 6: zig zags
        np.array([28, 18]),
        np.array([1, 10]),
        np.array([28, 10]),
        np.array([1, 1]),
        np.array([28, 1]),
        np.array([15, 18]),
        np.array([15, 1]),
        np.array([1, 18]),
        np.array([1, 1]),
    ],
    [
        # 7: side to side
        np.array([1, 18]),
        np.array([1, 1]),
        np.array([2, 18]),
        np.array([2, 1]),
        np.array([3, 18]),
        np.array([3, 1]),
        np.array([28, 1]),
        np.array([1, 2]),
        np.array([28, 2]),
        np.array([1, 3]),
        np.array([28, 3]),
        np.array([1, 4]),
    ],
]
