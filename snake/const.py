import numpy as np

WIDTH = 600
GAME_HEIGHT = 400
SCORE_HEIGHT = 200
GRID_SIZE = 20

COLS = int(WIDTH / GRID_SIZE)
GAME_ROWS = int(GAME_HEIGHT / GRID_SIZE)

RIGHT = np.array([1, 0])
LEFT = np.array([-1, 0])
UP = np.array([0, -1])
DOWN = np.array([0, 1])
