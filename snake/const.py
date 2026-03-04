import numpy as np
import pygame

RIGHT = np.array([1, 0])
LEFT = np.array([-1, 0])
UP = np.array([0, -1])
DOWN = np.array([0, 1])

CONTROLS_LEFT = [pygame.K_LEFT, pygame.K_h, pygame.K_a]
CONTROLS_RIGHT = [pygame.K_RIGHT, pygame.K_l, pygame.K_d]
CONTROLS_UP = [pygame.K_UP, pygame.K_k, pygame.K_w]
CONTROLS_DOWN = [pygame.K_DOWN, pygame.K_j, pygame.K_s]
