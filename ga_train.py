import numpy as np
import pygame

from snake.const import DOWN, GRID_SIZE, LEFT, RIGHT, UP
from snake.engine import Game, Player
from snake.renderer import Renderer
from snake.state import Apple, Snake, Wall
from snake.utils import get_random_color, get_squared_wall

COLS = 30
GAME_ROWS = 20
WIDTH = COLS * GRID_SIZE
GAME_HEIGHT = GAME_ROWS * GRID_SIZE

FPS = 240
POP_SIZE = 22
NGENS = 100
# limit number of steps per game
NSTEPS = 100

SCORE_HEIGHT = 350


class Genome:
    def __init__(self, arr: np.ndarray, name: str):
        self.arr = arr
        self.name = name


class GAController:
    def __init__(self, genome: Genome):
        self.genome = genome

    def set_dir(self, features: np.ndarray) -> np.ndarray:
        scores = features @ self.genome.arr  # linear combination
        move_idx = np.argmax(scores)

        directions = [UP, DOWN, LEFT, RIGHT]
        return directions[move_idx]

    def eval_state(self, snake: Snake, apple: Apple, wall: Wall) -> np.ndarray:
        AVAILABLE_WIDTH = WIDTH - 3 * GRID_SIZE
        AVAILABLE_HEIGHT = GAME_HEIGHT - 3 * GRID_SIZE

        head_x = snake.head_rect.x
        head_y = snake.head_rect.y

        # Distances to walls
        wall_right_rects_dists = (
            np.array(
                [rect.x for rect in wall.rects if rect.y == head_y and rect.x > head_x]
            )
            - head_x
            - GRID_SIZE
        )
        # No bodies rects found on the right
        wall_safety_right = (
            np.min(wall_right_rects_dists) / AVAILABLE_WIDTH
            if wall_right_rects_dists.size != 0
            else np.float64(1)
        )

        # Distances to snake body
        body_right_rects_dists = (
            np.array(
                [
                    rect.x
                    for rect in snake.rects[1:]
                    if rect.y == head_y and rect.x > head_x
                ]
            )
            - head_x
            - GRID_SIZE
        )
        body_safety_right = (
            np.min(body_right_rects_dists) / AVAILABLE_WIDTH
            if body_right_rects_dists.size != 0
            else np.float64(1)
        )

        wall_left_rects_dists = (
            head_x
            - GRID_SIZE
            - np.array(
                [rect.x for rect in wall.rects if rect.y == head_y and rect.x < head_x]
            )
        )

        wall_safety_left = (
            np.min(wall_left_rects_dists) / AVAILABLE_WIDTH
            if wall_left_rects_dists.size != 0
            else np.float64(1)
        )

        body_left_rects_dists = (
            head_x
            - GRID_SIZE
            - np.array(
                [
                    rect.x
                    for rect in snake.rects[1:]
                    if rect.y == head_y and rect.x < head_x
                ]
            )
        )

        body_safety_left = (
            np.min(body_left_rects_dists) / AVAILABLE_WIDTH
            if body_left_rects_dists.size != 0
            else np.float64(1)
        )

        wall_up_rects_dists = (
            head_y
            - GRID_SIZE
            - np.array(
                [rect.y for rect in wall.rects if rect.x == head_x and rect.y < head_y]
            )
        )

        wall_safety_up = (
            np.min(wall_up_rects_dists) / AVAILABLE_HEIGHT
            if wall_up_rects_dists.size != 0
            else np.float64(1)
        )

        body_up_rects_dists = (
            head_y
            - GRID_SIZE
            - np.array(
                [
                    rect.y
                    for rect in snake.rects[1:]
                    if rect.x == head_x and rect.y < head_y
                ]
            )
        )

        body_safety_up = (
            np.min(body_up_rects_dists) / AVAILABLE_HEIGHT
            if body_up_rects_dists.size != 0
            else np.float64(1)
        )

        wall_down_rects_dists = (
            np.array(
                [rect.y for rect in wall.rects if rect.x == head_x and rect.y > head_y]
            )
            - head_y
            - GRID_SIZE
        )

        wall_safety_down = (
            np.min(wall_down_rects_dists) / AVAILABLE_HEIGHT
            if wall_down_rects_dists.size != 0
            else np.float64(1)
        )

        body_down_rects_dists = (
            np.array(
                [
                    rect.y
                    for rect in snake.rects[1:]
                    if rect.x == head_x and rect.y > head_y
                ]
            )
            - head_y
            - GRID_SIZE
        )

        body_safety_down = (
            np.min(body_down_rects_dists) / AVAILABLE_HEIGHT
            if body_down_rects_dists.size != 0
            else np.float64(1)
        )

        safety_right = np.min([wall_safety_right, body_safety_right])
        safety_left = np.min([wall_safety_left, body_safety_left])
        safety_up = np.min([wall_safety_up, body_safety_up])
        safety_down = np.min([wall_safety_down, body_safety_down])

        # Distances to apple
        apple_dx = (apple.rect.x - snake.head_rect.x) / AVAILABLE_WIDTH
        apple_dy = (apple.rect.y - snake.head_rect.y) / AVAILABLE_HEIGHT

        apple_right = np.max((0, apple_dx))
        apple_left = np.max((0, -apple_dx))
        apple_up = np.max((0, -apple_dy))
        apple_down = np.max((0, apple_dy))

        features = np.array(
            [
                safety_left,
                safety_right,
                safety_up,
                safety_down,
                apple_right,
                apple_left,
                apple_up,
                apple_down,
            ]
        )

        print("features:")
        print(f"  {wall_safety_right=}")
        print(f"  {wall_safety_left=}")
        print(f"  {wall_safety_down=}")
        print(f"  {wall_safety_up=}")
        print(f"  {body_safety_right=}")
        print(f"  {body_safety_left=}")
        print(f"  {body_safety_down=}")
        print(f"  {body_safety_up=}")
        print(f"  {safety_right=}")
        print(f"  {safety_left=}")
        print(f"  {safety_down=}")
        print(f"  {safety_up=}")
        print(f"  {apple_right=}")
        print(f"  {apple_left=}")
        print(f"  {apple_up=}")
        print(f"  {apple_down=}")
        return features


def mutate(arr: np.ndarray) -> np.ndarray:
    MUTATION_RATE = 0.2
    MUTATION_SCALE = 0.25
    mask = np.random.uniform(0, 1, arr.shape) < MUTATION_RATE
    noise = np.random.uniform(-1, 1, arr.shape) * MUTATION_SCALE
    new_genome = arr.copy()
    new_genome[mask] += noise[mask]
    return new_genome


def crossover(arr_a, arr_b) -> np.ndarray:
    mask = np.random.uniform(0, 1, arr_a.shape) < 0.5
    return np.where(mask, arr_a, arr_b)


def init_genomes(pop_size: int):
    genomes = []

    for idx, arr in enumerate(np.random.uniform(-1, 1, (pop_size, 8, 4))):
        genomes.append(Genome(arr, f"G{idx}"))

    return genomes


def init_games(width: int, height: int, genomes: list[Genome]) -> list[Game]:
    # common wall for all games
    wall = get_squared_wall(width, height, GRID_SIZE)
    games = []
    for genome in genomes:
        color = get_random_color()
        controller = GAController(genome)
        player = Player(color, controller, name=genome.name)
        game = Game(width, height, player, wall, color)
        games.append(game)

    return games


def reset(games: list[Game], genomes: list[Genome]) -> None:
    for genome, game in zip(genomes, games):
        game.reset()
        game.player.controller = GAController(genome)
        game.player.name = genome.name


def eval_fitness(games: list[Game]) -> np.ndarray:
    scores = np.array([game.score for game in games])
    steps = np.array([game.steps for game in games])
    deaths = np.array([game.active is False for game in games])
    return 100 * scores + 0.1 * steps - 50 * deaths


def main():
    pygame.init()

    win = pygame.display.set_mode(size=(WIDTH, GAME_HEIGHT + SCORE_HEIGHT))
    pygame.display.set_caption("Snake")

    clock = pygame.time.Clock()

    game_rect = pygame.Rect(0, 0, WIDTH, GAME_HEIGHT)
    game_surf = win.subsurface(game_rect)

    score_rect = pygame.Rect(0, GAME_HEIGHT, WIDTH, SCORE_HEIGHT)
    score_surf = win.subsurface(score_rect)

    genomes = init_genomes(POP_SIZE)
    games = init_games(WIDTH, GAME_HEIGHT, genomes)
    renderer = Renderer(
        game_surf,
        score_surf,
        rect_radius=int(GRID_SIZE / 4),
        line_width=1,
        font_size=10,
    )
    renderer.render_frame(games)

    pygame.time.delay(500)

    for gen in range(NGENS):
        print(f"New gen {gen}")
        # game loop
        for _ in range(NSTEPS):
            clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            for game in games:
                if game.active is False:
                    continue

                features = game.player.controller.eval_state(
                    game.snake, game.apple, game.wall
                )
                direction = game.player.controller.set_dir(features)
                game.move(direction)
                game.eval_state()

            renderer.render_frame(games, gen=gen + 1)

            if all(game.active is False for game in games):
                break

        print([g.name for g in genomes])
        fitness = eval_fitness(games)
        # sort by fitness
        elite_idxs = np.argsort(fitness)[::-1]
        # select top 5 as elite
        nelites = int(POP_SIZE / 3)
        elite_genomes = [genomes[i] for i in elite_idxs[:nelites]]
        print([g.name for g in elite_genomes])

        # keep elites unchanged
        next_gen_genomes = elite_genomes.copy()

        for _ in range(int(POP_SIZE / 3)):
            parent_a, parent_b = np.random.choice(elite_genomes, 2)
            child_arr = crossover(parent_a.arr, parent_b.arr)
            child_arr = mutate(child_arr)
            child_name = f"{parent_a.name} C{parent_b.name}M"
            next_gen_genomes.append(Genome(child_arr, child_name))

        for i in range(POP_SIZE - len(next_gen_genomes)):
            parent = elite_genomes[i % len(elite_genomes)]
            child_arr = mutate(parent.arr)
            child_name = f"{parent.name}M"
            next_gen_genomes.append(Genome(child_arr, child_name))

        print([g.name for g in next_gen_genomes])
        pygame.time.delay(500)
        reset(games, next_gen_genomes)
        genomes = next_gen_genomes.copy()

    while True:
        event = pygame.event.wait()

        if event.type in [pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN]:
            break

    pygame.quit()


if __name__ == "__main__":
    main()
