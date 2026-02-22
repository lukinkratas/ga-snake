from collections import deque

import numpy as np
import pygame

WIDTH = 600
GAME_HEIGHT = 400
SCORE_HEIGHT = 200
SIZE = 20

COLS = int(WIDTH / SIZE)
GAME_ROWS = int(GAME_HEIGHT / SIZE)


def get_random_color() -> tuple[int]:
    return tuple(np.random.randint(50, 200, size=3))


def get_rect(x, y):
    return pygame.Rect(x, y, SIZE, SIZE)


def draw_circle(surf, rect, color, line_color, line_width: int = 2, radius: int = 8):
    # draw filled circle
    pygame.draw.circle(surface=surf, color=color, center=rect.center, radius=radius)
    # draw line around circle
    pygame.draw.circle(
        surface=surf,
        color=line_color,
        center=rect.center,
        radius=radius,
        width=line_width,
    )
    return rect


def draw_rect(surf, rect, color, line_color, line_width: int = 2, radius: int = 5):
    # draw filled rect
    pygame.draw.rect(surface=surf, color=color, rect=rect, border_radius=radius)
    # draw line around rect
    pygame.draw.rect(
        surface=surf,
        color=line_color,
        rect=rect,
        width=line_width,
        border_radius=radius,
    )
    return rect


class Snake:
    def __init__(self, surf, color: tuple[int]):
        self.surf = surf
        self.color = color
        self.pos = np.array([100, 100])
        self.rects = [
            get_rect(*self.pos),
            get_rect(*(self.pos - SIZE * np.array([1, 0]))),
            get_rect(*(self.pos - SIZE * np.array([2, 0]))),
        ]
        self.head_dir = np.array([1, 0])
        self.dirs_q = deque([self.head_dir, np.array([1, 0]), np.array([1, 0])])
        self.alive = True

    @property
    def head(self):
        return self.rects[0]

    @property
    def body(self):
        return self.rects[1:]

    @property
    def tail(self):
        return self.rects[-1]

    @property
    def tail_dir(self):
        return self.dirs_q[-1]

    def draw(self):

        for rect in self.rects:
            draw_rect(self.surf, rect, self.color, line_color=(50, 50, 50))

    def move(self):
        keymap = {
            pygame.K_LEFT: np.array([-1, 0]),
            pygame.K_h: np.array([-1, 0]),
            pygame.K_RIGHT: np.array([1, 0]),
            pygame.K_l: np.array([1, 0]),
            pygame.K_UP: np.array([0, -1]),
            pygame.K_k: np.array([0, -1]),
            pygame.K_DOWN: np.array([0, 1]),
            pygame.K_j: np.array([0, 1]),
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        keys = pygame.key.get_pressed()

        for key, direction in keymap.items():
            if keys[key]:
                self.head_dir = direction
                break

        self.dirs_q.appendleft(self.head_dir)
        self.dirs_q.pop()

        for rect, direction in zip(self.rects, self.dirs_q):
            rect.move_ip(*(direction * SIZE))

    def extend(self):
        self.dirs_q.append(self.tail_dir)
        self.rects.append(self.tail.move(*(-self.tail_dir * SIZE)))


class Apple:
    def __init__(self, surf, color: tuple[int]):
        self.surf = surf
        self.pos = np.array([200, 200])
        self.color = color

    @property
    def rect(self):
        return get_rect(*self.pos)

    def draw(self):
        draw_circle(self.surf, self.rect, self.color, line_color=(25, 25, 25))

    def move(self, exclude: list[pygame.Rect]):
        # exclude border by default
        new_x = np.random.randint(1, COLS - 1, size=1)
        new_y = np.random.randint(1, GAME_ROWS - 1, size=1)
        self.pos = SIZE * np.concatenate((new_x, new_y))

        # TODO FIX
        while self.rect.collidelist(exclude) != -1:
            print("re-regenarating apple")
            self.move(exclude)


class Player:
    def __init__(self, snake: Snake, color: tuple[int]):
        self.score = 0
        self.snake = snake
        self.color = color


class Scoreboard:
    def __init__(self, surf, font_color: tuple[int] = (175, 175, 175)):
        self.surf = surf
        self.font_size = 14
        self.font_color = font_color
        self.font = pygame.font.SysFont("Arial", self.font_size)
        self.font_bold = pygame.font.SysFont(
            "Arial", self.font_size, pygame.font.Font.bold
        )
        self.padding = 6
        self.row_size = self.font_size + self.padding

    def draw_row(self, text, font, font_color, rect):
        text_surf = font.render(text, False, font_color)
        text_rect = text_surf.get_rect(center=rect.center)
        self.surf.blit(text_surf, text_rect)

    def draw(self, players: list[Player]):
        controls_rect = pygame.Rect(0, 0, self.surf.get_width(), self.row_size)
        self.draw_row(
            text="Use arrows or vim-like H J K L",
            font=self.font,
            font_color=self.font_color,
            rect=controls_rect,
        )

        title_rect = pygame.Rect(0, self.row_size, self.surf.get_width(), self.row_size)
        self.draw_row(
            text="Scoreboard",
            font=self.font_bold,
            font_color=self.font_color,
            rect=title_rect,
        )

        y_offset = 2 * self.row_size
        for idx, player in enumerate(players):
            rect = pygame.Rect(
                0, y_offset + idx * self.row_size, self.surf.get_width(), self.row_size
            )
            text = f"Player no. {idx + 1}, score: {player.score}"
            if player.snake.alive is False:
                text += " - DEAD"
            self.draw_row(
                text,
                font=self.font_bold if player.snake.alive else self.font,
                font_color=player.color,
                rect=rect,
            )


class Border:
    def __init__(self, surf):
        self.surf = surf
        self.rects = self.get_rects()
        self.color = (50, 50, 50)

    def get_rects(self):
        rects = []

        rects += [get_rect(x, y=0) for x in np.arange(0, WIDTH - 1, SIZE)]

        rects += [
            get_rect(x, y=GAME_HEIGHT - SIZE) for x in np.arange(0, WIDTH - 1, SIZE)
        ]

        rects += [get_rect(0, y) for y in np.arange(SIZE, GAME_HEIGHT - SIZE - 1, SIZE)]

        rects += [
            get_rect(WIDTH - SIZE, y)
            for y in np.arange(SIZE, GAME_HEIGHT - SIZE - 1, SIZE)
        ]

        return rects

    def draw(self):

        for rect in self.rects:
            draw_rect(self.surf, rect, color=self.color, line_color=(25, 25, 25))


def draw_grid(surf, line_color: tuple[int] = (150, 150, 150)):
    for x in np.arange(SIZE, WIDTH - 1, SIZE):
        pygame.draw.line(
            surface=surf,
            color=line_color,
            start_pos=(x, 0),
            end_pos=(x, GAME_HEIGHT),
            width=1,
        )

    for y in np.arange(SIZE, GAME_HEIGHT - 1, SIZE):
        pygame.draw.line(
            surface=surf,
            color=line_color,
            start_pos=(0, y),
            end_pos=(WIDTH, y),
            width=1,
        )


def eval_frame(border: Border, apple: Apple, players: list[Player]) -> bool:

    for player in players:
        if player.snake.alive is False:
            continue

        # border collision
        if player.snake.head.collidelist(border.rects) != -1:
            player.snake.alive = False

        # self collision
        if player.snake.head.collidelist(player.snake.body) != -1:
            player.snake.alive = False

        if player.snake.head.colliderect(apple.rect):
            player.score += 1
            player.snake.extend()
            apple.move(exclude=player.snake.rects)


def render_frame(
    game_surf,
    score_surf,
    border: Border,
    apple: Apple,
    scoreboard: Scoreboard,
    players: list[Player],
):
    game_surf.fill(color=(175, 175, 175))
    score_surf.fill(color=(25, 25, 25))
    draw_grid(game_surf)
    border.draw()
    for player in players:
        player.snake.draw()
    apple.draw()
    scoreboard.draw(players)
    pygame.display.update()


def main():
    pygame.init()

    win = pygame.display.set_mode(size=(WIDTH, GAME_HEIGHT + SCORE_HEIGHT))

    game_rect = pygame.Rect(0, 0, WIDTH, GAME_HEIGHT)
    game_surf = win.subsurface(game_rect)

    score_rect = pygame.Rect(0, GAME_HEIGHT, WIDTH, SCORE_HEIGHT)
    score_surf = win.subsurface(score_rect)

    border = Border(surf=game_surf)
    color = get_random_color()
    apple = Apple(surf=game_surf, color=color)
    players = [Player(snake=Snake(surf=game_surf, color=color), color=color)]
    scoreboard = Scoreboard(surf=score_surf)

    pygame.display.set_caption("Snake")
    clock = pygame.time.Clock()

    run = True
    while run:
        # game loop
        pygame.time.delay(50)
        clock.tick(10)

        eval_frame(border, apple, players)

        render_frame(game_surf, score_surf, border, apple, scoreboard, players)

        for player in players:
            if player.snake.alive:
                player.snake.move()

        if not any(player.snake.alive for player in players):
            run = False

    while True:
        event = pygame.event.wait()

        if event.type == pygame.KEYDOWN:
            break

    pygame.quit()


if __name__ == "__main__":
    main()
