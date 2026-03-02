from typing import Any

import numpy as np
import pygame

from .const import (
    DOWN,
    GRID_SIZE,
    LEFT,
    RIGHT,
    UP,
)
from .state import Apple, Snake, Wall

ALPHA_MAP = {True: 255, False: 127}


def draw_circle(
    surf,
    rect,
    color,
    line_color,
    line_width: int,
    radius: int,
    *args: Any,
    **kwargs: Any,
):
    if len(color) == 3:
        color += (255,)

    if len(line_color) == 3:
        line_color += (255,)

    rect_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
    # draw filled circle
    pygame.draw.circle(
        surface=rect_surf,
        color=color,
        center=rect_surf.get_rect().center,
        radius=radius,
        *args,
        **kwargs,
    )
    # draw line around circle
    pygame.draw.circle(
        surface=rect_surf,
        color=line_color,
        center=rect_surf.get_rect().center,
        radius=radius,
        width=line_width,
        *args,
        **kwargs,
    )
    surf.blit(rect_surf, rect)
    return rect


def draw_rect(
    surf,
    rect,
    color,
    line_color,
    line_width: int,
    radius: int,
    *args: Any,
    **kwargs: Any,
):
    if len(color) == 3:
        color += (255,)

    if len(line_color) == 3:
        line_color += (255,)

    rect_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
    # draw filled rect
    pygame.draw.rect(
        surface=rect_surf,
        color=color,
        rect=rect_surf.get_rect(),
        border_radius=radius,
        *args,
        **kwargs,
    )
    # draw line around rect
    pygame.draw.rect(
        surface=rect_surf,
        color=line_color,
        rect=rect_surf.get_rect(),
        width=line_width,
        border_radius=radius,
        *args,
        **kwargs,
    )
    surf.blit(rect_surf, rect)
    return rect


def draw_row(surf, text, font, font_color, rect):
    text_surf = font.render(text, False, font_color)
    text_rect = text_surf.get_rect(center=rect.center)
    surf.blit(text_surf, text_rect)


class Renderer:
    def __init__(
        self,
        game_surf: pygame.Surface,
        score_surf: pygame.Surface,
        rect_radius: int,
        line_width: int,
        font_size: int,
        font_family: str = "Arial",
        scoreboard_row_size: int | None = None,
    ):
        self.game_surf = game_surf
        self.score_surf = score_surf
        self.rect_radius = rect_radius
        self.line_width = line_width
        self.font_size = font_size
        self.font = pygame.font.SysFont(font_family, font_size)
        self.font_bold = pygame.font.SysFont(
            font_family, font_size, pygame.font.Font.bold
        )
        self.scoreboard_row_size = scoreboard_row_size or int(1.4 * self.font_size)

    def draw_grid(self):
        LINE_COLOR = (150,) * 3

        x_range = np.arange(GRID_SIZE, self.game_surf.get_width() - 1, GRID_SIZE)
        for x in x_range:
            pygame.draw.line(
                surface=self.game_surf,
                color=LINE_COLOR,
                start_pos=(x, 0),
                end_pos=(x, self.game_surf.get_height()),
                width=1,
            )

        y_range = np.arange(GRID_SIZE, self.game_surf.get_height() - 1, GRID_SIZE)
        for y in y_range:
            pygame.draw.line(
                surface=self.game_surf,
                color=LINE_COLOR,
                start_pos=(0, y),
                end_pos=(self.game_surf.get_width(), y),
                width=1,
            )

    def draw_coords(self):
        COLOR = (150,) * 3
        FONT = pygame.font.SysFont("Arial", 8)

        x_range = np.arange(GRID_SIZE, self.game_surf.get_width() - 1, GRID_SIZE)
        y_range = np.arange(GRID_SIZE, self.game_surf.get_height() - 1, GRID_SIZE)

        for x in x_range:
            for y in y_range:
                text = f"x{int(x)}"
                rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE / 2)
                draw_row(self.game_surf, text, FONT, COLOR, rect)

                text = f"y{int(y)}"
                rect = pygame.Rect(x, y + GRID_SIZE / 2, GRID_SIZE, GRID_SIZE / 2)
                draw_row(self.game_surf, text, FONT, COLOR, rect)

    def draw_wall(self, wall: Wall):

        for rect in wall.rects:
            draw_rect(
                self.game_surf,
                rect,
                wall.color,
                line_color=tuple(25 * np.ones(3)),
                line_width=self.line_width,
                radius=self.rect_radius,
            )

    def draw_apple(self, apple: Apple, active: bool):
        alpha = ALPHA_MAP[active]
        draw_circle(
            self.game_surf,
            apple.rect,
            color=(*apple.color, alpha),
            line_color=(25, 25, 25, alpha),
            line_width=self.line_width,
            radius=int(0.8 * GRID_SIZE / 2),
        )

    def draw_snake(self, snake: Snake, active: bool):

        EXTRA_RADIUS = 10
        alpha = ALPHA_MAP[active]
        common_kwargs = {
            "surf": self.game_surf,
            "color": (*snake.color, alpha),
            "line_color": (50, 50, 50, alpha),
            "line_width": self.line_width,
            "radius": self.rect_radius,
        }

        # draw head
        head_kwargs_map = {
            tuple(RIGHT): {
                "border_top_right_radius": EXTRA_RADIUS,
                "border_bottom_right_radius": EXTRA_RADIUS,
            },
            tuple(LEFT): {
                "border_top_left_radius": EXTRA_RADIUS,
                "border_bottom_left_radius": EXTRA_RADIUS,
            },
            tuple(UP): {
                "border_top_left_radius": EXTRA_RADIUS,
                "border_top_right_radius": EXTRA_RADIUS,
            },
            tuple(DOWN): {
                "border_bottom_left_radius": EXTRA_RADIUS,
                "border_bottom_right_radius": EXTRA_RADIUS,
            },
        }
        head_kwargs = head_kwargs_map[tuple(snake.head_dir)]
        draw_rect(rect=snake.head_rect, **common_kwargs, **head_kwargs)

        # draw body
        for rect in snake.body_rects:
            draw_rect(rect=rect, **common_kwargs)

    def draw_game_row(self, game, rect):
        text = f"Player {game.player.name}, score: {game.score} steps: {game.steps}"
        if game.active is False:
            text += " - GAME OVER"
        draw_row(
            surf=self.score_surf,
            text=text,
            font=self.font_bold if game.active is True else self.font,
            font_color=game.color,
            rect=rect,
        )

    def draw_scoreboard(self, games, gen: str | None = None):

        FONT_COLOR = (175,) * 3

        text = "Use arrows, WSAD or vim-like HJKL"

        if all(game.active is False for game in games):
            text = "Press any key to quit."

        rect = pygame.Rect(0, 0, self.score_surf.get_width(), self.scoreboard_row_size)
        draw_row(self.score_surf, text, self.font, FONT_COLOR, rect)

        text = "Scoreboard"

        if gen is not None:
            text += f" gen: {gen}"

        rect = pygame.Rect(
            0,
            self.scoreboard_row_size,
            self.score_surf.get_width(),
            self.scoreboard_row_size,
        )
        draw_row(self.score_surf, text, self.font_bold, FONT_COLOR, rect)

        # TODO: active first, add score sorting
        active_games = sorted(
            [game for game in games if game.active],
            key=lambda g: (g.score, g.steps),
            reverse=True,
        )
        inactive_games = sorted(
            [game for game in games if not game.active],
            key=lambda g: (g.score, g.steps),
            reverse=True,
        )

        y_offset = 2 * self.scoreboard_row_size
        for idx, game in enumerate(active_games + inactive_games):
            rect = pygame.Rect(
                0,
                y_offset + idx * self.scoreboard_row_size,
                self.score_surf.get_width(),
                self.scoreboard_row_size,
            )
            self.draw_game_row(game, rect)

    def render_frame(self, games, gen: str | None = None, debug: bool = False):
        self.game_surf.fill(color=(175,) * 3)
        self.score_surf.fill(color=(25,) * 3)
        self.draw_grid()

        if debug:
            self.draw_coords()

        for game in games:
            self.draw_wall(game.wall)

        # separate for loop, bcs I want the snakes to be visible on top of wall
        for game in games:
            self.draw_snake(game.snake, game.active)
            self.draw_apple(game.apple, game.active)

        self.draw_scoreboard(games, gen)
        pygame.display.update()
