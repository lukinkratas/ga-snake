from typing import Any

import numpy as np
import pygame

from .const import DOWN, GAME_HEIGHT, GRID_SIZE, LEFT, RIGHT, UP, WIDTH
from .state import Apple, Snake, Wall


def draw_circle(
    surf,
    rect,
    color,
    line_color,
    line_width: int = 2,
    radius: int = 8,
    *args: Any,
    **kwargs: Any,
):
    # draw filled circle
    pygame.draw.circle(
        surface=surf, color=color, center=rect.center, radius=radius, *args, **kwargs
    )
    # draw line around circle
    pygame.draw.circle(
        surface=surf,
        color=line_color,
        center=rect.center,
        radius=radius,
        width=line_width,
        *args,
        **kwargs,
    )
    return rect


def draw_rect(
    surf,
    rect,
    color,
    line_color,
    line_width: int = 2,
    radius: int = 5,
    *args: Any,
    **kwargs: Any,
):
    # draw filled rect
    pygame.draw.rect(
        surface=surf, color=color, rect=rect, border_radius=radius, *args, **kwargs
    )
    # draw line around rect
    pygame.draw.rect(
        surface=surf,
        color=line_color,
        rect=rect,
        width=line_width,
        border_radius=radius,
        *args,
        **kwargs,
    )
    return rect


def draw_apple(surf: pygame.Surface, apple: Apple):
    draw_circle(surf, apple.rect, apple.color, line_color=tuple(25 * np.ones(3)))


def draw_snake(surf: pygame.Surface, snake: Snake):

    EXTRA_RADIUS = 10
    common_kwargs = {
        "surf": surf,
        "color": snake.color,
        "line_color": tuple(50 * np.ones(3)),
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
    head_kwargs = head_kwargs_map.get(tuple(snake.head_dir))
    draw_rect(rect=snake.head_rect, **common_kwargs, **head_kwargs)

    # draw body
    for rect in snake.body_rects:
        draw_rect(rect=rect, **common_kwargs)

    # draw tail
    tail_kwargs_map = {
        tuple(RIGHT): {
            "border_top_left_radius": EXTRA_RADIUS,
            "border_bottom_left_radius": EXTRA_RADIUS,
        },
        tuple(LEFT): {
            "border_top_right_radius": EXTRA_RADIUS,
            "border_bottom_right_radius": EXTRA_RADIUS,
        },
        tuple(UP): {
            "border_bottom_left_radius": EXTRA_RADIUS,
            "border_bottom_right_radius": EXTRA_RADIUS,
        },
        tuple(DOWN): {
            "border_top_left_radius": EXTRA_RADIUS,
            "border_top_right_radius": EXTRA_RADIUS,
        },
    }
    tail_kwargs = tail_kwargs_map.get(tuple(snake.tail_dir))
    draw_rect(rect=snake.tail_rect, **common_kwargs, **tail_kwargs)


def draw_grid(surf, line_color: tuple[int] = tuple(150 * np.ones(3))):
    for x in np.arange(GRID_SIZE, WIDTH - 1, GRID_SIZE):
        pygame.draw.line(
            surface=surf,
            color=line_color,
            start_pos=(x, 0),
            end_pos=(x, GAME_HEIGHT),
            width=1,
        )

    for y in np.arange(GRID_SIZE, GAME_HEIGHT - 1, GRID_SIZE):
        pygame.draw.line(
            surface=surf,
            color=line_color,
            start_pos=(0, y),
            end_pos=(WIDTH, y),
            width=1,
        )


def draw_wall(surf, wall: Wall):

    for rect in wall.rects:
        draw_rect(surf, rect, color=wall.color, line_color=tuple(25 * np.ones(3)))


def draw_scoreboard(surf, players):

    font_size = 14
    font_color = tuple(175 * np.ones(3))
    font = pygame.font.SysFont("Arial", font_size)
    font_bold = pygame.font.SysFont("Arial", font_size, pygame.font.Font.bold)
    row_size = font_size + 6

    def draw_row(surf, text, font, font_color, rect):
        text_surf = font.render(text, False, font_color)
        text_rect = text_surf.get_rect(center=rect.center)
        surf.blit(text_surf, text_rect)

    draw_row(
        surf=surf,
        text="Use arrows or vim-like H J K L or W S A D",
        font=font,
        font_color=font_color,
        rect=pygame.Rect(0, 0, surf.get_width(), row_size),
    )

    draw_row(
        surf=surf,
        text="Scoreboard",
        font=font_bold,
        font_color=font_color,
        rect=pygame.Rect(0, row_size, surf.get_width(), row_size),
    )

    y_offset = 2 * row_size
    for idx, player in enumerate(players):
        text = f"Player no. {idx + 1}, score: {player.score}"
        if player.snake.alive is False:
            text += " - DEAD"
        draw_row(
            surf=surf,
            text=text,
            font=font_bold if player.snake.alive else font,
            font_color=player.color,
            rect=pygame.Rect(0, y_offset + idx * row_size, surf.get_width(), row_size),
        )


def render_frame(game_surf: pygame.Surface, score_surf: pygame.Surface, game):
    game_surf.fill(color=tuple(175 * np.ones(3)))
    score_surf.fill(color=tuple(25 * np.ones(3)))
    draw_grid(game_surf)
    draw_wall(game_surf, game.wall)
    for player in [game.player]:
        draw_snake(game_surf, player.snake)
    draw_apple(game_surf, game.apple)
    draw_scoreboard(score_surf, [game.player])
    pygame.display.update()
