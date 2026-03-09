import math
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygame

from .const import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
)
from .engine import GameBase, HumanController
from .state import AppleBase, Snake, Wall

ALPHA_MAP = {True: 255, False: 63}


def render_circle(
    surf: pygame.Surface,
    rect: pygame.Rect,
    color: tuple[int, int, int, int],
    line_color: tuple[int, int, int, int],
    line_width: int,
    radius: int,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Render circle on the surface.

    Args:
        surf: pygame surface instance
        rect: pygame rectangle instance
        color: inner color of the rectangle
        line_color: outer color of the rectangle
        line_width: width of the outer line
        radius: radius of the circle
        args: arguments to be passed to pygame.draw method
        kwargs: arguments to be passed to pygame.draw method
    """
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


def render_rect(
    surf: pygame.Surface,
    rect: pygame.Rect,
    color: tuple[int, int, int, int],
    line_color: tuple[int, int, int, int],
    line_width: int,
    radius: int,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Render a rectangle on the surface.

    Args:
        surf: pygame surface instance
        rect: pygame rectangle instance
        color: inner color of the rectangle
        line_color: outer color of the rectangle
        line_width: width of the outer line
        radius: corner radius of the rectangle
        args: arguments to be passed to pygame.draw method
        kwargs: arguments to be passed to pygame.draw method
    """
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


def render_text_on_rect(
    surf,
    text: str,
    font: pygame.Font,
    font_color: tuple[int, int, int],
    rect: pygame.Rect,
) -> None:
    """Render text on given rectangle.

    Args:
        surf: pygame surface instance
        text: text
        font: font of the text
        font_color: color of the text
        rect: pygame rectangle instance
    """
    text_surf = font.render(text, False, font_color)
    text_rect = text_surf.get_rect(center=rect.center)
    surf.blit(text_surf, text_rect)


class Renderer:
    def __init__(
        self,
        game_surf: pygame.Surface,
        score_surf: pygame.Surface,
        ncols: int,
        nrows: int,
        grid_size: int,
        rect_radius: int,
        line_width: int,
        font_size: int,
        scoreboard_row_size: int,
        font_family: str = "Arial",
        plot_surf: pygame.Surface | None = None,
    ) -> None:
        self.game_surf = game_surf
        self.score_surf = score_surf
        self.ncols = ncols
        self.nrows = nrows
        self.grid_size = grid_size
        self.rect_radius = rect_radius
        self.line_width = line_width
        self.font_size = font_size
        self.font = pygame.font.SysFont(font_family, font_size)
        self.font_bold = pygame.font.SysFont(font_family, font_size, bold=True)
        self.scoreboard_row_size = scoreboard_row_size
        self.plot_surf = plot_surf

        if self.plot_surf:
            matplotlib.use("pygame")
            px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
            plt.rcParams["font.family"] = font_family
            width_px = self.plot_surf.get_width()
            height_px = self.plot_surf.get_height()
            self.fig, self.ax = plt.subplots(figsize=(width_px * px, height_px * px))
            self.fig.set_tight_layout(True)

    def get_square(self, coords: np.ndarray) -> pygame.rect:
        """Draw square of grid_size.

        Args:
            coords: coordinates of topleft corner of square

        Returns: rectangle instance
        """
        return pygame.Rect(*coords, self.grid_size, self.grid_size)

    def render_grid(self) -> None:
        """Render grid on the screen."""
        line_color = (150,) * 3

        x_range = np.arange(
            self.grid_size, self.game_surf.get_width() - 1, self.grid_size
        )
        for x in x_range:
            pygame.draw.line(
                surface=self.game_surf,
                color=line_color,
                start_pos=(x, 0),
                end_pos=(x, self.game_surf.get_height()),
                width=1,
            )

        y_range = np.arange(
            self.grid_size, self.game_surf.get_height() - 1, self.grid_size
        )
        for y in y_range:
            pygame.draw.line(
                surface=self.game_surf,
                color=line_color,
                start_pos=(0, y),
                end_pos=(self.game_surf.get_width(), y),
                width=1,
            )

    def render_wall(self, wall: Wall) -> None:
        """Render wall on the screen.

        Args:
            wall: wall
        """
        for c in wall.coords:
            render_rect(
                surf=self.game_surf,
                rect=self.get_square(c * self.grid_size),
                color=wall.color,
                line_color=(25, 25, 25, 255),
                line_width=self.line_width,
                radius=self.rect_radius,
            )

    def render_apple(self, apple: AppleBase, active: bool) -> None:
        """Render apple on the screen.

        Args:
            apple: apple
            active: whether or not is the apple active (affects opacity)
        """
        alpha = ALPHA_MAP[active]
        render_circle(
            surf=self.game_surf,
            rect=self.get_square(apple.coords * self.grid_size),
            color=(*apple.color, alpha),
            line_color=(25, 25, 25, alpha),
            line_width=self.line_width,
            radius=int(0.8 * self.grid_size / 2),
        )

    def render_snake(self, snake: Snake) -> None:
        """Render snake on the screen.

        Args:
            snake: snake
        """
        extra_radius = 10
        alpha = ALPHA_MAP[snake.is_alive]
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
                "border_top_right_radius": extra_radius,
                "border_bottom_right_radius": extra_radius,
            },
            tuple(LEFT): {
                "border_top_left_radius": extra_radius,
                "border_bottom_left_radius": extra_radius,
            },
            tuple(UP): {
                "border_top_left_radius": extra_radius,
                "border_top_right_radius": extra_radius,
            },
            tuple(DOWN): {
                "border_bottom_left_radius": extra_radius,
                "border_bottom_right_radius": extra_radius,
            },
        }
        head_kwargs = head_kwargs_map[tuple(snake.head_dir)]
        render_rect(
            rect=self.get_square(snake.head_coords * self.grid_size),
            **common_kwargs,
            **head_kwargs,
        )

        # draw body
        for c in snake.body_coords:
            render_rect(rect=self.get_square(c * self.grid_size), **common_kwargs)

    def render_coords(self) -> None:
        """Render coordinates for debugging on the screen."""
        color = (150,) * 3
        font = pygame.font.SysFont("Arial", 8)

        x_range = np.arange(self.ncols)
        y_range = np.arange(self.nrows)

        for x in x_range:
            for y in y_range:
                text = f"x{int(x)}"
                rect = pygame.Rect(
                    x * self.grid_size,
                    y * self.grid_size,
                    self.grid_size,
                    self.grid_size / 2,
                )
                render_text_on_rect(self.game_surf, text, font, color, rect)

                text = f"y{int(y)}"
                rect = pygame.Rect(
                    x * self.grid_size,
                    y * self.grid_size + self.grid_size / 2,
                    self.grid_size,
                    self.grid_size / 2,
                )
                render_text_on_rect(self.game_surf, text, font, color, rect)

    def render_games(self, games: list[GameBase], gen: str | None = None) -> None:
        """Render games and corresponding walls, snakesa and apples on the screen.

        Args:
            games: list of games
            gen: GA generation number (optional)
        """
        self.game_surf.fill(color=(175,) * 3)
        self.score_surf.fill(color=(25,) * 3)
        self.render_grid()

        # render only unique wall object, if wall is shared
        for wall in {game.wall for game in games}:
            self.render_wall(wall)

        # separate for loop, bcs I want the snakes to be visible on top of wall
        # inactive games with lowest score render first
        # active games with highest score render last
        sorted_games = sorted(
            [game for game in games],
            key=lambda g: (g.player.score, g.steps, not g.is_over),
        )
        for game in sorted_games:
            self.render_snake(game.snake)
            self.render_apple(game.apple, game.snake.is_alive)

    def render_game_row(self, game: GameBase, rect: pygame.Rect) -> None:
        """Render row per game/player in the scoreboard.

        Args:
            game: game
            rect: rectangle object
        """
        text = "Player"
        if game.player.name is not None:
            text += f" {game.player.name}"
        if isinstance(game.player.controller, HumanController) and game.steps == 0:
            text += f" (use {game.player.controller.keymap_name})"
        text += f", score: {game.player.score} steps: {game.steps}"

        if game.is_over:
            text += " - GAME OVER"

        render_text_on_rect(
            surf=self.score_surf,
            text=text,
            font=self.font if game.is_over else self.font_bold,
            font_color=game.player.color,
            rect=rect,
        )

    def render_scoreboard(self, games: list[GameBase], gen: str | None = None) -> None:
        """Render scoreboard including title and score per player on the screen.

        Args:
            games: list of games
            gen: GA generation number (optional)
        """
        font_color = (175,) * 3

        if all(game.is_over for game in games):
            text = "Q to quit or R to reset."
        else:
            text = "Press P to pause, Q to quit."

        rect = pygame.Rect(0, 0, self.score_surf.get_width(), self.scoreboard_row_size)
        render_text_on_rect(self.score_surf, text, self.font, font_color, rect)

        text = "Scoreboard"

        if gen is not None:
            text += f" gen: {gen}"

        rect = pygame.Rect(
            0,
            self.scoreboard_row_size,
            self.score_surf.get_width(),
            self.scoreboard_row_size,
        )
        render_text_on_rect(self.score_surf, text, self.font_bold, font_color, rect)

        sorted_games = sorted(
            [game for game in games],
            key=lambda g: (g.player.score, g.steps, not g.is_over),
            reverse=True,
        )

        available_height = self.score_surf.get_height()
        # count rows for player scores, 3 reserved for intructions, title and footer
        available_nrows = int(available_height / self.scoreboard_row_size) - 3
        # round up numbber of cols
        ncols = math.ceil((len(games) / available_nrows))
        y_offset = 2 * self.scoreboard_row_size
        for col_idx in range(ncols):
            from_idx = col_idx * available_nrows
            to_idx = (col_idx + 1) * available_nrows
            for row_idx, game in enumerate(sorted_games[from_idx:to_idx]):
                rect = pygame.Rect(
                    col_idx * self.score_surf.get_width() / ncols,
                    y_offset + row_idx * self.scoreboard_row_size,
                    self.score_surf.get_width() / ncols,
                    self.scoreboard_row_size,
                )
                self.render_game_row(game, rect)

    def render_plot(self, best_fitness_history: list[np.ndarray]) -> None:
        """Render matplotlib plot on the screen.

        Args:
            best_fitness_history: list of best fitness per generation
        """
        ngens = len(best_fitness_history)
        self.ax.clear()
        xs = np.arange(1, ngens + 1, dtype=np.int16)
        self.ax.bar(xs, best_fitness_history)

        self.ax.set_title("Best Fitness per Generation")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")

        # needed to invoke dtype on axis
        nx = np.linspace(0, ngens + 1, num=min(ngens + 2, 12), dtype=np.int16)
        self.ax.set_xticks(nx)

        plt.tight_layout(pad=3.0)
        # self.fig.subplots_adjust(left=0.2, bottom=0.2)
        # self.fig.tight_layout(rect=[2, 2, 2, 2])
        # self.fig.align_labels()
        # self.fig.align_titles()

        self.fig.canvas.draw()
        self.plot_surf.blit(self.fig)

    def render_paused(self) -> None:
        """Render paused on the screen."""
        font_color = (25,) * 3
        text = "|| P A U S E D"
        rect = pygame.Rect(
            int(self.grid_size * self.ncols * 0.1),
            int(self.grid_size * self.nrows * 0.1),
            100,
            self.scoreboard_row_size,
        )
        render_text_on_rect(self.game_surf, text, self.font_bold, font_color, rect)
