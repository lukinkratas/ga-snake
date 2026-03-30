import math
from typing import Any, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygame

from .const import (
    DIRECTIONS,
    DOWN,
    LEFT,
    RIGHT,
    UP,
)
from .engine import GAController, Game, HumanController
from .state import Apple, Snake, Wall


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
    font_color: tuple[int, int, int, int],
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
        font_family: str = "Arial",
        history_plot_surf: pygame.Surface | None = None,
        genome_plot_surf: pygame.Surface | None = None,
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
        self.history_plot_surf = history_plot_surf
        self.genome_plot_surf = genome_plot_surf
        self.scoreboard_row_height = int(1.4 * self.font_size)
        self.scoreboard_row_width = 160

        if self.history_plot_surf or self.genome_plot_surf:
            px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
            plt.rcParams["font.family"] = font_family
            matplotlib.use("pygame")

        if self.history_plot_surf:
            width_px = self.history_plot_surf.get_width()
            height_px = self.history_plot_surf.get_height()
            self.history_fig, self.history_ax = plt.subplots(
                figsize=(width_px * px, height_px * px)
            )

        if self.genome_plot_surf:
            width_px = self.history_plot_surf.get_width()
            height_px = self.history_plot_surf.get_height()
            self.genome_fig, self.genome_ax = plt.subplots(
                nrows=1, ncols=4, sharey=True, figsize=(width_px * px, height_px * px)
            )

    def get_square(self, coords: np.ndarray) -> pygame.Rect:
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

    def render_coords(self) -> None:
        """Render coordinates for debugging on the screen."""
        color = (150, 150, 150, 255)
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

    def render_wall(self, wall: Wall) -> None:
        """Render wall on the screen.

        Args:
            wall: wall
        """
        for c in wall.coords:
            render_rect(
                surf=self.game_surf,
                rect=self.get_square(c * self.grid_size),
                color=(*wall.color, 255),
                line_color=(25, 25, 25, 255),
                line_width=self.line_width,
                radius=self.rect_radius,
            )

    def render_apple(
        self, apple: Apple, color: tuple[int, int, int], alpha: int
    ) -> None:
        """Render apple on the screen.

        Args:
            apple: apple
            color: color to render the apple with
            alpha: opacity to render the apple with
        """
        rect = self.get_square(apple.coords * self.grid_size)
        render_circle(
            surf=self.game_surf,
            rect=rect,
            color=(*color, alpha),
            line_color=(25, 25, 25, alpha),
            line_width=self.line_width,
            radius=int(0.8 * self.grid_size / 2),
        )

    def render_snake(
        self,
        snake: Snake,
        color: tuple[int, int, int],
        alpha: int,
        name: str | None = None,
    ) -> None:
        """Render snake on the screen.

        Args:
            snake: snake
            color: color to render the snake with
            alpha: opacity to render the snake with
            name: name to render on the snake's head
        """
        extra_radius = 10
        common_kwargs = {
            "surf": self.game_surf,
            "color": (*color, alpha),
            "line_color": (25, 25, 25, alpha),
            "line_width": self.line_width,
            "radius": self.rect_radius,
        }

        # draw body first
        for c in snake.body_coords:
            render_rect(rect=self.get_square(c * self.grid_size), **common_kwargs)

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
        head_rect = self.get_square(snake.head_coords * self.grid_size)
        render_rect(rect=head_rect, **common_kwargs, **head_kwargs)

        if name:
            render_text_on_rect(
                surf=self.game_surf,
                text=name,
                font=self.font,
                font_color=(50, 50, 50, alpha),
                rect=head_rect,
            )

    def render_games(
        self, games: list[Game], alphas: Sequence[int] | None = None
    ) -> None:
        """Render games and corresponding walls, snakes and apples on the screen.

        Args:
            games: list of games
            alphas:
                sequence of opacities, corresponding to each game, used for snakes and
                apples rendering
        """
        if alphas is None:
            alphas = 255 * np.ones(len(games))

        self.game_surf.fill(color=(175,) * 3)
        self.score_surf.fill(color=(25,) * 3)
        self.render_grid()

        # render only unique wall object, if wall is shared
        uniq_walls = {game.wall for game in games}
        for wall in uniq_walls:
            self.render_wall(wall)

        # separate for loop, bcs I want the snakes to be visible on top of wall
        for game, alpha in zip(games, alphas):
            alpha = alpha if game.snake.is_alive else 31
            self.render_apple(game.apple, game.player.color, alpha)
            self.render_snake(game.snake, game.player.color, alpha)

    def render_player_row(self, game: Game, row_rect: pygame.Rect) -> None:
        """Render row per game/player in the scoreboard.

        Args:
            game: game
            row_rect: rectangle object
        """
        text = "Player"

        if game.player.name is not None:
            text += f" {game.player.name}"

        if isinstance(game.player.controller, HumanController) and not game.has_started:
            text += f" (use {game.player.controller.keymap_name})"

        text += f": {game.player.score}"

        if game.is_over:
            text += " - GAME OVER"

        render_text_on_rect(
            surf=self.score_surf,
            text=text,
            font=self.font if game.is_over else self.font_bold,
            font_color=(*game.player.color, 255),
            rect=row_rect,
        )

    def render_scoreboard(
        self,
        games: list[Game],
        gen: str | None = None,
    ) -> None:
        """Render scoreboard including title and score per player on the screen.

        Args:
            games: list of games
            gen: GA generation number (optional)
        """
        font_color = (175, 175, 175, 255)

        if all(game.is_over for game in games):
            text = "Q to quit or R to reset."
        else:
            text = "Press P to pause, Q to quit."

        row_rect = pygame.Rect(
            0, 0, self.score_surf.get_width(), self.scoreboard_row_height
        )
        render_text_on_rect(self.score_surf, text, self.font, font_color, row_rect)

        text = "Scoreboard"

        if gen is not None:
            text += f" gen: {gen}"

        row_rect = pygame.Rect(
            0,
            self.scoreboard_row_height,
            self.score_surf.get_width(),
            self.scoreboard_row_height,
        )
        render_text_on_rect(self.score_surf, text, self.font_bold, font_color, row_rect)

        # count rows for player scores, 3 reserved for intructions, title and footer
        nrows = int(self.score_surf.get_height() / self.scoreboard_row_height) - 3
        # ncols is either
        ncols = min(
            int(self.score_surf.get_width() / self.scoreboard_row_width),
            math.ceil(len(games) / nrows),
        )

        max_ngames = nrows * ncols
        print_more = len(games) > max_ngames
        available_games = games[: max_ngames - 1] if print_more else games

        y_offset = 2 * self.scoreboard_row_height
        for col_idx in range(ncols):
            from_idx = col_idx * nrows
            to_idx = (col_idx + 1) * nrows
            for row_idx, game in enumerate(available_games[from_idx:to_idx]):
                row_rect = pygame.Rect(
                    col_idx * self.score_surf.get_width() / ncols,
                    y_offset + row_idx * self.scoreboard_row_height,
                    self.score_surf.get_width() / ncols,
                    self.scoreboard_row_height,
                )
                self.render_player_row(game, row_rect)

        if print_more:
            row_rect = pygame.Rect(
                (ncols - 1) * self.score_surf.get_width() / ncols,
                y_offset + (nrows - 1) * self.scoreboard_row_height,
                self.score_surf.get_width() / ncols,
                self.scoreboard_row_height,
            )
            render_text_on_rect(
                surf=self.score_surf,
                text=f"and {len(games) - len(available_games)} more ....",
                font=self.font,
                font_color=font_color,
                rect=row_rect,
            )

    def render_paused(self) -> None:
        """Render paused on the screen."""
        font_color = (25, 25, 25, 255)
        text = "|| P A U S E D"
        rect = pygame.Rect(
            int(self.grid_size * self.ncols * 0.1),
            int(self.grid_size * self.nrows * 0.1),
            100,
            self.scoreboard_row_height,
        )
        render_text_on_rect(self.game_surf, text, self.font_bold, font_color, rect)

    def render_history_plot(
        self,
        best_fitness_history: list[float],
        avg_fitness_history: list[float],
        momentum: int | None = None,
    ) -> None:
        """Render matplotlib plot on the screen.

        Args:
            best_fitness_history: list of max fitness per generation
            avg_fitness_history: list of average fitness per generation
        """
        ngens = len(best_fitness_history)
        self.history_ax.clear()
        xs = np.arange(1, ngens + 1, dtype=np.int16)
        self.history_ax.bar(xs, best_fitness_history, label="max", color="tab:blue")
        self.history_ax.plot(
            xs, avg_fitness_history, label="avg", color="tab:orange", linewidth=2.0
        )
        if momentum:
            for x in np.arange(momentum, ngens + 1, momentum, dtype=np.int16):
                self.history_ax.axvline(x, color="tab:grey")

        self.history_ax.set_title("Fitness per Generation", fontweight="bold")
        self.history_ax.set_xlabel("Generation", fontsize=self.font_size)
        self.history_ax.set_ylabel("Fitness", fontsize=self.font_size)
        self.history_ax.legend(loc="upper left", fontsize=self.font_size, frameon=False)

        # needed to invoke dtype on axis
        nx = np.linspace(0, ngens + 1, num=min(ngens + 2, 12), dtype=np.int16)
        self.history_ax.set_xticks(nx)

        # plt.tight_layout(pad=6.0)
        # self.history_fig.subplots_adjust(left=0.5, bottom=0.5)
        self.history_fig.tight_layout(rect=[0.10, 0.05, 0.95, 0.95])
        # self.fig.align_labels()
        # self.fig.align_titles()

        self.history_fig.canvas.draw()
        self.history_plot_surf.blit(self.history_fig)

    def render_genome_plot(
        self,
        genome: np.ndarray,
        color: Sequence[float],
        name: str,
        fitness: float,
    ) -> None:
        """Render genome plot on the screen.

        Args:
            genome: array to plot
            color: color of genome's corresponding player
            name: name of genome's corresponding player
            fitness: fitness of genome's corresponding player
        """
        ys = [
            " ".join(fname.capitalize().split("_"))
            for fname in GAController.FEATURE_NAMES
        ]
        for idx, direction in enumerate(DIRECTIONS.keys()):
            ax = self.genome_ax[idx]

            ax.clear()

            xs = genome[:, idx]
            # ax.scatter(xs, feature_names, s=100 * np.abs(xs), alpha=1.0)
            ax.plot(xs, ys, marker="o", linestyle="", color=color)

            ax.set_title(direction.capitalize(), fontsize=self.font_size)
            ax.set_xlim(min(*xs, -1), max(*xs, 1))
            ax.grid(True)

        title = f"Last Best Genome {name} (fitness: {fitness:.2f})"
        self.genome_fig.suptitle(title, color=color, fontweight="bold")
        self.genome_fig.tight_layout(rect=[0.20, 0.05, 0.95, 0.95])

        self.genome_fig.canvas.draw()
        self.genome_plot_surf.blit(self.genome_fig)
