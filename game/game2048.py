import os
import pygame
import random
import json
import numpy as np
from copy import deepcopy

from config.config import load_config

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


class Game2048:
    def __init__(self, config=load_config("game2048"), silent_mode=True):
        self.config = config  # 配置
        self.silent_mode = silent_mode  # 可视化
        self.reset()

        if not silent_mode:
            pygame.init()
            pygame.display.set_caption("2048")
            self.screen = pygame.display.set_mode(
                (self.config["width"], self.config["height"])
            )
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, self.config["font_size"])

    def reset(self):
        self.grid = [
            [0] * self.config["grid_size"] for _ in range(self.config["grid_size"])
        ]
        self.score = 0
        self._add_random_tile()
        self._add_random_tile()
        info = {
            "grid": self.grid,
            "score": self.score,
            "max_tile": max(max(row) for row in self.grid),
        }
        return info

    def step(self, action):
        direction_map = {0: "left", 1: "right", 2: "up", 3: "down"}
        direction = direction_map.get(action, None)
        moved = self._move(direction)
        if direction and moved:
            self._add_random_tile()

        done = self._check_game_over()
        info = {
            "moved": moved,
            "grid": self.grid,
            "score": self.score,
            "max_tile": max(max(row) for row in self.grid),
        }
        return done, info

    def _add_random_tile(self):
        candidates = [
            (i, j)
            for i in range(self.config["grid_size"])
            for j in range(self.config["grid_size"])
            if self.grid[i][j] == 0
        ]
        if candidates:
            i, j = random.choice(candidates)
            randomness = random.random()
            cur_prob = 0
            for value, prob in self.config["new_tile_value"].items():
                cur_prob += prob
                if randomness < cur_prob:
                    self.grid[i][j] = value
                    return

    def _process_row(self, row, reverse=False):
        filtered = [x for x in (reversed(row) if reverse else row) if x != 0]
        merged = []
        skip = False

        for i in range(len(filtered)):
            if skip:
                skip = False
                continue
            if i + 1 < len(filtered) and filtered[i] == filtered[i + 1]:
                merged.append(filtered[i] * 2)
                self.score += filtered[i] * 2
                skip = True
            else:
                merged.append(filtered[i])

        padding = [0] * (self.config["grid_size"] - len(merged))
        return (merged + padding) if not reverse else padding + merged[::-1]

    def _move(self, direction):
        moved = False
        original = [row.copy() for row in self.grid]

        if direction in ("left", "right"):
            reverse = direction == "right"
            for i in range(self.config["grid_size"]):
                processed = self._process_row(original[i], reverse)
                if processed != original[i]:
                    moved = True
                self.grid[i] = processed

        elif direction in ("up", "down"):
            reverse = direction == "down"
            for j in range(self.config["grid_size"]):
                col = [self.grid[i][j] for i in range(self.config["grid_size"])]
                processed = self._process_row(col, reverse)
                if processed != col:
                    moved = True
                for i in range(self.config["grid_size"]):
                    self.grid[i][j] = processed[i]
        return moved

    def _render_grid(self):
        self.screen.fill(self.config["background_color"])
        for i in range(self.config["grid_size"]):
            for j in range(self.config["grid_size"]):
                self._render_tile(i, j)

    def _render_tile(self, i, j):
        tile_value = self.grid[i][j]
        tile_color = self._get_tile_color(tile_value)
        rect = pygame.Rect(
            j * self.config["tile_size"] + self.config["grid_padding"],
            i * self.config["tile_size"] + self.config["grid_padding"],
            self.config["tile_size"] - 2 * self.config["grid_padding"],
            self.config["tile_size"] - 2 * self.config["grid_padding"],
        )
        pygame.draw.rect(self.screen, tile_color, rect, border_radius=3)

        if tile_value != 0:
            font_color = self.config["font_colors"].get(tile_value, (255, 255, 255))
            text = self.font.render(str(tile_value), True, font_color)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

    def _get_tile_color(self, value):
        return self.config["tile_colors"].get(value, (60, 58, 50))

    def _check_game_over(self):
        if any(0 in row for row in self.grid):
            return False
        for i in range(self.config["grid_size"]):
            for j in range(self.config["grid_size"]):
                if (
                    j + 1 < self.config["grid_size"]
                    and self.grid[i][j] == self.grid[i][j + 1]
                ) or (
                    i + 1 < self.config["grid_size"]
                    and self.grid[i][j] == self.grid[i + 1][j]
                ):
                    return False
        return True


def replay(config, grid_history, action_history, delay=1500):
    """game replay function"""
    if not grid_history or not action_history:
        print("No replay data available.")
        return

    game = Game2048(config=config, silent_mode=False)
    screen = game.screen
    clock = game.clock
    font = game.font
    action_names = {0: "left", 1: "right", 2: "up", 3: "down"}

    current_step = 0
    total_steps = len(grid_history)
    paused = False
    speed_factor = 1.0
    running = True
    need_redraw = True

    while running:
        # render the current step if needed
        if need_redraw:
            screen.fill(config["background_color"])
            game.grid = deepcopy(grid_history[current_step])
            game._render_grid()

            info_surface = pygame.Surface((config["width"], 40), pygame.SRCALPHA)
            info_surface.fill((0, 0, 0, 96))
            screen.blit(info_surface, (0, 0))

            step_text = font.render(
                f"Step: {current_step}/{total_steps - 1}  Action: {action_names.get(action_history[current_step], 'N/A')}",
                True,
                (255, 255, 255),
            )
            text_rect = step_text.get_rect(center=(config["width"] // 2, 20))
            screen.blit(step_text, text_rect)

            pygame.display.flip()
            need_redraw = False

        # update game state during playback
        if not paused:
            if current_step < total_steps - 1:
                pygame.time.delay(int(delay / speed_factor))
                current_step += 1
                need_redraw = True
            else:
                paused = True

        # handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Exiting replay...")
                running = False
                break

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    current_step = current_step - 1 if need_redraw else current_step
                    need_redraw = False

                elif event.key == pygame.K_ESCAPE:
                    running = False

                elif paused and event.key == pygame.K_RIGHT:  # next step
                    current_step = min(current_step + 1, total_steps - 1)
                    need_redraw = True

                elif paused and event.key == pygame.K_LEFT:  # previous step
                    current_step = max(current_step - 1, 0)
                    need_redraw = True

                elif event.key == pygame.K_UP:
                    speed_factor = min(speed_factor * 1.2, 5.0)

                elif event.key == pygame.K_DOWN:
                    speed_factor = max(speed_factor / 1.2, 0.1)

        # control frame rate
        if not paused:
            clock.tick(int(60 * speed_factor))
        else:
            clock.tick(240)


def main(config=load_config("game2048")):
    # pygame 初始化和设置
    game = Game2048(config=config, silent_mode=False)
    screen = game.screen
    clock = game.clock
    font = game.font

    # 加载最高分和游戏记录
    high_score = 0
    try:
        with open(config["archive_file"], "r") as f:
            game_data = json.load(f)
            high_score = game_data.get("score", 0)
            grid_history = game_data.get("grid_history", [])  # 游戏网格记录
            action_history = game_data.get("action_history", [])  # 动作记录
    except (FileNotFoundError, json.JSONDecodeError):
        high_score, grid_history, action_history = 0, [], []

    # 主循环
    state = "menu"  # 游戏状态: menu, playing, game_over
    last_grid_surface = None  # 用于保存结束时的游戏画面
    running = True

    while running:
        screen.fill(config["background_color"])

        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Exiting game...")
                running = False

            # 鼠标点击事件
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()

                if state == "menu":
                    main_btn_rect = pygame.Rect(
                        config["width"] // 2 - 100, config["height"] // 2 - 40, 200, 60
                    )
                    replay_btn_rect = pygame.Rect(
                        config["width"] // 2 - 100, config["height"] // 2 + 40, 200, 60
                    )
                    if main_btn_rect.collidepoint(mouse_pos):
                        game = Game2048(silent_mode=False)
                        game._render_grid()
                        state = "playing"
                        grid_history = [deepcopy(game.grid)]
                        action_history = [-1]

                    if replay_btn_rect.collidepoint(mouse_pos):
                        replay(config, grid_history, action_history)
                        state = "menu"

                elif state == "game_over":
                    restart_btn_rect = pygame.Rect(
                        config["width"] // 2 - 100, config["height"] // 2, 200, 60
                    )
                    replay_btn_rect = pygame.Rect(
                        config["width"] // 2 - 100, config["height"] // 2 + 80, 200, 60
                    )

                    if restart_btn_rect.collidepoint(mouse_pos):
                        game.reset()
                        game._render_grid()
                        state = "playing"
                        grid_history = [deepcopy(game.grid)]
                        action_history = [-1]

                    if replay_btn_rect.collidepoint(mouse_pos):
                        replay(config, grid_history, action_history)
                        state = "game_over"

            # 键盘事件
            elif event.type == pygame.KEYDOWN and state == "playing":
                if event.key == pygame.K_ESCAPE:
                    state = "menu"
                    game = None

                else:
                    action = (
                        0
                        if event.key == pygame.K_LEFT
                        else (
                            1
                            if event.key == pygame.K_RIGHT
                            else (
                                2
                                if event.key == pygame.K_UP
                                else 3 if event.key == pygame.K_DOWN else None
                            )
                        )
                    )
                    _, info = game.step(action)
                    game._render_grid()
                    if info["moved"]:
                        action_history.append(action)
                        grid_history.append(deepcopy(game.grid))

        # 游戏逻辑状态处理
        if state == "playing" and game and game._check_game_over():
            if game.score > high_score:
                high_score = game.score
                with open(config["archive_file"], "w") as f:
                    json.dump(
                        {
                            "score": high_score,
                            "grid_history": grid_history,
                            "action_history": action_history,
                        },
                        f,
                    )

            last_grid_surface = screen.copy()
            state = "game_over"

        # 界面渲染
        if state == "menu":
            # 绘制菜单按钮
            main_btn_rect = pygame.Rect(
                config["width"] // 2 - 100, config["height"] // 2 - 40, 200, 60
            )
            pygame.draw.rect(
                screen, config["btn_color"], main_btn_rect, border_radius=15
            )
            btn_text = font.render("START", True, (255, 255, 255))
            text_rect = btn_text.get_rect(center=main_btn_rect.center)
            screen.blit(btn_text, text_rect)

            # 回放按钮
            replay_btn_rect = pygame.Rect(
                config["width"] // 2 - 100, config["height"] // 2 + 40, 200, 60
            )
            pygame.draw.rect(screen, (70, 130, 180), replay_btn_rect, border_radius=15)
            replay_text = font.render("REPLAY", True, (255, 255, 255))
            text_rect = replay_text.get_rect(center=replay_btn_rect.center)
            screen.blit(replay_text, text_rect)

            # 显示高分
            score_text = font.render(
                f"High Score: {high_score}", True, config["font_color"]
            )
            screen.blit(score_text, (20, 20))

        elif state == "playing":
            game._render_grid()

        elif state == "game_over" and last_grid_surface:
            # 绘制游戏结束界面
            screen.blit(last_grid_surface, (0, 0))
            overlay = pygame.Surface(
                (config["width"], config["height"]), pygame.SRCALPHA
            )
            overlay.fill((255, 255, 255, 128))
            screen.blit(overlay, (0, 0))

            # 游戏结束信息
            text = font.render(f"Final Score: {game.score}", True, (255, 87, 87))
            text_rect = text.get_rect(
                center=(config["width"] // 2, config["height"] // 2 - 60)
            )
            screen.blit(text, text_rect)

            # 重新开始按钮
            restart_btn_rect = pygame.Rect(
                config["width"] // 2 - 100, config["height"] // 2, 200, 60
            )
            pygame.draw.rect(
                screen, config["btn_color"], restart_btn_rect, border_radius=15
            )
            restart_text = font.render("RETRY", True, (255, 255, 255))
            text_rect = restart_text.get_rect(center=restart_btn_rect.center)
            screen.blit(restart_text, text_rect)

            # 回放按钮
            replay_btn_rect = pygame.Rect(
                config["width"] // 2 - 100, config["height"] // 2 + 80, 200, 60
            )
            pygame.draw.rect(screen, (70, 130, 180), replay_btn_rect, border_radius=15)
            replay_text = font.render("REPLAY", True, (255, 255, 255))
            text_rect = replay_text.get_rect(center=replay_btn_rect.center)
            screen.blit(replay_text, text_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
