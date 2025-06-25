import os
import pygame
import random
import json
import numpy as np

from config.config import load_config

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


class Game2048:
    def __init__(self, config=load_config("game2048"), silent_mode=True):
        self.config = config  # 配置
        self.silent_mode = silent_mode  # 可视化
        self.reset()
        if not silent_mode:
            pygame.init()
            self._init_gui()

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
            if not self.silent_mode:
                self._render_grid()

        done = self._check_game_over()
        info = {
            "moved": moved,
            "grid": self.grid,
            "score": self.score,
            "max_tile": max(max(row) for row in self.grid),
        }
        return done, info

    def _init_gui(self):
        self.font = pygame.font.Font(None, self.config["font_size"])
        self.screen = pygame.display.set_mode(
            (self.config["width"], self.config["height"])
        )
        pygame.display.set_caption("2048")
        self.clock = pygame.time.Clock()
        self._render_grid()

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
        pygame.display.flip()
        self.clock.tick(30)

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


def replay(config, grid_history, action_history, delay=1000):
    """重播游戏过程"""
    pygame.init()
    screen = pygame.display.set_mode((config["width"], config["height"]))
    pygame.display.set_caption("2048 Replay")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, config["font_size"])

    # 动作名称映射
    action_names = {0: "LEFT", 1: "RIGHT", 2: "UP", 3: "DOWN"}

    # 创建游戏实例用于渲染
    game = Game2048(silent_mode=False)

    # 回放状态
    current_step = 0
    total_steps = len(grid_history)
    paused = False
    speed_factor = 1.0

    # 主循环
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused  # 空格键暂停/继续
                elif event.key == pygame.K_RIGHT:
                    # 右箭头键前进一帧
                    current_step = min(current_step + 1, total_steps - 1)
                elif event.key == pygame.K_LEFT:
                    # 左箭头键后退一帧
                    current_step = max(current_step - 1, 0)
                elif event.key == pygame.K_UP:
                    # 上箭头键增加速度
                    speed_factor = min(speed_factor * 2, 8.0)
                elif event.key == pygame.K_DOWN:
                    # 下箭头键减小速度
                    speed_factor = max(speed_factor / 2, 0.125)
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if not paused:
            # 更新当前步
            current_step = min(current_step + 1, total_steps - 1)

        # 设置游戏网格到当前状态
        game.grid = [row[:] for row in grid_history[current_step]]

        # 渲染网格
        game._render_grid()

        # 显示回放信息
        info_surface = pygame.Surface((config["width"], 40), pygame.SRCALPHA)
        info_surface.fill((0, 0, 0, 128))
        screen.blit(info_surface, (0, 0))

        # 显示当前步和总步数
        step_text = font.render(
            f"Step: {current_step}/{total_steps-1}  Action: {action_names.get(action_history[current_step - 1], 'N/A')}",
            True,
            (255, 255, 255),
        )
        screen.blit(step_text, (10, 10))
        pygame.display.flip()

        # 控制播放速度
        if not paused:
            clock.tick(30 / speed_factor)
        else:
            clock.tick(30)

        # 延迟
        pygame.time.delay(int(delay / speed_factor))

        # 检查是否结束
        if current_step >= total_steps - 1 and not paused:
            pygame.display.flip()
            paused = True

    pygame.quit()


def main(config):
    # pygame 初始化和设置
    pygame.init()
    screen = pygame.display.set_mode((config["width"], config["height"]))
    pygame.display.set_caption("2048")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, config["font_size"])

    # 加载最高分
    high_score = 0
    try:
        with open(config["archive_file"], "r") as f:
            game_data = json.load(f)
            high_score = game_data.get("score", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # 游戏状态
    game = None
    state = "menu"  # menu, playing, game_over
    last_grid_surface = None  # 用于保存结束时的游戏画面

    # 游戏记录
    game_history = []  # 存储每一步的网格状态
    action_history = []  # 存储每一步的动作

    # 主循环
    running = True
    while running:
        screen.fill(config["background_color"])

        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # 鼠标点击事件
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()

                if state == "menu":
                    main_btn_rect = pygame.Rect(
                        config["width"] // 2 - 100, config["height"] // 2 - 35, 200, 60
                    )
                    if main_btn_rect.collidepoint(mouse_pos):
                        game = Game2048(silent_mode=False)
                        game._render_grid()
                        state = "playing"
                        # 重置记录
                        game_history = [np.array(game.grid)]
                        action_history = []

                elif state == "game_over":
                    restart_btn_rect = pygame.Rect(
                        config["width"] // 2 - 80, config["height"] // 2 + 20, 160, 50
                    )
                    replay_btn_rect = pygame.Rect(
                        config["width"] // 2 - 80, config["height"] // 2 + 80, 160, 50
                    )

                    if restart_btn_rect.collidepoint(mouse_pos):
                        game = Game2048(silent_mode=False)
                        game._render_grid()
                        state = "playing"
                        # 重置记录
                        game_history = [np.array(game.grid)]
                        action_history = []

                    if replay_btn_rect.collidepoint(mouse_pos):
                        # 开始回放
                        replay(config, game_history, action_history)
                        # 回放结束后返回游戏结束状态
                        state = "game_over"

            # 键盘事件
            elif event.type == pygame.KEYDOWN and state == "playing":
                if event.key == pygame.K_LEFT:
                    action = 0
                    done, info = game.step(action)
                    if info["moved"]:
                        action_history.append(action)
                        game_history.append(np.array(game.grid))
                elif event.key == pygame.K_RIGHT:
                    action = 1
                    done, info = game.step(action)
                    if info["moved"]:
                        action_history.append(action)
                        game_history.append(np.array(game.grid))
                elif event.key == pygame.K_UP:
                    action = 2
                    done, info = game.step(action)
                    if info["moved"]:
                        action_history.append(action)
                        game_history.append(np.array(game.grid))
                elif event.key == pygame.K_DOWN:
                    action = 3
                    done, info = game.step(action)
                    if info["moved"]:
                        action_history.append(action)
                        game_history.append(np.array(game.grid))
                elif event.key == pygame.K_ESCAPE:
                    state = "menu"
                    game = None

        # 游戏逻辑状态处理
        if state == "playing" and game and game._check_game_over():
            if game.score > high_score:
                high_score = game.score
                with open(config["archive_file"], "w") as f:
                    json.dump({"score": high_score}, f)

            # 保存最后画面并进入结束状态
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
            pygame.draw.rect(screen, (70, 130, 180), replay_btn_rect, border_radius=8)
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
            # 显示最后的游戏画面
            screen.blit(last_grid_surface, (0, 0))

            # 半透明遮罩
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
                screen, config["btn_color"], restart_btn_rect, border_radius=8
            )
            restart_text = font.render("RETRY", True, (255, 255, 255))
            text_rect = restart_text.get_rect(center=restart_btn_rect.center)
            screen.blit(restart_text, text_rect)

            # 回放按钮
            replay_btn_rect = pygame.Rect(
                config["width"] // 2 - 100, config["height"] // 2 + 80, 200, 60
            )
            pygame.draw.rect(screen, (70, 130, 180), replay_btn_rect, border_radius=8)
            replay_text = font.render("REPLAY", True, (255, 255, 255))
            text_rect = replay_text.get_rect(center=replay_btn_rect.center)
            screen.blit(replay_text, text_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    config = load_config("game2048")
    main(config)
