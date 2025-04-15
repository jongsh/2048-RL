import os
import pygame
import random
import pickle
from game.config import *

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'


class Game2048:
    def __init__(self, silent_mode=True):
        self.silent_mode = silent_mode
        self.reset()
        if not silent_mode:
            pygame.init()
            self._init_gui()
    

    def reset(self):
        self.grid = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
        self.score = 0
        self._add_random_tile()
        self._add_random_tile()
        return [row.copy() for row in self.grid]
        

    def step(self, action):
        direction_map = {0: "left", 1: "right", 2: "up", 3: "down"}
        direction = direction_map.get(action, None)
        
        if direction and self._move(direction):
            self._add_random_tile()
            if not self.silent_mode:
                self._render_grid()
        
        done = self._check_game_over()
        info = {
            'grid': self.grid,
            'score': self.score,
            'max_tile': max(max(row) for row in self.grid)
        }
        return done, info


    def _init_gui(self):
        self.font = pygame.font.Font(None, FONT_SIZE)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("2048")
        self.clock = pygame.time.Clock()
        self._render_grid()


    def _add_random_tile(self):
        candidates = [(i,j) for i in range(GRID_SIZE) 
                     for j in range(GRID_SIZE) if self.grid[i][j] == 0]
        if candidates:
            i,j = random.choice(candidates)
            randomness = random.random()
            cur_prob = 0
            for value, prob in NEW_TILE_VALUE.items():
                cur_prob += prob
                if randomness < cur_prob:
                    self.grid[i][j] = value
                    return


    def _process_row(self, row, reverse=False):
        filtered = [x for x in (reversed(row) if reverse else row) if x !=0]
        merged = []
        skip = False
        
        for i in range(len(filtered)):
            if skip: 
                skip = False
                continue
            if i+1 < len(filtered) and filtered[i] == filtered[i+1]:
                merged.append(filtered[i]*2)
                self.score += filtered[i]*2
                skip = True
            else:
                merged.append(filtered[i])

        padding = [0] * (GRID_SIZE - len(merged))
        return (merged + padding) if not reverse else padding + merged[::-1]


    def _move(self, direction):
        moved = False
        original = [row.copy() for row in self.grid]
        
        if direction in ("left", "right"):
            reverse = (direction == "right")
            for i in range(GRID_SIZE):
                processed = self._process_row(original[i], reverse)
                if processed != original[i]:
                    moved = True
                self.grid[i] = processed
                
        elif direction in ("up", "down"):
            reverse = (direction == "down")
            for j in range(GRID_SIZE):
                col = [self.grid[i][j] for i in range(GRID_SIZE)]
                processed = self._process_row(col, reverse)
                if processed != col:
                    moved = True
                for i in range(GRID_SIZE):
                    self.grid[i][j] = processed[i]
        return moved
    

    def _render_grid(self):
        self.screen.fill(BACKGROUND_COLOR)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                self._render_tile(i, j)
        pygame.display.flip()
        self.clock.tick(30)


    def _render_tile(self, i, j):
        tile_value = self.grid[i][j]
        tile_color = self._get_tile_color(tile_value)
        rect = pygame.Rect(j*TILE_SIZE + GRID_PADDING, 
                          i*TILE_SIZE + GRID_PADDING, 
                          TILE_SIZE - 2*GRID_PADDING, 
                          TILE_SIZE - 2*GRID_PADDING)
        pygame.draw.rect(self.screen, tile_color, rect, border_radius=3)

        if tile_value != 0:
            font_color = FONT_COLORS.get(tile_value, (255, 255, 255))
            text = self.font.render(str(tile_value), True, font_color)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
    

    def _get_tile_color(self, value):
        return TILE_COLORS.get(value, (60, 58, 50))


    def _check_game_over(self):
        if any(0 in row for row in self.grid):
            return False
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if (j+1 < GRID_SIZE and self.grid[i][j] == self.grid[i][j+1]) or \
                   (i+1 < GRID_SIZE and self.grid[i][j] == self.grid[i+1][j]):
                    return False
        return True


def main():
    # 2048 game setup
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2048")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 50)

    # game state
    game = None
    high_score = 0
    try:
        with open("highscore.dat", "rb") as f:
            high_score = pickle.load(f)
    except: pass

    # running main loop
    running = True
    game_over = False

    while running:
        screen.fill(BACKGROUND_COLOR)

        # 主界面按钮
        main_btn_rect = pygame.Rect(WIDTH//2-100, HEIGHT//2-35, 200, 60)
        pygame.draw.rect(screen, BTN_COLOR, main_btn_rect, border_radius=15)
        btn_text = font.render("START" if not game else "RESTART", True, (255, 255, 255))
        screen.blit(btn_text, (WIDTH//2-60, HEIGHT//2-20))

        # 显示高分
        score_text = font.render(f"High Score: {high_score}", True, FONT_COLOR)
        screen.blit(score_text, (20, 20))

        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                # 处理游戏结束时的按钮点击
                if game and game._check_game_over():
                    if restart_btn_rect.collidepoint(event.pos):
                        game = Game2048(silent_mode=False)
                        game._render_grid()
                        game_over = False
                
                # 处理主界面按钮点击
                elif main_btn_rect.collidepoint(event.pos):
                    game = Game2048(silent_mode=False)
                    game._render_grid()
                    game_over = False

            if game and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT: game.step(0)
                elif event.key == pygame.K_RIGHT: game.step(1)
                elif event.key == pygame.K_UP: game.step(2)
                elif event.key == pygame.K_DOWN: game.step(3)
                elif event.key == pygame.K_ESCAPE: game = None

        if game:
            # 只在游戏进行时更新画面
            if not game_over:
                game._render_grid()
                if game._check_game_over():
                    game_over = True
                    last_grid_surface = screen.copy()
                
            if game_over:
                # 游戏结束时使用缓存的最后一帧
                if last_grid_surface:
                    screen.blit(last_grid_surface, (0, 0))
                
                # 半透明遮罩层
                overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                overlay.fill((255, 255, 255, 128))
                screen.blit(overlay, (0,0))
                
                # 游戏结束信息
                text = font.render(f"Final Score: {game.score}", True, (255, 87, 87))
                text_rect = text.get_rect(center=(WIDTH//2, HEIGHT//2 - 40))
                screen.blit(text, text_rect)
                
                # 重新开始按钮
                restart_btn_rect = pygame.Rect(WIDTH//2-80, HEIGHT//2 + 20, 160, 50)
                pygame.draw.rect(screen, BTN_COLOR, restart_btn_rect, border_radius=8)
                restart_text = font.render("Retry", True, (255, 255, 255))
                screen.blit(restart_text, (WIDTH//2-40, HEIGHT//2 + 32))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()