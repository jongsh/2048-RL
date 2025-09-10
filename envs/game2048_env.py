# 2048 game environment for reinforcement learning
import math
import time
import gymnasium as gym
import pygame
import random
import numpy as np
from envs.game2048 import Game2048
from configs.config import Configuration


class Game2048Env(gym.Env):
    def __init__(self, config: Configuration = Configuration(), silent_mode=True):
        super(Game2048Env, self).__init__()

        self.game = Game2048(config=config, silent_mode=silent_mode)
        self.env_config = self.game.config

        # game state
        self.info = self.game.reset()
        self.action_space = gym.spaces.Discrete(4)  # 0: left, 1: right, 2: up, 3: down
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=32,
            shape=(self.env_config["grid_size"], self.env_config["grid_size"]),
            dtype=np.float32,
        )
        self.done = False

        # reward parameters
        self.alpha = 0.3  # space reward
        self.beta = 0.7  # max tile reward
        self.gamma = 0.1  # monotonicity reward
        self.delta = 0.1  # smoothness reward
        self.zeta = 1  # invalid action penalty
        self.eta = 5  # game over penalty

    def reset(self):
        """reset the game environment"""
        self.info = self.game.reset()
        self.done = False
        grid_array = np.array(self.game.grid, dtype=np.float32)
        grid_array[grid_array == 0] = 1
        return np.log2(grid_array)

    def step(self, action):
        """take a step in the game environment"""
        if self.done:
            raise ValueError("Episode has ended. Please reset the environment.")

        done, new_info = self.game.step(action)
        self.done = done
        reward = self._cal_reward(new_info, done)
        self.info = new_info

        grid_array = np.array(self.game.grid, dtype=np.float32)
        grid_array[grid_array == 0] = 1
        obs = np.log2(grid_array)
        return obs, reward, done, new_info

    def render(self):
        """render the game environment"""
        screen = self.game.screen
        screen.fill(self.env_config["style"]["background_color"])
        self.game._render_grid()
        pygame.display.flip()

    def _cal_reward(self, new_info, done):
        """calculate the reward based on the new game state"""
        old_info = self.info
        new_grid = np.array(new_info["grid"], dtype=np.int32)
        old_grid = np.array(old_info["grid"], dtype=np.int32)

        # 1. 合并奖励：只要有合并，就奖励 +1（不看 tile 大小）
        merge_reward = 0
        score_gain = new_info["score"] - old_info["score"]
        if score_gain > 0:
            # 每次合并动作按 log2(score_gain) 估计合并次数（大概估一下）
            merge_count = int(math.log2(score_gain + 1))
            merge_reward = merge_count * 1.0

        # 2. 空格奖励：保持盘面更宽松
        empty_before = np.sum(old_grid == 0)
        empty_after = np.sum(new_grid == 0)
        space_reward = (empty_after - empty_before) * 0.1

        # 3. 游戏结束惩罚
        done_reward = -5.0 if done else 0

        return merge_reward + space_reward + done_reward

    def _monotonicity(self, grid):
        """计算单调性"""
        mono_score = 0
        for row in grid:
            mono_score += sum(max(0, row[i] - row[i + 1]) for i in range(len(row) - 1))
        for col in grid.T:
            mono_score += sum(max(0, col[i] - col[i + 1]) for i in range(len(col) - 1))
        max_possible = np.max(grid) * (self.env_config["grid_size"] - 1) * 2 + 1e-5
        return 1.0 - mono_score / max_possible

    def _smoothness(self, grid):
        """计算平滑性"""
        smooth_score = 0
        for i in range(self.env_config["grid_size"]):
            for j in range(self.env_config["grid_size"]):
                val = grid[i][j]
                for dx, dy in [(1, 0), (0, 1)]:
                    ni, nj = i + dx, j + dy
                    if ni < self.env_config["grid_size"] and nj < self.env_config["grid_size"]:
                        smooth_score += abs(val - grid[ni][nj])
        max_diff = np.max(grid) * 2 * (self.env_config["grid_size"] ** 2) + 1e-5
        return 1.0 - smooth_score / max_diff


if __name__ == "__main__":
    config = Configuration()
    env = Game2048Env(config=config, silent_mode=False)
    env.reset()
    env.render()

    actions = ["left", "right", "up", "down"]
    running = True
    while running:
        time.sleep(1.5)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = random.randint(0, 3)
        obs, reward, done, info = env.step(action)
        print(f"Action: {actions[action]}, Reward: {reward}, Done: {done}")
        env.render()

        if done:
            print("Game Over!\n\nFinal Info:", info)
            break
