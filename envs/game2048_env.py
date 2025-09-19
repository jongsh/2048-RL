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
            low=0,
            high=np.inf,
            shape=(self.env_config["grid_size"], self.env_config["grid_size"]),
            dtype=np.int32,
        )
        self.done = False

    def reset(self):
        """reset the game environment"""
        self.info = self.game.reset()
        self.done = False
        grid_array = np.array(self.game.grid, dtype=np.float32)
        obs = np.log2(np.where(grid_array == 0, 1, grid_array)).astype(np.int32)
        action_mask = self._action_mask(self.info["grid"])
        self.info["action_mask"] = action_mask
        return obs, self.info

    def step(self, action):
        """take a step in the game environment"""
        if self.done:
            raise ValueError("Episode has ended. Please reset the environment.")

        done, new_info = self.game.step(action)
        self.done = done

        action_mask = self._action_mask(new_info["grid"])
        reward, reward_info = self._cal_reward(new_info, done)
        new_info["reward_info"] = reward_info
        new_info["action_mask"] = action_mask
        self.info = new_info

        grid_array = np.array(self.game.grid, dtype=np.int32)
        obs = np.log2(np.where(grid_array == 0, 1, grid_array)).astype(np.int32)
        return obs, reward, done, False, new_info  # gymnasium: (obs, reward, done, truncated, info)

    def render(self):
        """render the game environment"""
        screen = self.game.screen
        screen.fill(self.env_config["style"]["background_color"])
        self.game._render_grid()
        pygame.display.flip()

    def _action_mask(self, grid):
        """generate action mask for valid moves"""
        grid = np.array(grid)
        mask = [0, 0, 0, 0]  # [left, right, up, down]

        # left
        shift = np.zeros_like(grid)
        shift[:, 1:] = grid[:, :-1]
        cond_left = (grid != 0) & ((shift == 0) | (shift == grid))
        if np.any(cond_left[:, 1:]):
            mask[0] = 1
        # right
        shift = np.zeros_like(grid)
        shift[:, :-1] = grid[:, 1:]
        cond_right = (grid != 0) & ((shift == 0) | (shift == grid))
        if np.any(cond_right[:, :-1]):
            mask[1] = 1
        # up
        shift = np.zeros_like(grid)
        shift[1:, :] = grid[:-1, :]
        cond_up = (grid != 0) & ((shift == 0) | (shift == grid))
        if np.any(cond_up[1:, :]):
            mask[2] = 1
        # down
        shift = np.zeros_like(grid)
        shift[:-1, :] = grid[1:, :]
        cond_down = (grid != 0) & ((shift == 0) | (shift == grid))
        if np.any(cond_down[:-1, :]):
            mask[3] = 1

        return mask

    def _cal_reward(self, new_info, done):
        """calculate the reward based on the new game state"""
        old_info = self.info
        new_grid = np.array(new_info["grid"], dtype=np.int32)
        old_grid = np.array(old_info["grid"], dtype=np.int32)

        # 1. merge reward
        merge_reward = 0
        score_gain = new_info["score"] - old_info["score"]
        if score_gain > 0:
            merge_reward += math.log2(score_gain + 1) * 0.1
        # 2. space reward
        empty_before = np.sum(old_grid == 0)
        empty_after = np.sum(new_grid == 0)
        space_reward = float((empty_after - empty_before) * 0.05)
        # 3. penalize invalid action
        invalid_reward = -1.0 if new_info["moved"] == False else 0.0
        # 4. game over reward
        if done:
            done_reward = math.log2(new_info["max_tile"]) ** 2 - 100
        else:
            done_reward = 0

        # info
        info = {
            "merge_reward": merge_reward,
            "space_reward": space_reward,
            "invalid_reward": invalid_reward,
            "done_reward": done_reward,
        }

        return merge_reward + space_reward + invalid_reward + done_reward, info

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
        print("\n" + "=" * 20)
        time.sleep(1.5)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = random.randint(0, 3)
        print(f"\nTaking action: {actions[action]}")
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print("Observation:\n", obs)
        print("Info:\n", info)
        print(f"Action: {actions[action]}, Reward: {reward}, Done: {done}")
        env.render()

        if done:
            print("Game Over!\n\nFinal Info:", info)
            break
