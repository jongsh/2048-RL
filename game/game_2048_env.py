import math
import time
import gym
import random
import numpy as np
from game.game_2048 import Game2048
from game.config import *


class Game2048Env(gym.Env):
    def __init__(self, silent_mode=True):
        super(Game2048Env, self).__init__()
        self.game = Game2048(silent_mode=silent_mode)

        # 游戏状态
        self.info = self.game.reset()
        self.action_space = gym.spaces.Discrete(4)  # 0: left, 1: right, 2: up, 3: down
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=32,
            shape=(GRID_SIZE, GRID_SIZE),
            dtype=np.float32,
        )
        self.done = False

        # 奖励参数
        self.alpha = 0.3  # 空格奖励
        self.beta = 0.7  # 最大 tile 奖励
        self.gamma = 0.1  # 单调性奖励
        self.delta = 0.1  # 平滑性奖励
        self.zeta = 1  # 非法动作惩罚
        self.eta = 5  # 游戏结束惩罚

    def reset(self):
        """重置环境"""
        self.info = self.game.reset()
        self.done = False
        grid_array = np.array(self.game.grid, dtype=np.float32)
        grid_array[grid_array == 0] = 1
        return np.log2(grid_array)

    def step(self, action):
        """执行一步操作"""
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
        """渲染游戏界面"""
        self.game._render_grid()

    def _cal_reward(self, new_info, done):
        """计算奖励"""
        old_info = self.info
        new_grid = np.array(new_info["grid"], dtype=np.int32)

        # 合并奖励
        merge_score = new_info["score"] - old_info["score"]
        merge_reward = math.log2(merge_score + 1)

        # 空格比例奖励
        num_empty = np.sum(new_grid == 0)
        empty_reward = self.alpha * (num_empty / (GRID_SIZE * GRID_SIZE))

        # 最大 tile 奖励
        old_max, new_max = old_info["max_tile"], new_info["max_tile"]
        max_reward = (
            self.beta * max(0, math.log2(new_max) - math.log2(old_max))
            if new_max > old_max
            else 0
        )

        # 单调性奖励
        mono_reward = self.gamma * self._monotonicity(new_grid)

        # 平滑性奖励
        smooth_reward = self.delta * self._smoothness(new_grid)

        # 非法动作惩罚
        invalid_reward = -self.zeta if not new_info.get("moved", True) else 0

        # 游戏结束惩罚
        done_reward = -self.eta if done else 0

        return (
            merge_reward
            + empty_reward
            + max_reward
            + mono_reward
            + smooth_reward
            + invalid_reward
            + done_reward
        )

    def _monotonicity(self, grid):
        """计算单调性"""
        mono_score = 0
        for row in grid:
            mono_score += sum(max(0, row[i] - row[i + 1]) for i in range(len(row) - 1))
        for col in grid.T:
            mono_score += sum(max(0, col[i] - col[i + 1]) for i in range(len(col) - 1))
        max_possible = np.max(grid) * (GRID_SIZE - 1) * 2 + 1e-5
        return 1.0 - mono_score / max_possible

    def _smoothness(self, grid):
        """计算平滑性"""
        smooth_score = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                val = grid[i][j]
                for dx, dy in [(1, 0), (0, 1)]:
                    ni, nj = i + dx, j + dy
                    if ni < GRID_SIZE and nj < GRID_SIZE:
                        smooth_score += abs(val - grid[ni][nj])
        max_diff = np.max(grid) * 2 * (GRID_SIZE**2) + 1e-5
        return 1.0 - smooth_score / max_diff


if __name__ == "__main__":
    env = Game2048Env(silent_mode=False)
    env.reset()
    env.render()
    actions = ["left", "right", "up", "down"]
    while True:
        time.sleep(0.8)  # 控制渲染速度
        action = random.randint(0, 3)  # 随机选择一个动作
        obs, reward, done, info = env.step(action)
        env.render()  # 渲染游戏界面
        print(f"Action: {actions[action]}, Reward: {reward}, Done: {done}")
        if done:
            print("Game Over!\n\nFinal Info:", info)
            break
