import json
import pygame
import os
import argparse
import math
import numpy as np

from copy import deepcopy
from datetime import datetime

from envs.game2048_env import Game2048Env
from configs.config import Configuration
from agents.dqn_agent import DQNAgent
from agents.imitation_agent import ImitationAgent


def _read_2048_data(file, create=False):
    """read data from json file, if not exist and create=True, create a new file"""
    if os.path.exists(file) and os.path.getsize(file) > 0:
        with open(file, "r", encoding="utf-8") as f:
            try:
                episode_data = json.load(f)
            except json.JSONDecodeError:
                print("[⚠] JSON 文件损坏或为空，重新创建。")
                episode_data = None
    else:
        episode_data = None

    if episode_data is None and create:
        episode_data = {
            "metadata": {
                "env_name": "Game2048Env",
                "created_at": datetime.now().isoformat(),
                "total_episodes": 0,
                "total_steps": 0,
            },
            "episodes": [],
        }
    return episode_data


def _custom_save_json(episode_data, save_file):
    # Custom function to save JSON data with specific formatting
    metadata = episode_data["metadata"]
    episodes = episode_data["episodes"]

    with open(save_file, "w", encoding="utf-8") as f:
        f.write("{\n")
        f.write('  "metadata": {\n')
        for i, (key, value) in enumerate(metadata.items()):
            comma = "," if i < len(metadata) - 1 else ""
            if isinstance(value, str):
                f.write(f'    "{key}": "{value}"{comma}\n')
            else:
                f.write(f'    "{key}": {value}{comma}\n')
        f.write("  },\n")
        f.write('  "episodes": [\n')

        for ei, ep in enumerate(episodes):
            f.write("    {\n")
            f.write(f'      "episode_id": {ep["episode_id"]},\n')
            f.write(f'      "total_steps": {ep["total_steps"]},\n')
            f.write('      "steps": [\n')
            for si, step in enumerate(ep["steps"]):
                step_str = json.dumps(step, ensure_ascii=False, separators=(", ", ": "))
                comma = "," if si < len(ep["steps"]) - 1 else ""
                f.write(f"        {step_str}{comma}\n")

            f.write("      ]\n")
            f.write("    }")
            if ei < len(episodes) - 1:
                f.write(",\n")
            else:
                f.write("\n")

        f.write("  ]\n}\n")


def read_preprocess_2048_data(save_file, threshold_steps=450, shuffle=False):
    """
    Read data from a specified file.
    The returned data only contains episodes with total steps >= threshold_steps.
    Return a list of steps:
        [{"state":..., "action_mask":..., "action":..., "reward":..., "next_state":..., "next_action_mask":..., "done":...}, ...]
     If shuffle=True, the order of steps will be shuffled.
    """
    data = _read_2048_data(save_file, create=False)
    buffer = []
    for ep in data["episodes"]:
        if ep["total_steps"] < threshold_steps:
            continue
        for step in ep["steps"]:
            buffer.append(step)
    if shuffle:
        np.random.shuffle(buffer)
    return buffer


def collect_huamn_data(config, save_file="data/human_2048.json"):
    """Collect human gameplay data for imitation learning"""

    env = Game2048Env(config=config, silent_mode=False)
    config = env.env_config
    screen = env.game.screen
    clock = env.game.clock
    font = env.game.font

    episode_data = _read_2048_data(save_file, create=True)

    total_steps = episode_data["metadata"].get("total_steps", 0)
    total_episodes = episode_data["metadata"].get("total_episodes", 0)

    # main loop
    state = "menu"
    running = True
    current_episode = None
    obs, info = None, None
    episode_saved = False

    while running:
        screen.fill(config["style"]["background_color"])

        # event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                # start game
                if state == "menu":
                    start_btn = pygame.Rect(
                        config["style"]["width"] // 2 - 100,
                        config["style"]["height"] // 2 - 40,
                        200,
                        60,
                    )
                    if start_btn.collidepoint(mouse_pos):
                        obs, info = env.reset()
                        current_episode = {
                            "episode_id": total_episodes + 1,
                            "total_steps": 0,
                            "steps": [],
                        }
                        state = "playing"

                # game over
                elif state == "game_over":
                    # SAVE button
                    if not episode_saved:
                        save_btn = pygame.Rect(
                            config["style"]["width"] // 2 - 100,
                            config["style"]["height"] // 2 - 80,
                            200,
                            60,
                        )
                        if save_btn.collidepoint(mouse_pos):
                            episode_data["episodes"].append(current_episode)
                            total_episodes += 1
                            episode_data["metadata"]["total_episodes"] = total_episodes
                            episode_data["metadata"]["total_steps"] = total_steps
                            episode_data["metadata"]["updated_at"] = datetime.now().isoformat()
                            os.makedirs(os.path.dirname(save_file), exist_ok=True)
                            _custom_save_json(episode_data, save_file)
                            print(f"[✔] Saved episode #{current_episode['episode_id']} (total steps: {total_steps})")
                            episode_saved = True
                    # RETRY button
                    retry_btn = pygame.Rect(
                        config["style"]["width"] // 2 - 100,
                        config["style"]["height"] // 2,
                        200,
                        60,
                    )
                    if retry_btn.collidepoint(mouse_pos):
                        obs, info = env.reset()
                        current_episode = {
                            "episode_id": total_episodes + 1,
                            "total_steps": 0,
                            "steps": [],
                        }
                        episode_saved = False
                        state = "playing"

            elif event.type == pygame.KEYDOWN and state == "playing":
                action = None
                if event.key == pygame.K_LEFT:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_UP:
                    action = 2
                elif event.key == pygame.K_DOWN:
                    action = 3

                if action is not None:
                    state_before = obs.tolist()
                    action_mask = info["action_mask"]

                    obs_next, reward, done, _, new_info = env.step(action)
                    if new_info["moved"] is False:
                        continue  # invalid move, skip logging

                    step_data = {
                        "state": state_before,
                        "action_mask": action_mask,
                        "action": action,
                        "reward": float(reward),
                        "next_state": obs_next.tolist(),
                        "next_action_mask": new_info["action_mask"],
                        "done": done,
                    }
                    current_episode["steps"].append(step_data)
                    current_episode["total_steps"] += 1
                    total_steps += 1

                    obs, info = obs_next, new_info

                    if done:
                        state = "game_over"

        # rendering
        if state == "menu":
            btn_rect = pygame.Rect(config["style"]["width"] // 2 - 100, config["style"]["height"] // 2 - 40, 200, 60)
            pygame.draw.rect(screen, config["style"]["btn_color"], btn_rect, border_radius=15)
            btn_text = font.render("START", True, (255, 255, 255))
            screen.blit(btn_text, btn_text.get_rect(center=btn_rect.center))

        elif state == "playing":
            env.render()

        elif state == "game_over":
            overlay = pygame.Surface((config["style"]["width"], config["style"]["height"]), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 150))
            screen.blit(overlay, (0, 0))

            text = font.render("GAME OVER", True, (255, 0, 0))
            screen.blit(
                text, text.get_rect(center=(config["style"]["width"] // 2, config["style"]["height"] // 2 - 120))
            )

            # display SAVE button if not saved yet
            if not episode_saved:
                save_rect = pygame.Rect(
                    config["style"]["width"] // 2 - 100, config["style"]["height"] // 2 - 50, 200, 60
                )
                pygame.draw.rect(screen, (0, 195, 35), save_rect, border_radius=15)
                save_text = font.render("SAVE", True, (255, 255, 255))
                screen.blit(save_text, save_text.get_rect(center=save_rect.center))

            # display RETRY button
            retry_rect = pygame.Rect(config["style"]["width"] // 2 - 100, config["style"]["height"] // 2 + 30, 200, 60)
            pygame.draw.rect(screen, (70, 130, 180), retry_rect, border_radius=15)
            retry_text = font.render("RETRY", True, (255, 255, 255))
            screen.blit(retry_text, retry_text.get_rect(center=retry_rect.center))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def collect_model_data(config, save_file="data/agent_2048.json", threshold=600, k=1000):
    """Collect gameplay data using a trained model for imitation learning"""
    public_config = config["public"]
    checkpoint_dir = public_config["from_checkpoint"]
    assert checkpoint_dir and os.path.exists(checkpoint_dir), f"Checkpoint path {checkpoint_dir} does not exist!"

    agent = (
        DQNAgent()
        if public_config["agent"] == "dqn"
        else ImitationAgent() if public_config["agent"] == "imitation" else None
    )
    assert agent is not None, f"Unsupported agent type: {public_config['agent']}"
    agent.load(checkpoint_dir)
    env = Game2048Env(config=config, silent_mode=True)

    # read existing data if available
    episode_data = _read_2048_data(save_file, create=True)
    total_steps = episode_data["metadata"].get("total_steps", 0)
    total_episodes = episode_data["metadata"].get("total_episodes", 0)

    # main loop
    for _ in range(k):
        obs, info = env.reset()
        current_episode = {
            "episode_id": total_episodes + 1,
            "total_steps": 0,
            "steps": [],
        }
        while True:
            state_before = obs.tolist()
            action_mask = info["action_mask"]
            action = agent.select_action(obs, action_mask=action_mask, method="greedy")
            obs, reward, done, _, info = env.step(action)
            state_after = obs.tolist()
            step_data = {
                "state": state_before,
                "action_mask": action_mask,
                "action": action,
                "reward": float(reward),
                "next_state": state_after,
                "next_action_mask": info["action_mask"],
                "done": done,
            }
            current_episode["steps"].append(step_data)
            current_episode["total_steps"] += 1
            if done:
                break

        if current_episode["total_steps"] >= threshold:
            total_episodes += 1
            total_steps += current_episode["total_steps"]
            episode_data["episodes"].append(current_episode)
            episode_data["metadata"]["total_episodes"] = total_episodes
            episode_data["metadata"]["total_steps"] = total_steps
            episode_data["metadata"]["updated_at"] = datetime.now().isoformat()
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            _custom_save_json(episode_data, save_file)
            print(f"[✔] Saved episode #{current_episode['episode_id']} (steps: {current_episode['total_steps']})")
        else:
            print(f"[✕] Discarded episode #{current_episode['episode_id']} (steps: {current_episode['total_steps']})")


def clean_data(save_file="data/human_2048.json", threshold_steps=500):
    """Clean collected data by removing short episodes, update new rewards"""

    def _cal_new_reward(old_grid, new_grid, direction, done):
        # calculate new reward based on the old and new grid states
        old_grid = np.array(old_grid, dtype=np.int32)
        old_grid = np.where(old_grid == 0, 0, 2**old_grid)
        new_grid = np.array(new_grid, dtype=np.int32)
        new_grid = np.where(new_grid == 0, 0, 2**new_grid)
        grid_size = old_grid.shape[0]
        direction = ["left", "right", "up", "down"][direction]

        def process_row(row, reverse=False):
            filtered = [x for x in (reversed(row) if reverse else row) if x != 0]
            merged = []
            score = 0
            skip = False
            for i in range(len(filtered)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(filtered) and filtered[i] == filtered[i + 1]:
                    merged.append(filtered[i] * 2)
                    score += filtered[i] * 2
                    skip = True
                else:
                    merged.append(filtered[i])
            padding = [0] * (grid_size - len(merged))
            merged_row = (merged + padding) if not reverse else padding + merged[::-1]
            return merged_row, score

        def move(grid, direction):
            total_score = 0
            moved_grid = np.zeros_like(grid)
            if direction == "left":
                for i in range(grid_size):
                    row, s = process_row(grid[i, :], reverse=False)
                    moved_grid[i, :] = row
                    total_score += s
            elif direction == "right":
                for i in range(grid_size):
                    row, s = process_row(grid[i, :], reverse=True)
                    moved_grid[i, :] = row
                    total_score += s
            elif direction == "up":
                for j in range(grid_size):
                    col, s = process_row(grid[:, j], reverse=False)
                    moved_grid[:, j] = col
                    total_score += s
            elif direction == "down":
                for j in range(grid_size):
                    col, s = process_row(grid[:, j], reverse=True)
                    moved_grid[:, j] = col
                    total_score += s
            return total_score

        score_gain = move(deepcopy(old_grid), direction)

        # merge reward
        merge_reward = 0.0
        if score_gain > 0:
            merge_reward += math.log2(score_gain + 1) * 0.5

        # space reward
        empty_before = np.sum(old_grid == 0)
        empty_after = np.sum(new_grid == 0)
        space_reward = float((empty_after - empty_before) * 0.05)

        # invalid reward
        invalid_reward = -1.0 if np.array_equal(old_grid, new_grid) else 0.0

        # max tile reward
        max_tile_reward = 0.0
        old_max_tile = np.max(old_grid)
        new_max_tile = np.max(new_grid)
        if new_max_tile > old_max_tile:
            max_tile_reward += math.sqrt(new_max_tile) * 0.5

        # normalnize_factor = 100.0
        total_reward = merge_reward + space_reward + invalid_reward + max_tile_reward
        return total_reward

    data = _read_2048_data(save_file, create=False)
    if data is None:
        print(f"[X] Data file {save_file} does not exist or is empty.")
        return

    # metadata
    new_data = {
        "metadata": data["metadata"],
        "episodes": [],
    }
    new_data["metadata"]["updated_at"] = datetime.now().isoformat()
    total_episodes = 0
    total_steps = 0

    # episode data
    for ep in data["episodes"]:
        if ep["total_steps"] < threshold_steps:
            continue
        total_episodes += 1
        total_steps += ep["total_steps"]
        episode_data = {
            "episode_id": total_episodes,
            "total_steps": ep["total_steps"],
            "steps": [],
        }
        for step in ep["steps"]:
            new_reward = _cal_new_reward(step["state"], step["next_state"], step["action"], step["done"])
            new_step = deepcopy(step)
            new_step["reward"] = float(new_reward)
            episode_data["steps"].append(new_step)
        new_data["episodes"].append(episode_data)

    # update metadata
    new_data["metadata"]["total_episodes"] = total_episodes
    new_data["metadata"]["total_steps"] = total_steps

    # save cleaned data
    _custom_save_json(new_data, save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect data for 2048 RL")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--file", type=str, required=False, help="Data file path")
    parser.add_argument("--mode", type=str, choices=["human", "model", "clean"], default="human", help="Data ")

    args = parser.parse_args()
    config = Configuration(config_path=args.config)

    file = args.file
    if not args.file:
        if args.mode == "human":
            file = "data/human_2048.json"
        elif args.mode == "model":
            file = "data/agent_2048.json"
        elif args.mode == "clean":
            file = "data/human_2048.json"

    if args.mode == "human":
        collect_huamn_data(config=config, save_file=file)
    elif args.mode == "model":
        collect_model_data(config=config, save_file=file)
    elif args.mode == "clean":
        clean_data(save_file=file, threshold_steps=500)
