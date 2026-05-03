'''
Used to record videos on the RAM agent playing
'''

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import torch
import imageio
from ram_ppo2 import ActorCritic, make_single_env, _find_nes_env


def record(checkpoint: str, output: str, n_episodes: int = 1, fps: int = 30):
    device = torch.device("cpu")

    env = make_single_env()
    #source of raw pixel frames
    nes_env = _find_nes_env(env)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = ActorCritic(obs_dim, n_actions).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    print(f"Loaded: {checkpoint}")

    frames = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward, max_x = 0.0, 0

        while not done:
            frames.append(nes_env.screen.copy())

            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(obs_t)
            action = logits.argmax(dim=-1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            max_x = max(max_x, info.get("x_pos", 0))
            done = terminated or truncated

        #capture final frame
        frames.append(nes_env.screen.copy())
        flag = info.get("flag_get", False)
        print(f"  ep {ep+1}: reward={ep_reward:.1f}  x_pos={max_x}  {'COMPLETED' if flag else 'died'}")

    env.close()

    print(f"Writing {len(frames)} frames → {output}")
    imageio.mimwrite(output, frames, fps=fps)
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="mario_ram_ppo_final_mario_ram_ppo_v2.pt")
    parser.add_argument("--output", default="mario_playthrough.mp4")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    record(args.checkpoint, args.output, args.episodes, args.fps)
