'''
Used to record videos of the CNN PPO agent playing.
'''

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys

#this was tweaked slightly using Claude to get it to run with the new repo
#config
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_script_dir = os.path.abspath(os.path.dirname(__file__))
while _script_dir in sys.path:
    sys.path.remove(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
#end

import numpy as np
import torch
import imageio
from collections import deque
from PIL import Image
import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from torch.distributions import Categorical

from cnnPPO.ppo_agent_final import ActorCritic


def _find_nes_env(env):
    e = env
    while e is not None:
        if hasattr(e, "ram"):
            return e
        e = getattr(e, "env", None)
    raise AttributeError("No NES env found")


class RecordingSkipWrapper(gym.Wrapper):
    '''
    Frame-skip wrapper that also captures every raw frame for smooth video.
    '''
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._nes_env = _find_nes_env(env)
        self.recorded_frames = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.recorded_frames.append(self._nes_env.screen.copy())
        return obs, info

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            self.recorded_frames.append(self._nes_env.screen.copy())
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class GrayscaleResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(84, 84), dtype=np.float32
        )

    def observation(self, obs):
        img = Image.fromarray(obs).convert("L").resize((84, 84), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, n=4):
        super().__init__(env)
        self._n = n
        self._frames = deque(maxlen=n)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(n, 84, 84), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self._n):
            self._frames.append(obs)
        return np.stack(self._frames), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return np.stack(self._frames), reward, terminated, truncated, info


def make_recording_env():
    env = gym.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = RecordingSkipWrapper(env, skip=4)
    env = GrayscaleResizeWrapper(env)
    env = FrameStackWrapper(env, n=4)
    return env


def record(checkpoint, output, greedy, n_episodes=1, fps=60):
    device = torch.device("cpu")

    model = ActorCritic(n_actions=7).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    print(f"Loaded: {checkpoint}  ({'greedy' if greedy else 'stochastic'})")

    env = make_recording_env()
    skip_wrapper = env.env.env

    for ep in range(n_episodes):
        skip_wrapper.recorded_frames = []
        obs, _ = env.reset()
        done, ep_reward, max_x = False, 0.0, 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(obs_t)
            if greedy:
                action = logits.argmax(-1).item()
            else:
                action = Categorical(logits=logits).sample().item()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            max_x = max(max_x, info.get("x_pos", 0))
            done = terminated or truncated

        flag = info.get("flag_get", False)
        print(f"  ep {ep+1}: reward={ep_reward:.1f}  x_pos={max_x}  {'COMPLETED' if flag else 'died'}")

    env.close()

    frames = skip_wrapper.recorded_frames
    print(f"Writing {len(frames)} frames → {output}")
    imageio.mimwrite(output, frames, fps=fps)
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="mario_cnn_ppo_final.pt")
    parser.add_argument("--greedy-output", default="cnn_ppo_greedy.mp4")
    parser.add_argument("--stochastic-output", default="cnn_ppo_stochastic.mp4")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()

    record(args.checkpoint, args.greedy_output,   greedy=True,  n_episodes=args.episodes, fps=args.fps)
    record(args.checkpoint, args.stochastic_output, greedy=False, n_episodes=args.episodes, fps=args.fps)
