'''
A comparison class for CNN and RAM based PPO.
'''

import warnings
warnings.filterwarnings("ignore")

import os, sys, time, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.distributions import Categorical
import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from core.core import SkipWrapper, GrayscaleResizeWrapper, FrameStackWrapper
from ramPPO.ram_ppo import ActorCritic as RamActorCritic, RamFeatureWrapper

from cnnPPO.ppo_agent_final import ActorCritic as CnnActorCritic


# ---------------------------------------------------------------------------
# Eval envs (no reward shaping)
# ---------------------------------------------------------------------------

def make_ram_eval_env():
    env = gym.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = RamFeatureWrapper(env)
    return env


def make_cnn_eval_env():
    env = gym.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipWrapper(env, skip=4)
    env = GrayscaleResizeWrapper(env)
    env = FrameStackWrapper(env, n=4)
    return env


def run_episodes(model, make_env_fn, n_episodes, greedy, device):
    results = []
    for _ in range(n_episodes):
        env = make_env_fn()
        obs, _ = env.reset()
        done, ep_reward, max_x, steps = False, 0.0, 0, 0
        inf_times = []

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits, _ = model(obs_t)
            inf_times.append(time.perf_counter() - t0)

            action = logits.argmax(-1).item() if greedy else Categorical(logits=logits).sample().item()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            max_x = max(max_x, info.get("x_pos", 0))
            done = terminated or truncated
            steps += 1

        results.append((ep_reward, max_x, info.get("flag_get", False), steps, inf_times))
        env.close()

    return results


def inference_benchmark(model, obs_shape, device, n=5000):
    x = torch.rand(1, *obs_shape, device=device)
    with torch.no_grad():
        for _ in range(200):
            model(x)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            model(x)
    elapsed = time.perf_counter() - t0
    return n / elapsed, elapsed / n * 1e6


def print_section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def summarise(label, results):
    rewards = [r[0] for r in results]
    x_pos   = [r[1] for r in results]
    flags   = [r[2] for r in results]
    steps   = [r[3] for r in results]
    inf_ms  = [t * 1e3 for r in results for t in r[4]]

    print(f"\n  {label}")
    print(f"    {'ep':>4}  {'reward':>8}  {'x_pos':>6}  {'steps':>6}  {'done':>5}")
    for i, (rew, x, flag, st, _) in enumerate(results):
        print(f"    {i+1:>4}  {rew:>8.1f}  {x:>6}  {st:>6}  {'YES' if flag else 'no':>5}")
    print(f"    {'─'*40}")
    print(f"    Mean reward      : {np.mean(rewards):8.1f}  ±{np.std(rewards):.1f}")
    print(f"    Max  reward      : {np.max(rewards):8.1f}")
    print(f"    Mean x_pos       : {np.mean(x_pos):8.1f}  (level end ~3100)")
    print(f"    Max  x_pos       : {np.max(x_pos):8.0f}")
    print(f"    Completion rate  : {100*np.mean(flags):8.1f}%")
    print(f"    Mean ep length   : {np.mean(steps):8.1f} steps  ±{np.std(steps):.1f}")
    print(f"    Inference latency: {np.mean(inf_ms):8.2f} ms/step")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cpu")

    ram_model = RamActorCritic(128, 7).to(device)
    ram_ckpt = torch.load(args.ram_checkpoint, map_location=device)
    ram_model.load_state_dict(ram_ckpt["model"] if "model" in ram_ckpt else ram_ckpt)
    ram_model.eval()

    run_cnn = bool(args.cnn_checkpoint)
    if run_cnn:
        cnn_model = CnnActorCritic(n_actions=7).to(device)
        cnn_ckpt = torch.load(args.cnn_checkpoint, map_location=device)
        cnn_model.load_state_dict(cnn_ckpt["model"] if "model" in cnn_ckpt else cnn_ckpt)
        cnn_model.eval()
    else:
        print("CNN PPO not available — running RAM PPO only.")

    print_section("Model Size")
    ram_params = sum(p.numel() for p in ram_model.parameters())
    ram_mb = os.path.getsize(args.ram_checkpoint) / 1e6
    print(f"  {'':20s}  {'RAM PPO':>12s}" + (f"  {'CNN PPO':>12s}" if run_cnn else ""))
    print(f"  {'Parameters':20s}  {ram_params:>12,}" + (f"  {sum(p.numel() for p in cnn_model.parameters()):>12,}" if run_cnn else ""))
    print(f"  {'Checkpoint (MB)':20s}  {ram_mb:>12.2f}" + (f"  {os.path.getsize(args.cnn_checkpoint)/1e6:>12.2f}" if run_cnn else ""))
    print(f"  {'Obs shape':20s}  {'(128,)':>12s}" + (f"  {'(4,84,84)':>12s}" if run_cnn else ""))

    print_section("Inference Speed (pure forward pass, 5000 calls)")
    ram_fps, ram_us = inference_benchmark(ram_model, (128,), device)
    print(f"  {'':20s}  {'RAM PPO':>12s}" + (f"  {'CNN PPO':>12s}" if run_cnn else ""))
    print(f"  {'Inferences/sec':20s}  {ram_fps:>12,.0f}", end="")
    if run_cnn:
        cnn_fps, cnn_us = inference_benchmark(cnn_model, (4, 84, 84), device)
        print(f"  {cnn_fps:>12,.0f}")
        print(f"  {'Latency (µs)':20s}  {ram_us:>12.1f}  {cnn_us:>12.1f}")
    else:
        print()
        print(f"  {'Latency (µs)':20s}  {ram_us:>12.1f}")

    n = args.episodes
    print_section(f"Playthrough Performance ({n} episodes, raw game rewards)")

    for label, greedy in [("Greedy (argmax)", True), ("Stochastic (sampled)", False)]:
        print(f"\n  [ {label} ]")
        ram_res = run_episodes(ram_model, make_ram_eval_env, n, greedy, device)
        summarise("RAM PPO  (128-dim RAM features)", ram_res)
        if run_cnn:
            cnn_res = run_episodes(cnn_model, make_cnn_eval_env, n, greedy, device)
            summarise("CNN PPO  (4x84x84 pixels, skip=4)", cnn_res)

    print(f"\n{'─'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ram-checkpoint", default=os.path.join(os.path.dirname(__file__), "..", "ramPPO", "ppo_final.pt"))
    parser.add_argument("--cnn-checkpoint", default=os.path.join(os.path.dirname(__file__), "..", "cnnPPO", "bestCNN.pt"))
    parser.add_argument("--episodes", type=int, default=10)
    main(parser.parse_args())
