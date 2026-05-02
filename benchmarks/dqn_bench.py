'''
eval_dqn.py
Evaluate the CNN DQN checkpoint and compare against RAM PPO and CNN PPO.
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
from dqn.dqn_agent import MarioQNetwork


# ---------------------------------------------------------------------------
# Eval envs
# ---------------------------------------------------------------------------

def make_cnn_eval_env():
    env = gym.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipWrapper(env, skip=4)
    env = GrayscaleResizeWrapper(env)
    env = FrameStackWrapper(env, n=4)
    return env


def make_ram_eval_env():
    env = gym.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = RamFeatureWrapper(env)
    return env


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def run_episodes_ppo(model, make_env_fn, n_episodes, greedy, device):
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


def run_episodes_dqn(model, make_env_fn, n_episodes, epsilon, device):
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
                q_values = model(obs_t)
            inf_times.append(time.perf_counter() - t0)
            if epsilon > 0 and np.random.random() < epsilon:
                action = np.random.randint(q_values.shape[-1])
            else:
                action = q_values.argmax(-1).item()
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

    dqn_ckpt = torch.load(args.dqn_checkpoint, map_location=device, weights_only=False)
    dqn_model = MarioQNetwork(obs_shape=(4, 84, 84), num_actions=7).to(device)
    dqn_model.load_state_dict(dqn_ckpt["q_network"])
    dqn_model.eval()
    dqn_steps = dqn_ckpt.get("num_timesteps", "?")
    trained_eps = dqn_ckpt.get("hyperparams", {}).get("exploration_final_eps", 0.1)

    ram_model = RamActorCritic(128, 7).to(device)
    ram_ckpt = torch.load(args.ram_checkpoint, map_location=device, weights_only=False)
    ram_model.load_state_dict(ram_ckpt["model"] if "model" in ram_ckpt else ram_ckpt)
    ram_model.eval()

    cnn_model = CnnActorCritic(n_actions=7).to(device)
    cnn_ckpt = torch.load(args.cnn_checkpoint, map_location=device, weights_only=False)
    cnn_model.load_state_dict(cnn_ckpt["model"] if "model" in cnn_ckpt else cnn_ckpt)
    cnn_model.eval()

    print_section("Model Size")
    dqn_params = sum(p.numel() for p in dqn_model.parameters())
    ram_params = sum(p.numel() for p in ram_model.parameters())
    cnn_params = sum(p.numel() for p in cnn_model.parameters())
    print(f"  {'':25s}  {'DQN':>10s}  {'RAM PPO':>10s}  {'CNN PPO':>10s}")
    print(f"  {'Training steps':25s}  {dqn_steps:>10,}  {'~70M':>10s}  {'~30M':>10s}")
    print(f"  {'Parameters':25s}  {dqn_params:>10,}  {ram_params:>10,}  {cnn_params:>10,}")
    print(f"  {'Checkpoint (MB)':25s}  {os.path.getsize(args.dqn_checkpoint)/1e6:>10.2f}  {os.path.getsize(args.ram_checkpoint)/1e6:>10.2f}  {os.path.getsize(args.cnn_checkpoint)/1e6:>10.2f}")
    print(f"  {'Obs shape':25s}  {'(4,84,84)':>10s}  {'(128,)':>10s}  {'(4,84,84)':>10s}")

    print_section("Inference Speed (5000 calls, CPU)")
    dqn_fps, dqn_us = inference_benchmark(dqn_model, (4, 84, 84), device)
    ram_fps, ram_us = inference_benchmark(ram_model, (128,), device)
    cnn_fps, cnn_us = inference_benchmark(cnn_model, (4, 84, 84), device)
    print(f"  {'':25s}  {'DQN':>10s}  {'RAM PPO':>10s}  {'CNN PPO':>10s}")
    print(f"  {'Inferences/sec':25s}  {dqn_fps:>10,.0f}  {ram_fps:>10,.0f}  {cnn_fps:>10,.0f}")
    print(f"  {'Latency (µs)':25s}  {dqn_us:>10.1f}  {ram_us:>10.1f}  {cnn_us:>10.1f}")

    n = args.episodes
    print_section(f"Playthrough Performance ({n} episodes, raw game rewards)")

    print(f"\n  [ Greedy / epsilon=0 ]")
    dqn_res = run_episodes_dqn(dqn_model, make_cnn_eval_env, n, epsilon=0.0, device=device)
    ram_res = run_episodes_ppo(ram_model, make_ram_eval_env, n, greedy=True, device=device)
    cnn_res = run_episodes_ppo(cnn_model, make_cnn_eval_env, n, greedy=True, device=device)
    summarise(f"DQN  ({dqn_steps:,} steps, ε=0)", dqn_res)
    summarise("RAM PPO  (128-dim RAM features)", ram_res)
    summarise("CNN PPO  (4×84×84 pixels, skip=4)", cnn_res)

    print(f"\n  [ Stochastic / epsilon={trained_eps} (as trained) ]")
    dqn_res_s = run_episodes_dqn(dqn_model, make_cnn_eval_env, n, epsilon=trained_eps, device=device)
    ram_res_s = run_episodes_ppo(ram_model, make_ram_eval_env, n, greedy=False, device=device)
    cnn_res_s = run_episodes_ppo(cnn_model, make_cnn_eval_env, n, greedy=False, device=device)
    summarise(f"DQN  ({dqn_steps:,} steps, ε={trained_eps})", dqn_res_s)
    summarise("RAM PPO  (128-dim RAM features, sampled)", ram_res_s)
    summarise("CNN PPO  (4×84×84 pixels, sampled)", cnn_res_s)

    print(f"\n{'─'*60}\n")


if __name__ == "__main__":
    _here = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn-checkpoint", default=os.path.join(_here, "..", "dqn", "mario_dqn_step1150000.pt"))
    parser.add_argument("--ram-checkpoint", default=os.path.join(_here, "..", "ramPPO", "ppo_final.pt"))
    parser.add_argument("--cnn-checkpoint", default=os.path.join(_here, "..", "cnnPPO", "bestCNN.pt"))
    parser.add_argument("--episodes", type=int, default=10)
    main(parser.parse_args())
