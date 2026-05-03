'''
eval_dqn.py
Evaluate the CNN DQN checkpoint and compare against RAM PPO and CNN PPO.
Modified version of our benchmarks script to deal with the annoyingness of
loading the DQN in.

Basically, DQN is different enough checkpoint-wise and action-selection-wise that it was
less of a pain in the ass to give it its own script than to Frankenstein benchmarks.py.
'''

import warnings
warnings.filterwarnings("ignore")

import os, sys, time, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

#also lmfao
from core.core import (
    inference_benchmark,
    make_cnn_eval_env,
    make_ram_eval_env,
    print_section,
    run_episodes_ppo,
    summarise,
)
from ramPPO.ram_ppo import ActorCritic as RamActorCritic
from cnnPPO.ppo_agent_final import ActorCritic as CnnActorCritic
from dqn.dqn_agent import MarioQNetwork


def run_episodes_dqn(model, make_env_fn, n_episodes, epsilon, device):
    '''
    Run DQN episodes. Given the high training rewards, still not 100%
    sure we (Nico and Alex) did this right. Oops.
    '''
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


def main(args):
    '''
    Load all three checkpoints and run the comparison.

    Everything is evaluated on CPU.

    S/o Claude for the fancy table formatting!
    '''
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
    summarise("CNN PPO  (4x84x84 pixels, skip=4)", cnn_res)

    print(f"\n  [ Stochastic / epsilon={trained_eps} (as trained) ]")
    dqn_res_s = run_episodes_dqn(dqn_model, make_cnn_eval_env, n, epsilon=trained_eps, device=device)
    ram_res_s = run_episodes_ppo(ram_model, make_ram_eval_env, n, greedy=False, device=device)
    cnn_res_s = run_episodes_ppo(cnn_model, make_cnn_eval_env, n, greedy=False, device=device)
    summarise(f"DQN  ({dqn_steps:,} steps, ε={trained_eps})", dqn_res_s)
    summarise("RAM PPO  (128-dim RAM features, sampled)", ram_res_s)
    summarise("CNN PPO  (4x84x84 pixels, sampled)", cnn_res_s)

    print(f"\n{'─'*60}\n")


if __name__ == "__main__":
    '''
    Argparse slop, cont.
    '''
    _here = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn-checkpoint", default=os.path.join(_here, "..", "dqn", "mario_dqn_step1150000.pt"))
    parser.add_argument("--ram-checkpoint", default=os.path.join(_here, "..", "ramPPO", "ppo_final.pt"))
    parser.add_argument("--cnn-checkpoint", default=os.path.join(_here, "..", "cnnPPO", "bestCNN.pt"))
    parser.add_argument("--episodes", type=int, default=10)
    main(parser.parse_args())
