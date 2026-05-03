'''
A comparison script for CNN and RAM based PPO.

This uses raw game rewards. Training used shaped rewards because Mario
is sparse(ish) and painful otherwise, but evaluation should only focus on perf in the 
real environment.

Also not really sure if doing each w/ its custom shaped reward is a model or reward
ablation (and least on completion).
'''

import warnings
warnings.filterwarnings("ignore")

import os, sys, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch

from core.core import (
    inference_benchmark,
    make_cnn_eval_env,
    make_ram_eval_env,
    print_section,
    run_episodes_ppo as run_episodes,
    summarise,
)
from ramPPO.ram_ppo import ActorCritic as RamActorCritic

from cnnPPO.ppo_agent_final import ActorCritic as CnnActorCritic


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
    '''
    Argparse slop.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--ram-checkpoint", default=os.path.join(os.path.dirname(__file__), "..", "ramPPO", "ppo_final.pt"))
    parser.add_argument("--cnn-checkpoint", default=os.path.join(os.path.dirname(__file__), "..", "cnnPPO", "bestCNN.pt"))
    parser.add_argument("--episodes", type=int, default=10)
    main(parser.parse_args())
