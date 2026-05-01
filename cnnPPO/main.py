import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import time
import os
import torch
from collections import deque
import numpy as np
from ppo_agent_final import PPO, ActorCritic, RolloutBuffer, make_vec_env

#Training loop
def train(args):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training CNN PPO on {device} | {args.num_envs} envs")
    os.makedirs("checkpoints", exist_ok=True)

    env = make_vec_env(num_envs=args.num_envs)
    model = ActorCritic(n_actions=env.single_action_space.n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    buffer = RolloutBuffer(args.n_steps, args.num_envs, device)
    ppo = PPO(model, optimizer, buffer, args)

    if not args.no_tb:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f"runs/{args.run_name}")
    else:
        writer = None

    timestep = 0
    ep_count = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            timestep = ckpt["timestep"]
            ep_count = ckpt.get("ep_count", 0)
        else:
            model.load_state_dict(ckpt)
        print(f"Resumed from {args.resume} at timestep {timestep:,}")

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    ep_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_rewards = deque(maxlen=100)
    last_log = timestep
    last_save = timestep
    t0 = time.time()
    timestep_start = timestep

    print(f"Starting training for {args.total_timesteps:,} timesteps...")

    while timestep < args.total_timesteps:
        with torch.no_grad():
            for _ in range(args.n_steps):
                actions, log_probs, _, values = model.get_action(obs)

                next_obs, rewards, terminated, truncated, infos = env.step(actions.cpu().numpy())
                dones = terminated | truncated

                buffer.store(
                    obs,
                    actions,
                    log_probs,
                    torch.tensor(rewards, dtype=torch.float32, device=device),
                    torch.tensor(dones, dtype=torch.float32, device=device),
                    values,
                )

                ep_returns += rewards
                for i, done in enumerate(dones):
                    if done:
                        episode_rewards.append(ep_returns[i])
                        ep_returns[i] = 0.0
                        ep_count += 1

                obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
                timestep += args.num_envs

                if timestep >= args.total_timesteps:
                    break

            _, last_values = model(obs)

        pg_loss, v_loss, entropy = ppo.update(last_values)

        if writer:
            writer.add_scalar("train/pg_loss", pg_loss, timestep)
            writer.add_scalar("train/value_loss", v_loss, timestep)
            writer.add_scalar("train/entropy", entropy, timestep)
            if episode_rewards:
                writer.add_scalar("train/mean_reward_100ep", np.mean(episode_rewards), timestep)

        if timestep - last_log >= args.log_every:
            fps = (timestep - timestep_start) / (time.time() - t0)
            mean_rew = np.mean(episode_rewards) if episode_rewards else 0.0
            print(f"[{timestep:>8,}]  episodes={ep_count:4d}  mean_rew={mean_rew:7.1f}  "
                  f"pg={pg_loss:.4f}  v={v_loss:.4f}  ent={entropy:.4f}  fps={fps:.0f}")
            last_log = timestep

        if args.save_every > 0 and timestep - last_save >= args.save_every:
            ckpt_path = f"checkpoints/mario_cnn_ppo_{timestep}.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "timestep": timestep, "ep_count": ep_count}, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")
            last_save = timestep

    env.close()
    if writer:
        writer.close()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "timestep": timestep, "ep_count": ep_count}, "mario_cnn_ppo_final.pt")
    print("Training complete. Saved mario_cnn_ppo_final.pt")


#Entry point/argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--vf-coef", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=20_000)
    parser.add_argument("--run-name", type=str, default="mario_cnn_ppo")
    parser.add_argument("--no-tb", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=1_000_000)
    args = parser.parse_args()

    train(args)