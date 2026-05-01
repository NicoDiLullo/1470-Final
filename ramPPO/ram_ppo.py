import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import time
from collections import deque
import argparse
from typing import List

from core.core import compute_ppo_loss

def _find_nes_env(env):
    e = env
    while e is not None:
        if hasattr(e, "ram"):
            return e
        e = getattr(e, "env", None)
    raise AttributeError("No NES env with 'ram' found in wrapper chain")


class RamFeatureWrapper(gym.ObservationWrapper):
    '''
    Learning off of raw memory inputs was inefficient. Learn of a small subset instead.
    Thanks to Claude for picking these out
    '''
    # Confirmed RAM addresses from gym-super-mario-bros smb_env.py source:
    #   x_pos:       ram[0x6D]*256 + ram[0x86]
    #   screen_x:    (ram[0x86] - ram[0x071C]) % 256
    #   y_pos:       ram[0x03B8]
    #   x_vel:       ram[0x007B] (signed)
    #   player_state:ram[0x000E]
    #   status:      ram[0x0756]  (0=small,1=big,2=fire)
    #   lives:       ram[0x075A]
    #   time:        ram[0x07F8..0x07FA] (BCD)
    #   coins:       ram[0x07ED..0x07EE] (BCD)
    #   score:       ram[0x07DE..0x07E3] (BCD)
    #   enemy types: ram[0x0016..0x001A]
    #   enemy x:     ram[0x0087..0x008B]  (screen coords)
    #   enemy y:     ram[0x00B6..0x00BA]

    OBS_DIM = 128

    def __init__(self, env):
        super().__init__(env)
        self._nes_env = _find_nes_env(env)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )

    def observation(self, obs):
        ram = self._nes_env.ram
        features = []

        # --- Mario position & motion (confirmed addresses) ---
        mario_abs_x = int(ram[0x6D]) * 256 + int(ram[0x86])
        mario_screen_x = (int(ram[0x86]) - int(ram[0x071C])) % 256
        mario_y = int(ram[0x03B8])
        mario_x_vel = np.int8(ram[0x7B]) / 16.0   # signed pixel velocity
        mario_y_vel = np.int8(ram[0x9F]) / 16.0   # signed (approx)
        features.extend([
            mario_abs_x / 65535.0,
            mario_screen_x / 255.0,
            mario_y / 255.0,
            mario_x_vel,
            mario_y_vel,
        ])

        # --- Player status ---
        player_state  = int(ram[0x000E])   # 0x06=dead, 0x08=normal, 0x0B=dying
        player_status = int(ram[0x0756])   # 0=small, 1=big, 2=fire
        lives         = int(ram[0x075A])
        features.extend([
            player_state / 12.0,
            player_status / 2.0,
            lives / 5.0,
        ])

        # --- Time, coins, score ---
        time_left = int(ram[0x07F8]) * 100 + int(ram[0x07F9]) * 10 + int(ram[0x07FA])
        coins = int(ram[0x07ED]) * 10 + int(ram[0x07EE])
        score = sum(int(ram[0x07DE + i]) * (10 ** (5 - i)) for i in range(6))
        camera_x = int(ram[0x071C])
        features.extend([
            time_left / 400.0,
            coins / 99.0,
            score / 999_999.0,
            camera_x / 255.0,
        ])

        # --- Enemies: 5 slots × 4 features = 20 ---
        for i in range(5):
            etype  = int(ram[0x0016 + i])   # confirmed enemy type address
            ex     = int(ram[0x0087 + i])   # enemy x on screen
            ey     = int(ram[0x00B6 + i])   # enemy y
            active = float(etype != 0)
            features.extend([
                etype / 255.0,
                ex / 255.0,
                ey / 255.0,
                active,
            ])

        # --- Strategic raw RAM regions (3 × 32 = 96 bytes) ---
        # 0x00-0x1F: object/enemy state flags
        # 0x6D-0x8C: position data
        # 0xB0-0xCF: velocity / misc
        for addr in range(0x00, 0x20):
            features.append(int(ram[addr]) / 255.0)
        for addr in range(0x6D, 0x8D):
            features.append(int(ram[addr]) / 255.0)
        for addr in range(0xB0, 0xD0):
            features.append(int(ram[addr]) / 255.0)

        assert len(features) == self.OBS_DIM, f"feature count mismatch: {len(features)}"
        return np.array(features, dtype=np.float32)


class CustomRewardWrapper(gym.Wrapper):
    '''
    Learned through an accidental ablation (models weren't training) that you need custom
    shaped rewards for Mario (the raw game rewards are too sparse)
    '''
    def __init__(self, env):
        super().__init__(env)
        self.prev_x_pos = 0
        self.prev_coins = 0
        self.prev_score = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x_pos = info.get("x_pos", 0)
        self.prev_coins = info.get("coins", 0)
        self.prev_score = info.get("score", 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        x_reward = (info.get("x_pos", 0) - self.prev_x_pos) * 1.0
        coin_reward = max(0, info.get("coins", 0) - self.prev_coins) * 10.0
        score_reward = (info.get("score", 0) - self.prev_score) * 0.05
        time_penalty = -0.01
        death_penalty = -15.0 if terminated and info.get("life", 1) < 2 else 0.0
        backward_penalty = min(0, x_reward) * 0.5

        shaped_reward = x_reward + coin_reward + score_reward + time_penalty + death_penalty + backward_penalty

        self.prev_x_pos = info.get("x_pos", 0)
        self.prev_coins = info.get("coins", 0)
        self.prev_score = info.get("score", 0)

        return obs, shaped_reward, terminated, truncated, info


def make_single_env(level: str = "SuperMarioBros-1-1-v0"):
    # apply_api_compatibility converts old 4-value step to new 5-value form
    # at the gym.make level, so all outer wrappers see new-style API
    env = gym.make(level, apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = RamFeatureWrapper(env)
    env = CustomRewardWrapper(env)
    return env


def make_vec_env(num_envs: int = 8, levels: List[str] = None):
    '''
    Vectorization is fun!
    '''
    if levels is None:
        levels = ["SuperMarioBros-1-1-v0"] * num_envs
    else:
        levels = (levels * (num_envs // len(levels) + 1))[:num_envs]

    env_fns = [lambda l=l: make_single_env(l) for l in levels]
    return gym.vector.AsyncVectorEnv(env_fns)

class ActorCritic(nn.Module):
    '''
    ActorCritic definition
    '''
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value


class RolloutBuffer:
    '''
    RolloutBuffer class
    '''
    def __init__(self, n_steps: int, obs_dim: int, num_envs: int, device: torch.device):
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.device = device
        self.obs = torch.zeros((n_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((n_steps, num_envs), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((n_steps, num_envs), device=device)
        self.rewards = torch.zeros((n_steps, num_envs), device=device)
        self.dones = torch.zeros((n_steps, num_envs), device=device)
        self.values = torch.zeros((n_steps, num_envs), device=device)
        self.ptr = 0

    def store(self, obs, actions, log_probs, rewards, dones, values):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.values[self.ptr] = values
        self.ptr += 1

    def full(self):
        return self.ptr >= self.n_steps

    def reset(self):
        self.ptr = 0

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float):
        advantages = torch.zeros(self.n_steps, self.num_envs, device=self.device)
        last_gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.n_steps)):
            next_val = last_values if t == self.n_steps - 1 else self.values[t + 1]
            next_done = 0.0 if t == self.n_steps - 1 else self.dones[t + 1]
            delta = self.rewards[t] + gamma * next_val * (1 - next_done) - self.values[t]
            last_gae = delta + gamma * lam * (1 - next_done) * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns


class PPO:
    '''
    PPO class
    '''
    def __init__(self, obs_dim: int, n_actions: int, device: torch.device,
                 lr: float = 2.5e-4, gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.03, n_steps: int = 2048,
                 batch_size: int = 256, n_epochs: int = 4,
                 max_grad_norm: float = 0.5):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.model = ActorCritic(obs_dim, n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.buffer = None

    def update(self, last_values: torch.Tensor):
        advantages, returns = self.buffer.compute_returns(last_values, self.gamma, self.lam)

        total_pg_loss = total_v_loss = total_ent = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = torch.randperm(self.buffer.n_steps * self.buffer.num_envs, device=self.device)
            for start in range(0, self.buffer.n_steps * self.buffer.num_envs, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                b_obs = self.buffer.obs.view(-1, self.buffer.obs.shape[-1])[idx]
                b_act = self.buffer.actions.view(-1)[idx]
                b_old_lp = self.buffer.log_probs.view(-1)[idx]
                b_adv = advantages.view(-1)[idx]
                b_ret = returns.view(-1)[idx]

                loss, pg_loss, v_loss, entropy = compute_ppo_loss(
                    self.model, b_obs, b_act, b_old_lp, b_adv, b_ret,
                    self.clip_eps, self.value_coef, self.entropy_coef,
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_ent += entropy.item()
                n_updates += 1

        self.buffer.reset()
        return total_pg_loss / n_updates, total_v_loss / n_updates, total_ent / n_updates


def train(args):
    '''
    Training loop
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} | {args.num_envs} parallel environments")

    env = make_vec_env(num_envs=args.num_envs)
    obs_dim = env.single_observation_space.shape[0]
    n_actions = env.single_action_space.n

    ppo = PPO(obs_dim=obs_dim, n_actions=n_actions, device=device,
              n_steps=args.n_steps, entropy_coef=args.entropy_start)
    ppo.buffer = RolloutBuffer(args.n_steps, obs_dim, args.num_envs, device)

    if args.load:
        ppo.model.load_state_dict(torch.load(args.load, map_location=device))
        print(f"Loaded weights from {args.load}")

    writer = SummaryWriter(log_dir=f"runs/{args.run_name}") if args.use_tb else None

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    timestep = 0
    episode_rewards = deque(maxlen=100)
    ep_returns = np.zeros(args.num_envs, dtype=np.float32)
    ep_count = 0
    last_log = 0
    t0 = time.time()

    print(f"Starting training for {args.total_timesteps:,} timesteps...")
    print(f"Entropy: {args.entropy_start} → {args.entropy_end} (linear decay)")

    while timestep < args.total_timesteps:
        # Linear entropy decay
        progress = min(timestep / args.total_timesteps, 1.0)
        ppo.entropy_coef = args.entropy_start + progress * (args.entropy_end - args.entropy_start)

        with torch.no_grad():
            for _ in range(args.n_steps):
                logits, values = ppo.model(obs)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

                next_obs, rewards, terminated, truncated, infos = env.step(actions.cpu().numpy())
                dones = terminated | truncated

                ppo.buffer.store(obs, actions, log_probs,
                                 torch.tensor(rewards, dtype=torch.float32, device=device),
                                 torch.tensor(dones, dtype=torch.float32, device=device),
                                 values)

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

            with torch.no_grad():
                _, last_values = ppo.model(obs)

        pg_loss, v_loss, entropy = ppo.update(last_values)

        if writer:
            writer.add_scalar("train/pg_loss", pg_loss, timestep)
            writer.add_scalar("train/value_loss", v_loss, timestep)
            writer.add_scalar("train/entropy", entropy, timestep)
            writer.add_scalar("train/entropy_coef", ppo.entropy_coef, timestep)
            if episode_rewards:
                writer.add_scalar("train/mean_reward_100ep", np.mean(episode_rewards), timestep)

        if timestep - last_log >= args.log_every:
            elapsed = time.time() - t0
            fps = timestep / elapsed
            mean_rew = np.mean(episode_rewards) if episode_rewards else 0.0
            print(f"[{timestep:>8,}]  episodes={ep_count:4d}  "
                  f"mean_rew={mean_rew:7.1f}  pg={pg_loss:.4f}  v={v_loss:.4f}  "
                  f"ent={entropy:.4f}  ent_coef={ppo.entropy_coef:.4f}  fps={fps:.0f}")
            last_log = timestep

    env.close()
    if writer:
        writer.close()
    torch.save(ppo.model.state_dict(), f"mario_ram_ppo_final_{args.run_name}.pt")
    print("Training complete! Model saved.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--log-every", type=int, default=20_000)
    parser.add_argument("--run-name", type=str, default="mario_ram_ppo_v2")
    parser.add_argument("--entropy-start", type=float, default=0.05)
    parser.add_argument("--entropy-end", type=float, default=0.002)
    parser.add_argument("--load", type=str, default=None, help="Path to checkpoint to continue from")
    parser.add_argument("--no-tb", action="store_true")
    args = parser.parse_args()

    args.use_tb = not args.no_tb
    train(args)