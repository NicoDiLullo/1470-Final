'''
In our effort to be "good" software engineers, we've abstracted some of the
core logic out from RAM and CNN PPOs. 
'''

from collections import deque
import time

import gym
import numpy as np
import torch
from torch.distributions import Categorical
from PIL import Image

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


class SkipWrapper(gym.Wrapper):

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        '''
        Apply one chosen action for several emulator frames.
        '''
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            #unless, of course, Mario got slimed or something
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class GrayscaleResizeWrapper(gym.ObservationWrapper):
    '''
    Convert RGB frames to grayscale 84x84 images, scaled to [0, 1]. "standard Atari
    preprocessing" described in our report"
    '''

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(84, 84), dtype=np.float32
        )

    def observation(self, obs):
        img = Image.fromarray(obs).convert("L").resize((84, 84), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0


class FrameStackWrapper(gym.Wrapper):
    '''
    Stack the last n frames along a new leading axis.
    '''

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


def compute_ppo_loss(model, obs_b, act_b, old_log_probs_b, advantages_b, returns_b,
                     clip_eps, vf_coef, ent_coef):
    '''
    Shared PPO loss used by both the CNN and RAM agents.

    The policy part compares the new log-probabilities to the old ones saved during
    rollout collection. I think we talk about ratio clipping, our value loss, and entropy bonus in the report.
    Would be funny if we didn't.
    '''
    logits, values = model(obs_b)
    dist = Categorical(logits=logits)
    new_log_probs = dist.log_prob(act_b)
    entropy = dist.entropy().mean()

    ratio = torch.exp(new_log_probs - old_log_probs_b)
    pg_loss = torch.max(
        -advantages_b * ratio,
        -advantages_b * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps),
    ).mean()

    v_loss = 0.5 * (values - returns_b).pow(2).mean()
    loss = pg_loss + vf_coef * v_loss - ent_coef * entropy
    return loss, pg_loss, v_loss, entropy


def make_ram_eval_env():
    '''
    Build the RAM evaluation env.

    This mirrors the RAM PPO observation setup, but leaves out reward shaping so the
    score is comparable to the pixel agent and to the base Mario environment.
    '''
    from ramPPO.ram_ppo import RamFeatureWrapper

    env = gym.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = RamFeatureWrapper(env)
    return env


def make_cnn_eval_env():
    '''
    Build the CNN evaluation env.

    This is the same pixel preprocessing stack used by the CNN PPO agent: simple
    actions, frame skipping, grayscale resize, and a 4-frame stack so the model can
    infer motion.
    '''
    env = gym.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipWrapper(env, skip=4)
    env = GrayscaleResizeWrapper(env)
    env = FrameStackWrapper(env, n=4)
    return env


def run_episodes_ppo(model, make_env_fn, n_episodes, greedy, device):
    '''
    Run full Mario episodes and keep the stats that are useful for comparison.

    We track raw reward, furthest x-position, flag completion (done %), episode length, and
    per-step inference time. Greedy mode shows the model's favorite behavior; sampled
    mode shows what it does when we let the learned policy stay stochastic.
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
    '''
    Time pure model forward passes without the emulator in the loop (trying to remove
    the environment from the equation).
    '''
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
    '''
    Print the episode-level table plus aggregate stats.

    S/o Claude for the formatting.
    '''
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
