"""
CNN-based PPO for Super Mario Bros (pixel observations)

Final Draft.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from collections import deque

from PIL import Image

#bridge
class OldGymToGymnasium(gym.Env):
    def __init__(self, old_env):
        super().__init__()
        self.env = old_env
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=old_env.observation_space.shape, dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(old_env.action_space.n)

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()


#preprocess
class GrayscaleResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(84, 84), dtype=np.float32)

    def observation(self, obs):
        img = Image.fromarray(obs).convert("L").resize((84, 84), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0


class SkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


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


# Custom shaped rewards (same as RAM)
class CustomRewardWrapper(gym.Wrapper):
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
        time_penalty = -0.1
        death_penalty = -50.0 if terminated and info.get("life", 1) < 2 else 0.0
        backward_penalty = min(0, x_reward) * 0.5

        shaped_reward = x_reward + coin_reward + score_reward + time_penalty + death_penalty + backward_penalty

        self.prev_x_pos = info.get("x_pos", 0)
        self.prev_coins = info.get("coins", 0)
        self.prev_score = info.get("score", 0)

        return obs, shaped_reward, terminated, truncated, info


# Environment factory (pixel version)
def make_single_env(level: str = "SuperMarioBros-1-1-v0"):
    env = gym.make(level, apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipWrapper(env, skip=4)
    env = CustomRewardWrapper(env)
    env = GrayscaleResizeWrapper(env)
    env = FrameStackWrapper(env, n=4)
    return env


def make_vec_env(num_envs: int = 8):
    env_fns = [lambda: make_single_env() for _ in range(num_envs)]
    return gym.vector.AsyncVectorEnv(env_fns)


#Actor-Critic
class ActorCritic(nn.Module):
    def __init__(self, n_actions=7):
        super().__init__()

        # shared CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )

        # flattens to expected input size
        self.fc = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())

        # policy head -> actor
        self.policy_head = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

        # value head -> critic
        self.value_head = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1)
        )

        # Orthogonal init -> used to help out stability
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01) # shirnks initial policy logits so action bias isn't too bad at beginning
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def forward(self, x):
        '''
        forward pass
        '''
        h = self.fc(self.encoder(x).flatten(1))
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def get_action(self, x):
        '''
        returns an action based on policy dist
        '''
        logits, value = self(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


#RolloutBuffer
class RolloutBuffer:
    def __init__(self, n_steps, num_envs, device):
        '''
        Essentially stores the experience of an actor through some
        number of timesteps over all the environments. These experiences
        will then be used to train the actor-critic network based on
        gae computations of the actor's performance.
        '''
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.device = device
        self.obs = torch.zeros(n_steps, num_envs, 4, 84, 84, device=device)
        self.actions = torch.zeros(n_steps, num_envs, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(n_steps, num_envs, device=device)
        self.rewards = torch.zeros(n_steps, num_envs, device=device)
        self.dones = torch.zeros(n_steps, num_envs, device=device)
        self.values = torch.zeros(n_steps, num_envs, device=device)
        self.ptr = 0

    def store(self, obs, actions, log_probs, rewards, dones, values):
        '''
        Stores all of the necessary values for each experience.
        '''
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.values[self.ptr] = values
        self.ptr += 1

    def reset(self):
        self.ptr = 0

    def compute_returns(self, last_values, gamma, lam):
        '''
        Computes gae returns. 

        Advantages = how much better the actor performed than the critic thought
        Returns = how good the critic thinks the current state is + advantages
        '''
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

#PPO
class PPO:
    def __init__(self, model, optimizer, buffer, args):
        '''
        Puts together both the RolloutBuffer and the ActorCritic into
        one system to make our PPO system.
        '''
        self.model = model
        self.optimizer = optimizer
        self.buffer = buffer
        self.args = args

    def update(self, last_values):
        '''
        Performs one like buffer step. In PPO we reuse the buffer
        some number of times before flushing it and making a new
        buffer with an updated policy. This performs one full step
        for one buffer.
        '''
        # find our old advantages and returns
        advantages, returns = self.buffer.compute_returns(last_values, self.args.gamma, self.args.gae_lambda)

        flat_obs = self.buffer.obs.view(-1, 4, 84, 84)
        flat_act = self.buffer.actions.view(-1)
        flat_lp = self.buffer.log_probs.view(-1)
        flat_adv = advantages.view(-1)
        flat_ret = returns.view(-1)

        total_pg = total_v = total_ent = 0.0
        n_updates = 0

        # train over several epochs using one buffer
        for _ in range(self.args.n_epochs):

            # minibatching logic
            idx = torch.randperm(flat_obs.shape[0], device=self.buffer.device)
            for start in range(0, flat_obs.shape[0], self.args.batch_size):
                b = idx[start:start + self.args.batch_size]

                # get current model distrubtion
                logits, values = self.model(flat_obs[b])
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(flat_act[b])
                entropy = dist.entropy().mean()

                # compare current model dist to old model dist
                ratio = torch.exp(new_lp - flat_lp[b])
                pg_loss = torch.max(
                    -flat_adv[b] * ratio,
                    -flat_adv[b] * torch.clamp(ratio, 1 - self.args.clip_range, 1 + self.args.clip_range)
                ).mean()

                # calculate our loss
                v_loss = 0.5 * (values - flat_ret[b]).pow(2).mean()
                loss = pg_loss + self.args.vf_coef * v_loss - self.args.ent_coef * entropy

                # step and clip
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

                # logging
                total_pg += pg_loss.item()
                total_v += v_loss.item()
                total_ent += entropy.item()
                n_updates += 1

        self.buffer.reset() # reset out buffer
        return total_pg / n_updates, total_v / n_updates, total_ent / n_updates

