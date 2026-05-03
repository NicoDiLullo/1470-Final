'''
In our effort to be "good" software engineers, we've abstracted some of the
core logic out from RAM and CNN PPOs. 
'''

from collections import deque

import gym
import numpy as np
import torch
from torch.distributions import Categorical
from PIL import Image


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
