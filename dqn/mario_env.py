import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from collections import deque
import cv2
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

# Bridge wrapper: old gym to gymnasium
class OldGymToGymnasium(gym.Env):
  """
  Converts the old gym-super-mario-bros env to the gymnasium API that stable-baselines3 expects
  """
  def __init__(self, old_env):
    super().__init__()
    self.env = old_env
    self.observation_space = Box(
        low=0, high=255,
        shape=old_env.observation_space.shape,
        dtype=np.uint8
    )
    self.action_space = gym.spaces.Discrete(old_env.action_space.n)

  def reset(self, seed=None, options=None):
    obs = self.env.reset() #old gym only returns the observation
    return obs, {} #gymnasium returns obs and info dict

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    return obs, reward, done, False, info #gymnasium returns obs, reward, terminated (like if mario died), truncated(always False), info

  def render(self):
    return self.env.render()

  def close(self):
    self.env.close()


#Frame Skip: because NES runs super mario at 60fps we cant make a 60 decisions every second, so we choose the same decision for 4 frames to reduce the amount of decisions
class SkipFrame(gym.Wrapper):
  """
  Repeat same action for 'skip' frames and then sum the rewards
  """
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


#Grayscale
class GrayScaleObservation(gym.ObservationWrapper):
  """
  converts RGB to grayscale
  """
  def __init__(self,env):
    super().__init__(env)
    obs_shape = self.observation_space.shape[:2]
    self.observation_space = Box(low=0,high=255, shape=obs_shape, dtype=np.uint8)

  def observation(self, obs):
    return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)


# resize to 84 x 84, because NES output is shaped (240, 256, 3)
class ResizeObservation(gym.ObservationWrapper):
  """
  downsample frames to 84x84 for the CNN
  """

  def __init__(self, env, shape=84):
    super().__init__(env)
    self.shape = (shape,shape)
    self.observation_space = Box(low=0,high=255,shape=self.shape,dtype=np.uint8)

  def observation(self, obs):
    return cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA) #inter area averages pixel blocks instead of dropping pixels


class StackFrames(gym.ObservationWrapper):
  """
  stacking consecutive frames so the agent can understand motion and velocity
  """
  def __init__(self, env, num_stack=4):
    super().__init__(env)
    self.num_stack = num_stack
    self.frames = deque(maxlen=num_stack) #double ended queue with max length num stack
    low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
    high = np.repeat(self.observation_space.high[np.newaxis, ...],num_stack,axis=0)
    self.observation_space = Box(low=low, high=high, dtype=np.uint8)

  # since our agent sees four frames at a time, so it takes the first frame and copies it 4 times to show that there is nothing moving yet
  def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    for _ in range(self.num_stack):
      self.frames.append(obs)
    return np.array(self.frames), info

  #handles every step after reset.
  def observation(self, obs):
    self.frames.append(obs)
    return np.array(self.frames)

  #After reset:   [frame1, frame1, frame1, frame1]    # no motion info
  #After step 1:  [frame1, frame1, frame1, frame2]    # starting to see change
  #After step 2:  [frame1, frame1, frame2, frame3]
  #After step 3:  [frame1, frame2, frame3, frame4]    # full history now
  #After step 4:  [frame2, frame3, frame4, frame5]    # frame1 dropped off

#Environment factory
def make_mario_env(level = "SuperMarioBros-v3", action_set =SIMPLE_MOVEMENT):
  """
  Creates and wraps a Mario environment ready for sb3
  """

  env = gym_super_mario_bros.make(level)
  env = JoypadSpace(env, action_set)
  env = OldGymToGymnasium(env)
  env = SkipFrame(env, skip=4)
  env = GrayScaleObservation(env)
  env = ResizeObservation(env,shape=84)
  env = StackFrames(env, num_stack=4)
  return env


