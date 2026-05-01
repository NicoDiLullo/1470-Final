import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class MarioQNetwork(nn.Module):
  """
  CNN Q-network.

  Expects (B, C, H, W) observations.
  Head outputs (B, num_actions) Q-values.

  obs_shape: (C, H, W)
  num_actions: int
  """
  def __init__(self, obs_shape, num_actions, feature_dim=512):
    super().__init__()
    self.obs_shape = obs_shape
    self.num_actions = num_actions
    self.feature_dim = feature_dim

    c, h, w = self.obs_shape

    self.encoder = nn.Sequential(
        nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.Flatten(),
    )

    # Compute flattened CNN output size by doing a dummy forward (on CPU,
    # before .to(device) is called by the agent).
    with torch.no_grad():
        dummy = torch.zeros(1, c, h, w)
        n_flatten = self.encoder(dummy).shape[1]

    self.q_head = nn.Sequential(
        nn.Linear(n_flatten, feature_dim),
        nn.ReLU(),
        nn.Linear(feature_dim, self.num_actions)
    )

  def forward(self, obs):
    x = self.encoder(obs)
    q_values = self.q_head(x)
    return q_values


class ReplayBuffer:
  """
  Stores transitions for DQN. Frames as uint8 to save memory -- a buffer
  of 100k (4, 84, 84) frames at uint8 is ~2.8 GB, at float32 it would be ~11 GB.

  Capacity is the total number of transitions stored, NOT the number of env-steps.
  With num_envs > 1, the buffer fills in capacity / num_envs env-steps.
  """
  def __init__(self, capacity, num_envs, obs_shape, device):
    self.capacity = capacity
    self.num_envs = num_envs
    self.obs_shape = obs_shape
    self.device = device

    self.obs_buf      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
    self.next_obs_buf = np.zeros((capacity, *obs_shape), dtype=np.uint8)
    self.actions_buf  = np.zeros((capacity,), dtype=np.int64)
    self.rewards_buf  = np.zeros((capacity,), dtype=np.float32)
    self.dones_buf    = np.zeros((capacity,), dtype=np.float32)

    self.ptr = 0
    self.size = 0

  def add(self, obs, next_obs, actions, rewards, dones):
    """
    obs/next_obs: (num_envs, C, H, W) uint8
    actions: (num_envs,) int
    rewards: (num_envs,) float
    dones:   (num_envs,) bool

    For vectorization, computes indices for all envs at once and does a single
    batched assignment per buffer.
    """
    n = self.num_envs
    idx = (self.ptr + np.arange(n)) % self.capacity

    self.obs_buf[idx]      = obs
    self.next_obs_buf[idx] = next_obs
    self.actions_buf[idx]  = actions
    self.rewards_buf[idx]  = rewards
    self.dones_buf[idx]    = dones.astype(np.float32)

    self.ptr  = (self.ptr + n) % self.capacity
    self.size = min(self.size + n, self.capacity)

  def sample(self, batch_size):
    """
    returns a batch of transitions on the proper device, with obs normalized to [0,1]
    """
    idx = np.random.randint(0, self.size, size=batch_size)
    return {
        "obs":      torch.tensor(self.obs_buf[idx],      dtype=torch.float32, device=self.device) / 255.0,
        "next_obs": torch.tensor(self.next_obs_buf[idx], dtype=torch.float32, device=self.device) / 255.0,
        "actions":  torch.tensor(self.actions_buf[idx],  dtype=torch.int64,   device=self.device),
        "rewards":  torch.tensor(self.rewards_buf[idx],  dtype=torch.float32, device=self.device),
        "dones":    torch.tensor(self.dones_buf[idx],    dtype=torch.float32, device=self.device),
    }


class DQNAgent:
  """
  methods: learn, predict, save, load
  Implements a Double DQN agent that mirrors the PPOAgent interface.
  The obs_shape and num_actions parameters essentially replace the under the
  hood inference stuff that SB3 does for you from the env.

  verbose: 0 = silent, 1 = periodic stdout summaries (~every 10k env-steps,
  plus target syncs and start/end), 2 = print every training step.
  """

  # Attributes excluded from save() — runtime/non-hyperparameter state.
  _SAVE_HYPERPARAM_KEYS = (
      "obs_shape", "num_actions", "learning_rate", "buffer_size",
      "learning_starts", "batch_size", "gamma", "target_update_interval",
      "train_freq", "gradient_steps", "exploration_fraction",
      "exploration_initial_eps", "exploration_final_eps", "max_grad_norm",
      "num_envs",
  )

  def __init__(
      self,
      env,
      obs_shape,
      num_actions,
      learning_rate = 5e-5,
      buffer_size = 100_000,
      learning_starts = 10_000,
      batch_size = 32,
      gamma = 0.99,
      target_update_interval = 5_000,
      train_freq = 4,
      gradient_steps = 1,
      exploration_fraction = 0.5,
      exploration_initial_eps = 1.0,
      exploration_final_eps = 0.1,
      max_grad_norm = 1.0,
      tensorboard_log = None,
      tb_log_name = "DQN",
      log_interval = 100,
      verbose = 0,
      device = None):
    self.env = env
    self.num_envs = getattr(env, "num_envs", 1)

    self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.q_network      = MarioQNetwork(obs_shape, num_actions).to(self.device)
    self.target_network = MarioQNetwork(obs_shape, num_actions).to(self.device)
    self.target_network.load_state_dict(self.q_network.state_dict())

    # target net is never trained directly, just copied into
    for p in self.target_network.parameters():
      p.requires_grad = False

    self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
    self.buffer = ReplayBuffer(buffer_size, self.num_envs, obs_shape, self.device)

    self.obs_shape = obs_shape
    self.num_actions = num_actions
    self.learning_rate = learning_rate
    self.buffer_size = buffer_size
    self.learning_starts = learning_starts
    self.batch_size = batch_size
    self.gamma = gamma
    self.target_update_interval = target_update_interval
    self.train_freq = train_freq
    self.gradient_steps = gradient_steps
    self.exploration_fraction = exploration_fraction
    self.exploration_initial_eps = exploration_initial_eps
    self.exploration_final_eps = exploration_final_eps
    self.max_grad_norm = max_grad_norm

    # logging setup
    self.tensorboard_log = tensorboard_log
    self.tb_log_name = tb_log_name
    self.log_interval = log_interval  # episodes between rollout-stat dumps
    self.writer = None
    if tensorboard_log is not None:
      os.makedirs(tensorboard_log, exist_ok=True)
      existing = [d for d in os.listdir(tensorboard_log) if d.startswith(f"{tb_log_name}_")]
      run_idx = len(existing) + 1
      run_dir = os.path.join(tensorboard_log, f"{tb_log_name}_{run_idx}")
      self.writer = SummaryWriter(run_dir)
      print(f"Logging to {run_dir}")

    # rolling windows for episode stats (matches sb3's 100-ep window)
    self._ep_rew_buffer = deque(maxlen=100)
    self._ep_len_buffer = deque(maxlen=100)
    self._ep_x_pos_buffer = deque(maxlen=100)
    self._ep_count = 0

    # running state across learn() calls
    self._num_timesteps = 0

    # internal counters for train/target-update scheduling. These count
    # env-steps since the last trigger and are robust to any num_envs.
    self._steps_since_train = 0
    self._steps_since_target_update = 0

    # verbosity
    self.verbose = verbose
    self._train_step_count = 0

  def _current_epsilon(self, total_timesteps, learn_call_start_step):
    """
    Linear decay from initial_eps to final_eps over exploration_fraction of total_timesteps.
    Progress is measured relative to the start of THIS learn() call so that
    repeated learn() calls each get a fresh decay schedule.
    """
    steps_this_call = self._num_timesteps - learn_call_start_step
    progress = steps_this_call / max(1, total_timesteps)
    frac = min(progress / self.exploration_fraction, 1.0)
    return self.exploration_initial_eps + frac * (self.exploration_final_eps - self.exploration_initial_eps)

  def _log_episode_infos(self, infos):
    """
    Pulls per-episode stats out of VecMonitor's info dicts and Mario's info,
    pushes them into rolling windows. Dumps to tensorboard every log_interval episodes.
    """
    for info in infos:
      if "episode" in info:
        ep = info["episode"]
        self._ep_rew_buffer.append(ep["r"])
        self._ep_len_buffer.append(ep["l"])
        
        # x_pos sometimes lives on info, sometimes on info["episode"], depending on wrapper
        x_pos = info.get("x_pos", ep.get("x_pos", None))
        if x_pos is not None:
          self._ep_x_pos_buffer.append(x_pos)
        self._ep_count += 1

        if self.writer is not None and self._ep_count % self.log_interval == 0:
          self.writer.add_scalar("rollout/ep_rew_mean", np.mean(self._ep_rew_buffer), self._num_timesteps)
          self.writer.add_scalar("rollout/ep_len_mean", np.mean(self._ep_len_buffer), self._num_timesteps)
          if len(self._ep_x_pos_buffer) > 0:
            self.writer.add_scalar("mario/x_pos_mean", np.mean(self._ep_x_pos_buffer), self._num_timesteps)
            self.writer.add_scalar("mario/x_pos_max",  np.max(self._ep_x_pos_buffer),  self._num_timesteps)

      if info.get("flag_get", False) and self.writer is not None:
        self.writer.add_scalar("mario/flag_get", 1.0, self._num_timesteps)

  def predict(self, obs, deterministic=False, epsilon=0.0):
    '''
    Based on an observation, picks an action. If not deterministic, with
    probability epsilon picks a random action (epsilon-greedy).

    Returns (actions, q_values). q_values is a zero array of shape
    (batch, num_actions) on the random-action branch so the return type
    is consistent.
    '''
    if obs.ndim == 3:
      obs = obs[None, ...]

    batch = obs.shape[0]

    if (not deterministic) and np.random.rand() < epsilon:
      actions = np.random.randint(0, self.num_actions, size=batch)
      return actions, np.zeros((batch, self.num_actions), dtype=np.float32)

    # The network has no dropout or batchnorm, so train/eval mode doesn't
    # actually change behavior here. We use no_grad() for the speed/memory win.
    obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device) / 255.0
    with torch.no_grad():
      q_values = self.q_network(obs_t)
      actions = torch.argmax(q_values, dim=-1)

    return actions.cpu().numpy(), q_values.cpu().numpy()

  def learn(self, total_timesteps, callback=None, reset_num_timesteps=True):
    '''
    trains the model.

    reset_num_timesteps: if True, resets _num_timesteps and the epsilon schedule
    relative to the start of this call. SB3 uses the same flag.
    '''
    if reset_num_timesteps:
      self._num_timesteps = 0
      self._steps_since_train = 0
      self._steps_since_target_update = 0

    learn_call_start_step = self._num_timesteps
    target_total = learn_call_start_step + total_timesteps

    obs = self.env.reset()
    start_time = time.time()
    last_train_log_step = 0
    last_stdout_step = 0

    if self.verbose >= 1:
      print(f"[learn] starting: total_timesteps={total_timesteps}, num_envs={self.num_envs}, device={self.device}")

    while self._num_timesteps < target_total:
      epsilon = self._current_epsilon(total_timesteps, learn_call_start_step)

      if self._num_timesteps < self.learning_starts:
        actions = np.array([self.env.action_space.sample() for _ in range(self.num_envs)])
      else:
        actions, _ = self.predict(obs, deterministic=False, epsilon=epsilon)

      next_obs, rewards, dones, infos = self.env.step(actions)

      self.buffer.add(obs, next_obs, actions, rewards, dones)

      obs = next_obs
      self._num_timesteps += self.num_envs
      self._steps_since_train += self.num_envs
      self._steps_since_target_update += self.num_envs

      self._log_episode_infos(infos)

      # Train after warmup, once per train_freq env-steps. Using a counter
      # rather than a modulo on _num_timesteps so this is correct for any
      # num_envs (including cases where num_envs doesn't divide train_freq).
      if (self._num_timesteps >= self.learning_starts
          and self._steps_since_train >= self.train_freq):
        self._steps_since_train = 0

        train_stats_accum = {"td_loss": 0.0, "q_values": 0.0,
                             "td_target_mean": 0.0, "td_target_max": float("-inf")}
        for _ in range(self.gradient_steps):
          stats = self._train_step()
          train_stats_accum["td_loss"]        += stats["td_loss"]
          train_stats_accum["q_values"]       += stats["q_values"]
          train_stats_accum["td_target_mean"] += stats["td_target_mean"]
          train_stats_accum["td_target_max"]   = max(train_stats_accum["td_target_max"], stats["td_target_max"])

        # per-train-step stdout at verbose>=2
        if self.verbose >= 2:
          avg_loss = train_stats_accum["td_loss"] / self.gradient_steps
          avg_q = train_stats_accum["q_values"] / self.gradient_steps
          print(f"[train step {self._train_step_count:>7d} | env_step {self._num_timesteps:>8d}] "
                f"td_loss={avg_loss:.5f}  q_mean={avg_q:+.3f}  eps={epsilon:.3f}  buf={self.buffer.size}")
        self._train_step_count += 1

        if self.writer is not None and (self._num_timesteps - last_train_log_step) >= 1000:
          self.writer.add_scalar("train/td_loss", train_stats_accum["td_loss"] / self.gradient_steps, self._num_timesteps)
          self.writer.add_scalar("train/q_values", train_stats_accum["q_values"] / self.gradient_steps, self._num_timesteps)
          self.writer.add_scalar("train/td_target_mean", train_stats_accum["td_target_mean"] / self.gradient_steps, self._num_timesteps)
          self.writer.add_scalar("train/td_target_max", train_stats_accum["td_target_max"], self._num_timesteps)
          self.writer.add_scalar("train/epsilon", epsilon, self._num_timesteps)
          self.writer.add_scalar("train/buffer_size", self.buffer.size, self._num_timesteps)
          fps = int(self._num_timesteps / max(1e-6, time.time() - start_time))
          self.writer.add_scalar("time/fps", fps, self._num_timesteps)
          last_train_log_step = self._num_timesteps

      # periodic stdout summary at verbose>=1 (every ~10k env-steps)
      if self.verbose >= 1 and (self._num_timesteps - last_stdout_step) >= 10_000:
        elapsed = time.time() - start_time
        fps = int(self._num_timesteps / max(1e-6, elapsed))
        ep_rew = np.mean(self._ep_rew_buffer) if len(self._ep_rew_buffer) > 0 else float("nan")
        ep_len = np.mean(self._ep_len_buffer) if len(self._ep_len_buffer) > 0 else float("nan")
        x_max = np.max(self._ep_x_pos_buffer) if len(self._ep_x_pos_buffer) > 0 else float("nan")
        print(f"[{self._num_timesteps:>8d} steps | {elapsed:6.1f}s | {fps:>4d} fps] "
              f"eps={epsilon:.3f}  ep_rew={ep_rew:+.2f}  ep_len={ep_len:.0f}  x_max={x_max:.0f}  "
              f"episodes={self._ep_count}  buf={self.buffer.size}")
        last_stdout_step = self._num_timesteps

      # Same counter pattern for target sync.
      if (self._num_timesteps >= self.learning_starts
          and self._steps_since_target_update >= self.target_update_interval):
        self._steps_since_target_update = 0
        self.target_network.load_state_dict(self.q_network.state_dict())
        if self.verbose >= 1:
          print(f"[target sync @ env_step {self._num_timesteps}]")

      if callback is not None:
        callback(locals(), globals())

    if self.writer is not None:
      self.writer.flush()

    if self.verbose >= 1:
      elapsed = time.time() - start_time
      print(f"[learn] done: {self._num_timesteps} env-steps in {elapsed:.1f}s "
            f"({int(self._num_timesteps/max(1e-6,elapsed))} fps)")

    return self

  def _train_step(self):
    '''
    one gradient update on a batch sampled from the replay buffer.
    Uses Double DQN: action selection by online net, value evaluation by target net.
    '''
    batch = self.buffer.sample(self.batch_size)

    obs      = batch["obs"]
    next_obs = batch["next_obs"]
    actions  = batch["actions"]
    rewards  = batch["rewards"]
    dones    = batch["dones"]

    # Double DQN target: pick best action with the online net, evaluate it with
    # the target net. This decouples action selection from value estimation and
    # substantially reduces the overestimation bias that vanilla DQN suffers.
    with torch.no_grad():
      next_actions = self.q_network(next_obs).argmax(dim=1, keepdim=True)
      next_q = self.target_network(next_obs).gather(1, next_actions).squeeze(1)
      td_target = rewards + self.gamma * next_q * (1.0 - dones)

    current_q = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Huber loss is more robust to reward outliers than MSE
    loss = F.smooth_l1_loss(current_q, td_target)

    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
    self.optimizer.step()

    return {
        "td_loss": loss.item(),
        "q_values": current_q.mean().item(),
        "td_target_mean": td_target.mean().item(),
        "td_target_max": td_target.max().item(),
    }

  def save(self, path):
    # Allowlist: only persist named hyperparameters, not arbitrary attributes.
    hyperparams = {k: getattr(self, k) for k in self._SAVE_HYPERPARAM_KEYS}
    torch.save({
        "q_network":      self.q_network.state_dict(),
        "target_network": self.target_network.state_dict(),
        "optimizer":      self.optimizer.state_dict(),
        "num_timesteps":  self._num_timesteps,
        "hyperparams":    hyperparams,
    }, path)

  def load(self, path):
    checkpoint = torch.load(path, map_location=self.device, weights_only=False)
    self.q_network.load_state_dict(checkpoint["q_network"])
    self.target_network.load_state_dict(checkpoint["target_network"])
    self.optimizer.load_state_dict(checkpoint["optimizer"])
    self._num_timesteps = checkpoint.get("num_timesteps", 0)
    return self