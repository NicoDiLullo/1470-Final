from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import os
import time
from dqn_agent import DQNAgent
from mario_env import make_mario_env

# Fixed the placeholder path to use the established Drive directory
SAVE_DIR = '/content/drive/MyDrive/mario_rl_checkpoints/'
env = DummyVecEnv([lambda: make_mario_env()])
env = VecMonitor(env)
os.makedirs(SAVE_DIR, exist_ok=True)

def checkpoint_cb(local_vars, _globals):
    """Save a checkpoint every 50k env-steps to SAVE_DIR."""
    self = local_vars['self']
    n = self._num_timesteps
    if n > 0 and n % 50_000 < self.num_envs:
        path = f"{SAVE_DIR}/mario_dqn_step{n}.pt"
        self.save(path)
        if self.verbose >= 1:
            print(f"[checkpoint] saved {path}")

obs_shape = env.observation_space.shape
num_actions = env.action_space.n

model_dqn = DQNAgent(
    env,
    obs_shape,
    num_actions,
    learning_rate=5e-5,
    buffer_size=200_000,
    learning_starts=10_000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=5_000,
    train_freq=4,
    exploration_fraction=0.5,
    exploration_final_eps=0.1,
    max_grad_norm=1.0,
    tensorboard_log="./mario_tensorboard_dqn/",
    tb_log_name="DQN",
    verbose=1,
)

# NOTE: If you still get a Sympy error, go to Runtime -> Restart session,
# then run the environment setup cells and come back here.
model_dqn.learn(total_timesteps=3_000_000, callback=checkpoint_cb)
model_dqn.save(f"{SAVE_DIR}/mario_dqn_3M")