## Setup

```bash
conda env create -f environment.yml
conda activate csci1470
```

## Running

**RAM PPO** (uses NES RAM features as observations):
```bash
cd ramPPO
python ram_ppo.py
```

Key options:
- `--total-timesteps 5000000` — how long to train (default: 5M)
- `--num-envs 8` — parallel environments (default: 8)
- `--run-name my_run` — name for TensorBoard logs
- `--load path/to/checkpoint.pt` — resume from a checkpoint
- `--no-tb` — disable TensorBoard logging

**Monitoring training:**
```bash
tensorboard --logdir runs
```

**Resume from checkpoint:**
```bash
python ram_ppo.py --load mario_ram_ppo_final_mario_ram_ppo_v2.pt
```
