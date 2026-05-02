## Important Note
This is not the repository where development for this project was done. This is because no such repository exists. Instead, most of this was done on Google Colab (anything that is a .ipynb file was created and run on Colab), or elsewhere on group member's local machines. 

This repository is an attempt to centralize all our code, and make it reproducible. We have also done some basic abstractions (see core) so that we at least look like comptetent engineering students. 

There are, however, a few important caveats. 

1. We provide an environment configuration file for "easy" reproducibility. This code is reliant on old software, and thus does not run on everyone's computer. For example, Alex's recently purchased laptop doesn't have the necessary backwards compatability in PyTorch to run this code (you can get around this by using the nightly PyTorch, and then will run into Gym issues).
2. For the sake of Nico's GitHub storage limits, most model checkpoints and TensorBoard log files have been omitted, and only final models are included.
3. Everything here should run. It has not been run all the way through since migration (as most things here have runtimes on the order of 10s of hours). 

## Poster
[Here](https://docs.google.com/presentation/d/1q2TjukW4iPlxBO3WBwFGq5fGd6xwkAj9gHBWKH5mZqk/edit?usp=sharing) is a link to our poster. A PDF is also included in the repo (DLFinalPoster.pdf).

## Playthrough Videos
The pptx we presented with playthrough videos can be found [here](https://drive.google.com/drive/folders/1H6Rd_VW72AOp4M2yVPIHqIFJOPvsnYXq?usp=drive_link)

## Setup

```bash
conda env create -f environment.yml
conda activate csci1470
```

## Running

**RAM PPO:**
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
