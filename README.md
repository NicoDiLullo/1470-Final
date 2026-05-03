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

## General Repository Structure:

# benchmarks
Our benchmarks. benchmarks.py compares RAM and CNN PPOs. DQN also adds DQNs in addition
to the other two.

# cnnPPO
Our CNN PPO implementation. Our best CNN model (~50m steps and 15 hours) is here as well
(bestCNN.pt). Trained on Nico's laptop (M3 Max).

# core
core.py is a set of abstractions from other classes, for extenisibility, and "good" SWE practice (thanks Andy van Dam). 

Also, video recording utils.

# dqn
Our DQN implemntation (and best model). Trained on a T4 on Colab.

# otherOptimizations
Our futile attempts of quantizing and compiling our CNN model to make it more performant. 
Results included in the ipynb.

Also included is to_quant.pt (a copy of bestCNN.pt) that is needed to run it. Originally run on a T4 Colab GPU for fairness; fp16 is not designed for CPU acceleration (I guess the
other way lol).

# plots
Training plots. Had some issues with TensorBoard, so they are missing some data, but the shape is pretty informative nonetheless.

# ramPPO.
Our ramPPO implementation (ram_ppo.py). Also contains our best RAM PPO model (ppo_final.pt).
Trained on Nico's laptop (M3 Max) in ~4 hours for 80 million iterations.

## Other implementation details
Our code is somewhat commented (variably by author, yes; one of us really write like that),and our writeup renders some of this redundant, here is the general gist of each (lifted from our poster). 

# RAM PPO
128-dimensional NES RAM features, encoding Mario's position, velocity, enemy locations, and game state are passed through a two-layer MLP into separate policy and value heads trained with PPO.

Was originally written as an optimization for CNN PPO.

# CNN PPO
Four stacked grayscale frames are processed through a three-layer CNN into a 512-dim shared representation, which feeds separate policy and value heads trained end-to-end with PPO.

We also tried quantizing to various precisions, compiling, custom rewards, entropy tuning, and various vectored training optimizations. 

Quantizing to float16 slightly improved model size at the cost of throughput (see otherOptimizations), quantizing to int8 was a waste of time, and compiling was also pretty useless. 

# DQN
Really high training rewards, really bad 

## Setup

Incomplete, sorry. Have decent docs from RAM PPO but not much else, but they're all just Python scripts that should run pretty easily. Colab Notebooks should run directly from Colab, though you may need to pass models for some.

The only thing that really matters is that if you want to train something, you run it with
```--num-envs 8` (or more, if you can). Prepare to not be able to use your computer for a large number of hours!

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
