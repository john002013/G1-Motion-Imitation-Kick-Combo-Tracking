# G1 Motion Imitation — Kick Combo Tracking

Motion-imitation policy for a Unitree G1 humanoid that learns to reproduce a reference kick-combo motion clip, trained with [`mjlab`](https://github.com/mujocolab/mjlab)'s motion-tracking pipeline (built on the [BeyondMimic](https://github.com/mujocolab/mjlab) approach), using reference motion data sourced from Jason Peng's [MimicKit](https://github.com/xbpeng/MimicKit).

Rather than tracking a commanded velocity (see the companion [locomotion RL repo](#)), this policy tracks a specific reference trajectory — full-body joint and root motion from a captured/authored kick-combination clip — learning to reproduce the motion on the physical G1 morphology while staying dynamically balanced.

## Why this project

Motion imitation is the complementary half of the locomotion story: instead of a general-purpose velocity controller, this trains skill-specific behavior directly from reference motion data, which is the standard route to expressive, non-trivial humanoid behaviors (kicks, dances, martial-arts sequences) that would be hard to hand-craft as a reward function alone. It's also a good testbed for thinking about how imitation-learned skills could later be selected/composed on top of a general locomotion policy.

## Pipeline overview

```
MimicKit reference motion (.pkl)
        │  pkl_to_csv.py  (from mujocolab/g1_spinkick_example)
        ▼
   Motion CSV  (with start/end transition padding)
        │  mjlab.scripts.csv_to_npz
        ▼
   Motion NPZ  →  pushed to W&B Registry as an artifact
        │
        ▼
  mjlab.scripts.train  Mjlab-Tracking-Flat-Unitree-G1
        │  (PPO via rsl-rl-lib, reads motion from --registry-name)
        ▼
   Trained tracking policy  (checkpointed + logged to W&B)
```

## Data source

- Reference motion: `g1_kick_combo.pkl`, from the MimicKit motion library ([xbpeng/MimicKit](https://github.com/xbpeng/MimicKit), Xue Bin "Jason" Peng)
- Conversion tooling: `pkl_to_csv.py` from [`mujocolab/g1_spinkick_example`](https://github.com/mujocolab/g1_spinkick_example), which adapts MimicKit clips onto the G1's morphology and adds smooth start/end transitions to and from a safe standing pose
- Conversion flags used: `--add-start-transition --add-end-transition --transition-duration 0.5`
- Frame rate: retimed from 120 fps (source clip) → 50 fps (mjlab sim rate) via `mjlab.scripts.csv_to_npz`

> ⚠️ MimicKit clips are for research/educational use — check the source repo's license and terms before redistributing converted motion data.

## Training setup

| | |
|---|---|
| Task | `Mjlab-Tracking-Flat-Unitree-G1` |
| Algorithm | PPO (via `rsl-rl-lib`) |
| Motion artifact | `csv_to_npz/g1_kick:v0` (W&B Registry) |
| Parallel environments | 1024 |
| Total training | 3000 iterations, resumed for 2000 more (5000 total) |
| Checkpoint interval | every 50 iterations |
| Logging | Weights & Biases (`mjlab` project) |
| Compute | Kaggle notebook, NVIDIA Tesla T4 (15 GB) |
| Python | 3.12 |
| Key packages | `mjlab==1.3.0`, `mujoco==3.8.1.dev`, `mujoco-warp==3.8.0.2`, `torch==2.9.0`, `rsl-rl-lib==5.2.0`, `wandb==0.22.3` |

### Result snapshot (iteration ~4998/5000)
- Mean reward: **18.67**, mean episode length: **422**
- Reward breakdown: body position tracking was the strongest term (`motion_body_pos`: 0.83), root orientation and body orientation tracked well (`motion_global_root_ori`: 0.36, `motion_body_ori`: 0.59); joint-velocity error (`error_joint_vel`: 12.69) was the largest residual error, suggesting the policy nails the pose shape more than fine velocity timing — a reasonable target for further reward tuning.

## Reproducing

### 1. Install mjlab
```bash
git clone https://github.com/mujocolab/mjlab.git
cd mjlab
uv sync
```

### 2. Get the reference motion and convert it
```bash
git clone https://github.com/xbpeng/MimicKit
git clone https://github.com/mujocolab/g1_spinkick_example
cd g1_spinkick_example

uv run pkl_to_csv.py \
    --pkl-file <path-to>/g1_kick_combo.pkl \
    --csv-file g1_kick.csv \
    --add-start-transition \
    --add-end-transition \
    --transition-duration 0.5
```

```bash
cd ../mjlab
uv run python -m mjlab.scripts.csv_to_npz \
    --input-file  ../g1_spinkick_example/g1_kick.csv \
    --output-name g1_kick \
    --input-fps   120 \
    --output-fps  50
```

This uploads the motion as a versioned artifact to your W&B Registry (e.g. `<entity>/csv_to_npz/g1_kick:v0`), which the training task pulls from via `--registry-name`.

### 3. Train
```bash
export WANDB_ENTITY="<your-wandb-entity>"
export WANDB_API_KEY="<your-key>"
export MUJOCO_GL="egl"

uv run train Mjlab-Tracking-Flat-Unitree-G1 \
    --registry-name <your-entity>/csv_to_npz/g1_kick:v0 \
    --env.scene.num-envs 1024 \
    --agent.max-iterations 3000 \
    --agent.save-interval 50
```

### 4. Resume (if interrupted)
```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 \
    --registry-name <your-entity>/csv_to_npz/g1_kick:v0 \
    --env.scene.num-envs 1024 \
    --agent.max-iterations 2000 \
    --agent.save-interval 50 \
    --agent.resume True \
    --wandb-run-path <your-entity>/mjlab/<run-id>
```

### 5. Evaluate
```bash
uv run play Mjlab-Tracking-Flat-Unitree-G1 \
    --wandb-run-path <your-entity>/mjlab/<run-id>
```

### Notes on running on constrained/free-tier GPUs
Trained on a Kaggle T4 (15 GB VRAM) notebook. Long runs were launched with `nohup` and `start_new_session=True` so training survived notebook cell interruptions and kernel disconnects — worth doing on any free-tier / session-limited compute (Colab, Kaggle) for multi-hour runs.

## Repo structure
```
.
├── README.md
├── notebooks/
│   └── g1_kick_imitation.ipynb   # Kaggle notebook: conversion + training
├── data/
│   ├── g1_kick.csv                # converted motion (or link to source, if licensing restricts redistribution)
│   └── g1_kick.npz
├── configs/
└── checkpoints/                   # gitignored — reference via W&B artifacts
```

## Results

_Add a side-by-side GIF/video of the reference motion vs. the trained policy, and the W&B reward/tracking-error curves here._

W&B run: `<link to your W&B run>`

## Acknowledgments
- [`mjlab`](https://github.com/mujocolab/mjlab) — Kevin Zakka, Qiayuan Liao, Brent Yi, Louis Le Lay, Koushil Sreenath, Pieter Abbeel (UC Berkeley, Sorbonne University)
- [MimicKit](https://github.com/xbpeng/MimicKit) — Xue Bin "Jason" Peng, for the reference motion data and imitation learning research it's built on
- [`g1_spinkick_example`](https://github.com/mujocolab/g1_spinkick_example) — mjlab's example project for adapting MimicKit clips to the G1, whose conversion tooling this project reuses
- Unitree Robotics for the G1 platform

## Author

Babalola John Abidemi — final-year B.Tech. Electronic & Electrical Engineering, LAUTECH; undergraduate researcher.
