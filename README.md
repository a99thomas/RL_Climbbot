# RL Climbing Robot

This repository contains the reinforcement learning environment, MuJoCo simulation assets, training scripts, and visualization tools for developing a controller for a two-arm climbing robot. The project uses **MuJoCo**, **Gymnasium**, and **Stable-Baselines3 (PPO)** to train end-effector reaching and grasping behaviors in a contact-rich environment.

---

## 🚀 Features

- Custom Gymnasium environment (`ClimbBotEnv`) with IK-based end-effector control  
- MuJoCo simulation of climbing robot hardware (meshes, telescoping arm joints, handholds)  
- PPO training pipeline with TensorBoard logging, evaluation, and checkpointing  
- Policy visualization tool with real-time MuJoCo viewer  
- Modular experiment directories for running multiple training configurations  

---

## 📦 Installation

It is recommended to use a Python virtual environment (Conda or `venv`).  
Python **3.11** is confirmed to work well.

### 1. Create and activate a virtual environment

```bash
# Using conda
conda create -n climbbot python=3.11
conda activate climbbot

# Using venv
python3.11 -m venv climbbot_env
source climbbot_env/bin/activate
```

### 2. Install required packages

```bash
pip install mujoco gymnasium stable-baselines3 tensorboard numpy opencv-python matplotlib tqdm
```

---

## 🧠 Training a Policy

Training scripts are located inside experiment directories (e.g., `test_5/`).

To begin training:

```bash
cd test_5
python train_5_evals.py --xml "/path/to/mujoco_scene.xml"
```

### Common arguments

| Argument         | Description                                      |
|------------------|--------------------------------------------------|
| `--xml`          | Path to the MuJoCo scene XML file                |
| `--timesteps`    | Total training steps (default: 10M)              |
| `--eval-freq`    | Evaluation interval                              |
| `--control-speed`| MuJoCo physics steps per RL action               |
| `--render-train` | Enable viewer during training (slow)             |

### Example

```bash
python train_5_evals.py     --xml "./assets/scene.xml"     --timesteps 5000000     --eval-freq 100000
```

Training logs and TensorBoard summaries will appear under the `./runs/` directory.

---

## 👀 Viewing a Trained Policy

To visualize a trained PPO model in the MuJoCo viewer:

```bash
python view_model.py     --model "/path/to/policy.zip"     --vecnormalize "/path/to/vecnormalize.pkl"     --xml "/path/to/mujoco_scene.xml"
```

### Why `VecNormalize` is required?

Stable-Baselines3 stores observation and reward normalization statistics in a `.pkl` file.  
This file must be loaded **together with the policy** to reproduce correct behavior.

---

