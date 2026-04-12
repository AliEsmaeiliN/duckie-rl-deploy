# 🐥 duckie-rl-deploy: Sim2Real Inference
### **Deployment Suite for DB21J Duckiebots**

---

## 🎯 **Core Objective**
This repository is the dedicated **inference engine** for transferring SAC/TD3 Reinforcement Learning policies from simulation to the physical **Duckietown DB21J**. It bridges the "Reality Gap" by precisely mirroring the simulation's data processing pipeline.

---

## 🏗️ **Repository Architecture**
Standardized Duckietown structure for hardware-aware builds:

* **`assets/`**: Model weight storage (`sac_Final.cleanrl_model`).
* **`packages/`**: Core inference modules:
    * **`solution.py`**: ROS Node managing Camera Subscriptions and Wheel Command publishing.
    * **`agent.py`**: Vision pipeline (cropping/resizing) and 4-frame temporal stacking.
    * **`models.py`**: ImpalaCNN and Actor network definitions.
    * **`debug_bot.py`**: Remote telemetry for real-time visualization on a laptop.
* **`launchers/`**: Entry point scripts for the `dts` runtime.

---

## 🛠️ **Sim2Real Alignment Features**
Designed to ensure simulation-to-reality parity:

* **Vision Fidelity**: Replicates the 25% top-crop and 84x84 resize from training wrappers.
* **Temporal Consistency**: 4-frame `deque` buffer matching the training `FrameStackObservation`.
* **Kinematic Mapping**: Translates RL outputs $[v, \omega]$ into physical PWM duty cycles $[u_l, u_r]$ for DB21J motors.

---

## 🚀 **Quick Start Commands**

### **1. Build for ARM64**
Perform cross-compilation on your laptop for the Jetson Nano:
```bash
dts devel build -f
```

### **2. Deploy to Robot**
Run the container on the physical robot with hardware permissions:
```bash
dts devel run -H <robot_name>.local
```

---

## 📦 **Technical Dependencies**
Optimized Duckietown binaries for **daffy**:
* **`torch==2.4.1`**
* **`torchvision==0.19.1`**
* **`numpy==1.23.5`**
* **`opencv-python-headless==4.10.0.84`**