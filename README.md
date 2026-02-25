# Adaptive Sampling Modulation Layer for Robot Navigation

This repository provides the implementation for an Adaptive Sampling Modulation Layer that enhances the Dynamic Window Approach (DWA) and the Targeted Sampling DWA using reinforcement learning. The system is designed to improve navigation efficiency and safety in social environments like corridors and doorways.



## Features
- **Adaptive Sampling:** A learned layer that dynamically modulates DWA or TS-DWA sampling spaces.
- **TD3 Integration:** Robust actor-critic architecture for continuous action selection.
- **Social Navigation Scenarios:** Evaluated against complex pedestrian behaviors.
- **Experiment Tracking:** Native integration with Weights & Biases (WandB).

## Repository Structure
- `src/learning/`: Core logic for training and testing algorithms.

## Usage

### 1. Evaluation and Testing
To run evaluations, use the `test.py` script. Append the `--render` flag to visualize the robot's navigation in the GUI.

**Train**
- TD3-DWA: python3 src/learning/train.py --algo dwa --agent td3 --use-wandb --wandb-project PredictiveDWA
- TD3-TS: python3 src/learning/train.py --algo ts_dwa --agent td3 --use-wandb --wandb-project PredictiveDWA --action-select-interval 60

**Test**
- DWA: python3 src/learning/test.py --algo dwa
- TD3-DWA: python3 src/learning/test.py --algo td3_dwa --action-select-interval 1 --model checkpoints/td3_dwa.pt
- TS: python3 src/learning/test.py --algo ts_dwa
- TD3-TS: python3 src/learning/test.py --algo td3_ts_dwa --action-select-interval 60 --model checkpoints/td3_ts_dwa.pt
