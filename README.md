# Adaptive Sampling Modulation Layer for Robot Navigation

This repository provides the implementation for an Adaptive Sampling Modulation Layer that enhances the Dynamic Window Approach (DWA) and the Targeted Sampling DWA using reinforcement learning. The system is designed to improve navigation efficiency and safety in social environments like corridors and doorways.



## Features
- **Adaptive Sampling:** A learned layer that dynamically modulates DWA or TS-DWA sampling spaces.
- **TD3 Integration:** Robust actor-critic architecture for continuous action selection.
- **Social Navigation Scenarios:** Evaluated against complex pedestrian behaviors.
- **Experiment Tracking:** Native integration with Weights & Biases (WandB).

## Repository Structure
- `src/learning/`: Core logic for training and testing algorithms.
- `checkpoints/`: Storage for pre-trained TD3 models (`.pt` files).
- `figures/`: Documentation and visualization assets.

## Usage

### 1. Evaluation and Testing
To run evaluations, use the `test.py` script. Append the `--render` flag to visualize the robot's navigation in the GUI.

**Standard DWA (Baseline)**
```bash
python3 src/learning/test.py --algo dwa --render
The TD3 policy uses an actor MLP (dobs​→256→256→dact​) with ReLU activations and a scaled tanh output. The critics utilize twin networks receiving concatenated state-action inputs to provide stable Q-value estimates.
