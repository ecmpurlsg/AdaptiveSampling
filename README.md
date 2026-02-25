# AdaptiveSampling
PredictiveDWA: Adaptive Sampling for Robot Navigation

This repository implements an adaptive sampling modulation layer for robot navigation, integrating the Dynamic Window Approach (DWA) with Reinforcement Learning (TD3). The system is designed to improve navigation performance in challenging social environments such as doorways and corridors.
Features

    Adaptive Sampling: Learned modulation for DWA sampling spaces.

    RL Integration: TD3-based actor-critic architecture for dynamic action selection.

    Benchmarking: Comparisons between standard DWA, Time Scaling (TS), and learned variants.

    Experiment Tracking: Native support for Weights & Biases (WandB).

Evaluation Scenarios

The algorithms are evaluated across three primary configurations:

    Door-Proximal Emergence: Pedestrians emerging in close proximity to doorways.

    Sequential Enclosure: Sequential encounters creating bottlenecks in narrow spaces.

    Wall-Adjacent Obstacles: Navigation through corridors with obstacles positioned near walls.

Usage
Testing and Visualization

To run the evaluation scripts with a GUI, include the --render flag.

Standard DWA
Bash

python3 src/learning/test_algorithm.py --algo dwa

TD3-DWA (Learned)
Bash

python3 src/learning/test_algorithm.py --algo td3_dwa_door_aware --action-select-interval 1 --model checkpoints/td3_dwa.pt

Time Scaling (TS)
Bash

python3 src/learning/test_algorithm.py --algo ts_dwa

TD3-TS
Bash

python3 src/learning/test_algorithm.py --algo td3_ts_dwa --action-select-interval 60 --model checkpoints/td3_ts_dwa.pt

Training

Models are trained using a TD3 policy with twin Q-networks. Ensure you have configured your WandB environment before running training scripts.

Train TD3-DWA
Bash

python3 src/learning/train_v12.py --algo dwa --agent td3 --use-wandb --wandb-project PredictiveDWA

Train TD3-TS
Bash

python3 src/learning/train_v12.py --algo ts_dwa --agent td3 --use-wandb --wandb-project PredictiveDWA --action-select-interval 60

Implementation Details

The TD3 policy uses an actor MLP (dobs​→256→256→dact​) with ReLU activations and a scaled tanh output. The critics utilize twin networks receiving concatenated state-action inputs to provide stable Q-value estimates.
