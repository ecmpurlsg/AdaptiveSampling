import os
import sys
import math
import argparse
import random
import csv
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch

import matplotlib.pyplot as plt  # not strictly needed, but handy if you later add plots

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sim.sim_complex import Simulation
from agents.ppo import PPO
from agents.ppo_lstm import PPO_LSTM
from agents.td3_original import TD3
from learning.train_complex import extract_nav_features
from learning.train_complex import compute_reward


def load_rl_agent(model_path: str,
                  algo: str,
                  agent_type: str = "ppo",
                  device: Optional[torch.device] = None) -> Tuple[Any, int]:
    """Load a trained RL agent (PPO / PPO_LSTM / TD3) and return (agent, input_dim)."""
    tmp_sim = Simulation(corridor_width=3.0,
                         corridor_length=10.0,
                         door_side='right',
                         num_people=5,
                         spawn_interval = random.uniform(0.5, 2.0),
                         people_speeds=[random.uniform(0.6, 1.0) for _ in range(10)],
                         door_spacing_m=5.0)
    # Ensure the feature dimension matches the evaluation algorithm mode.
    configure_nav(tmp_sim, algo)
    _ = tmp_sim.step(1 / 60.0)
    input_dim = int(len(extract_nav_features(tmp_sim)))

    num_actions = 1  # single continuous offset in [-1, 1]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent_type = str(agent_type).lower().strip()
    if agent_type == "td3":
        agent = TD3(
            obs_dim=input_dim,
            act_dim=num_actions,
            act_limit=1.0,
            gamma=0.99,
            polyak=0.995,
            pi_lr=3e-4,
            q_lr=3e-4,
            policy_delay=2,
            act_noise=0.1,
            target_noise=0.2,
            noise_clip=0.5,
        )
    elif agent_type == "ppo_lstm":
        agent = PPO_LSTM(
            state_dim=input_dim,
            action_dim=num_actions,
            lr_actor=1e-3,
            lr_critic=1e-3,
            gamma=0.99,
            K_epochs=10,
            eps_clip=0.2,
            has_continuous_action_space=True,
            action_std_init=0.4,
        )
    else:
        agent = PPO(
            state_dim=input_dim,
            action_dim=num_actions,
            lr_actor=1e-3,
            lr_critic=1e-3,
            gamma=0.99,
            K_epochs=10,
            eps_clip=0.2,
            has_continuous_action_space=True,
            action_std_init=0.4,
        )

    agent.load(model_path)
    return agent, input_dim


def configure_nav(sim: Simulation, algo: str) -> None:
    """Select local planner (TS-DWA or DWA) for the robot."""
    from algo.dwa import DWA
    from algo.ts_dwa import TSDWA

    algo = algo.lower()
    if algo == "dwa":
        # Plain DWA
        sim.robot.nav = DWA(
            position=sim.robot.position,
            velocity=sim.robot.velocity,
            max_speed=sim.robot.max_speed,
            goal=tuple(sim.robot.goal),
            radius=sim.robot.radius,
            corridor_bounds=sim.robot.corridor_bounds,
        )
        # Provide door information just like in Simulation.__init__
        if hasattr(sim.robot.nav, "set_door_info"):
            sim.robot.nav.set_door_info(sim.get_door_position(), sim.door_side)
        sim.robot.nav_type = "dwa"
        sim.robot.nav.door_aware_sampling = False
    elif algo == "dwa_door_aware":
        # DWA with door-aware sampling enabled
        sim.robot.nav = DWA(
            position=sim.robot.position,
            velocity=sim.robot.velocity,
            max_speed=sim.robot.max_speed,
            goal=tuple(sim.robot.goal),
            radius=sim.robot.radius,
            corridor_bounds=sim.robot.corridor_bounds,
        )
        if hasattr(sim.robot.nav, "set_door_info"):
            sim.robot.nav.set_door_info(sim.get_door_position(), sim.door_side)
        sim.robot.nav.door_aware_sampling = True
        sim.robot.nav_type = "dwa_door_aware"
    else:
        # Default TS-DWA (also used for PPO + TS-DWA)
        sim.robot.nav = TSDWA(
            position=sim.robot.position,
            velocity=sim.robot.velocity,
            max_speed=sim.robot.max_speed,
            goal=tuple(sim.robot.goal),
            radius=sim.robot.radius,
            corridor_bounds=sim.robot.corridor_bounds,
        )
        sim.robot.nav_type = "ts_dwa"

    # Ensure the local planner knows the goal
    if hasattr(sim.robot.nav, "set_goal") and sim.robot.goal is not None:
        sim.robot.nav.set_goal(tuple(sim.robot.goal))


def run_evaluation(
    algo: str,
    episodes: int,
    max_steps: int,
    render: bool,
    model_path: str,
    action_select_interval: int,
    base_seed: int,
    output_csv: Optional[str] = None,
    macro_step: bool = False,
    w_max_min: float = -0.2 * math.pi,
    w_max_max: float = math.pi,
) -> str:
    """Run multiple evaluation episodes and save per-episode stats to a CSV."""

    algo = algo.lower()
    planner_mode: Optional[str] = None
    agent_kind: Optional[str] = None  # "ppo" | "ppo_lstm" | "td3"

    if algo in ("ppo_ts_dwa", "ppo_lstm_ts_dwa", "td3_ts_dwa"):
        planner_mode = "ts_dwa"
    elif algo in ("ppo_dwa_door_aware", "ppo_lstm_dwa_door_aware", "td3_dwa_door_aware"):
        planner_mode = "dwa"

    if algo.startswith("ppo_lstm_"):
        agent_kind = "ppo_lstm"
    elif algo.startswith("ppo_"):
        agent_kind = "ppo"
    elif algo.startswith("td3_"):
        agent_kind = "td3"

    use_rl = (planner_mode is not None) and (agent_kind is not None)
    use_ppo = agent_kind in ("ppo", "ppo_lstm")
    use_lstm = agent_kind == "ppo_lstm"
    use_td3 = agent_kind == "td3"

    agent: Optional[Any] = None
    if use_rl:
        agent, _ = load_rl_agent(model_path, algo=planner_mode, agent_type=agent_kind)

    if render:
        import pygame
        pygame.init()
        width, height = 1000, 400
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"Algorithm Evaluation: {algo.upper()}")
        clock = pygame.time.Clock()
    else:
        screen = None
        clock = None

    # Prepare output CSV only if not rendering
    writer: Optional[csv.DictWriter] = None
    if not render:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "simulation_data")
        os.makedirs(data_dir, exist_ok=True)
        if output_csv is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = os.path.join(data_dir, f"algo_eval_{algo}_{ts}")

        fieldnames = [
            "episode",
            "seed",
            "algo",
            "return",
            "steps",
            "collisions",
            "overlap_free_pct",
            "overlap_person_pct",
            "overlap_door_pct",
            "overlap_both_pct",
            "avg_abs_offset",
            "max_abs_offset",
            "avg_door_dy",
        ]

        f = open(output_csv, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    try:
        for ep in range(episodes):
            # Per-episode seed
            seed = base_seed + ep
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            corridor_width_ep = 2.5#random.uniform(2.0, 4.0)
            door_num = 1  # Use 3 doors as specified
            sim = Simulation(
                corridor_width=float(corridor_width_ep),
                door_num=door_num,
                num_people = random.randint(2, 4),
                people_speeds=[random.uniform(0.6, 1.0) for _ in range(10)],
                spawn_interval = random.uniform(0.5, 2.0),
                door_spacing_m=5.0,
            )
            configure_nav(sim, planner_mode if use_rl else algo)

            # Get closest door for mirroring (will be updated each step)
            closest_door_pos, closest_door_side = sim.get_closest_door()
            should_mirror = (closest_door_side == 'left')

            # Warm-up step to initialize internal state
            _, _, _ = sim.step(1 / 60.0)

            episode_return = 0.0
            overlap_counts = {"none": 0, "person": 0, "door": 0, "both": 0}
            # TS-DWA: alpha values in meters (for Gaussian bump) ; DWA-door-aware: w_max in rad/s
            offset_history: List[float] = []
            door_dy_history: List[float] = []

            prev_offset = None
            lstm_hidden = None
            if use_ppo and use_lstm:
                # Reset LSTM hidden state per episode (evaluation-time recurrence)
                try:
                    lstm_hidden = agent.policy_old.init_hidden(batch_size=1)
                except Exception:
                    lstm_hidden = None

            # Stuck detection: track best distance and steps without progress
            # Using same defaults as train_v9.py
            stuck_detection_enabled = True
            stuck_threshold_steps = 400
            stuck_min_progress = 0.005
            stuck_progress_reset_steps = 10
            stuck_no_progress_window = 200
            
            robot_pos = sim.robot.position
            goal_pos = sim.robot.goal
            prev_dist = float(np.linalg.norm(goal_pos - robot_pos))
            best_dist = prev_dist
            steps_without_progress = 0
            consecutive_progress_steps = 0
            steps_since_last_progress = 0
            stuck_terminated = False

            t = 0
            while t < max_steps:
                dt = (clock.tick(60) / 1000.0) if render else (1 / 60.0)

                # Handle events
                if render:
                    import pygame
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return output_csv

                # Decision boundary: compute features and (optionally) choose a new action.
                # Update closest door each step (robot position changes)
                closest_door_pos, closest_door_side = sim.get_closest_door()
                should_mirror = (closest_door_side == 'left')
                
                feat = extract_nav_features(sim)
                if use_rl:
                    is_decision = (t % action_select_interval == 0) or (prev_offset is None) or bool(macro_step)
                    if is_decision:
                        if use_td3:
                            if agent is None:
                                raise RuntimeError("TD3 agent was not loaded.")
                            a = agent.select_action(feat, noise_scale=0.0)  # Deterministic evaluation
                            prev_offset = max(-1.0, min(1.0, float(np.asarray(a).reshape(-1)[0])))
                        elif use_lstm:
                            # Recurrent inference: sample from policy_old and advance hidden ONCE per decision.
                            st = torch.as_tensor(feat, dtype=torch.float32, device=next(agent.policy_old.parameters()).device)
                            with torch.no_grad():
                                if lstm_hidden is None:
                                    lstm_hidden = agent.policy_old.init_hidden(batch_size=1)
                                action_t, _, _, lstm_hidden = agent.policy_old.act(st, lstm_hidden)
                            prev_offset = max(-1.0, min(1.0, float(action_t.view(-1)[0].item())))
                        else:
                            offset_norm = agent.select_action(feat)
                            prev_offset = max(-1.0, min(1.0, float(offset_norm[0])))
                    else:
                        # Old (non-macro) action-hold behavior: still advance LSTM hidden each step.
                        if use_lstm:
                            st = torch.as_tensor(feat, dtype=torch.float32, device=next(agent.policy_old.parameters()).device)
                            with torch.no_grad():
                                if lstm_hidden is None:
                                    lstm_hidden = agent.policy_old.init_hidden(batch_size=1)
                                _, lstm_hidden = agent.policy_old._forward_seq(st.view(1, -1), lstm_hidden)

                    # Apply RL action with mirroring if door is on left
                    if planner_mode == "ts_dwa":
                        # Map action from [-1, 1] to [0, -corridor_width/1.25] for alpha
                        # Then pass alpha=-alpha to the function (so final alpha is in [0, corridor_width/1.25])
                        action_clamped = max(-1.0, min(1.0, float(prev_offset)))
                        #alpha_raw = (action_clamped + 1.0) / 2.0 * (-(sim.corridor_width / (1.25)))
                        if should_mirror:
                            # Mirror the action: flip alpha and swap weights
                            alpha_raw = (action_clamped + 1.0) / 2.0 * (-(sim.corridor_width / (4.25)))
                            alpha_raw = -alpha_raw
                            sim.robot.nav.left_weight = 10
                            sim.robot.nav.right_weight = 2
                        else:
                            alpha_raw = (action_clamped + 1.0) / 2.0 * (-(sim.corridor_width / (1.75)))
                            sim.robot.nav.left_weight = 2
                            sim.robot.nav.right_weight = 10
                        sim.robot.add_gaussian_bump_to_path(alpha=alpha_raw, sigma=2.0)
                        offset = 0.0  # No offset used anymore
                        offset_history.append(abs(float(alpha_raw)))  # Track the alpha value instead
                    else:
                        # For dwa we control the DWA angular sampling limit directly:
                        # map a clamped action a in [-1.0, 1.0] to nav.w_max in [w_max_min, w_max_max].
                        if hasattr(sim.robot, "nav") and hasattr(sim.robot.nav, "w_max"):
                            # Calculate the value using your existing w_max bounds
                            # w_max_min = -0.2 * pi, w_max_max = pi
                            a = max(-1.0, min(1.0, float(prev_offset)))
                            a01 = (float(a) + 1.0) * 0.5 
                            val = float(w_max_min + a01 * (w_max_max - w_max_min))

                            if should_mirror:
                                # Mirroring: Use the negative value for w_min
                                # If a01 is 0, val is -0.2*pi -> w_min becomes 0.2*pi
                                sim.robot.nav.w_min = -val
                                
                                # Reset w_max to default so the window is [0.2*pi, pi]
                                sim.robot.nav.w_max = math.pi
                                
                                offset_history.append(-val)
                            else:
                                # Standard: Use the value for w_max
                                # If a01 is 0, w_max becomes -0.2*pi
                                sim.robot.nav.w_max = val
                                
                                # Reset w_min to default so the window is [-pi, -0.2*pi]
                                sim.robot.nav.w_min = -math.pi
                                
                                offset_history.append(val)
                        offset = 0.0
                        
                else:
                    offset = 0.0

                # Execute either one sim step or N micro-steps (macro-step).
                inner_steps = action_select_interval if (macro_step and action_select_interval > 1) else 1
                done = False
                reward_sum = 0.0
                last_info: Dict[str, Any] = {}
                for _ in range(inner_steps):
                    if t >= max_steps:
                        break
                    
                    # # For dwa_door_aware: adjust w_max based on heading
                    # heading = float(sim.robot.nav.orientation)
                    # if (-math.pi < heading < -math.pi/3) or (math.pi/3 < heading < math.pi):
                    #     sim.robot.nav.w_max = math.pi
                    # print(f"heading: {sim.robot.nav.orientation}")  
                    # print(f"w_max: {sim.robot.nav.w_max}")

                    # if sim.robot.goal[0] - sim.robot.position[0] < 2.0:
                    #     sim.robot.nav.w_max = math.pi

                    _, _, done = sim.step(dt)

                    old_dist = prev_dist
                    reward, prev_dist, info = compute_reward(sim, progress_prev_dist=prev_dist, offset=offset)
                    reward_sum += float(reward)
                    last_info = info
                    
                    # Calculate and print each reward component separately (matching train_v10.py)
                    progress = old_dist - prev_dist
                    progress_reward = 10.0 * (progress ** 2)
                    
                    # Overlap reward component
                    overlap_type = info.get('overlap_type', 'none')
                    if overlap_type == 'none':
                        overlap_reward = -0.2
                    elif overlap_type == 'person':
                        overlap_reward = -2.0
                    elif overlap_type == 'door':
                        overlap_reward = -0.2
                    elif overlap_type == 'both':
                        overlap_reward = -2.0
                    else:
                        overlap_reward = -0.2
                    
                    # Wall proximity penalty
                    wall_penalty = 0.0
                    if hasattr(sim.robot, 'corridor_bounds'):
                        bounds = sim.robot.corridor_bounds
                        y_min = float(bounds['y_min'])
                        y_max = float(bounds['y_max'])
                        y = float(sim.robot.position[1])
                        dist_left = max(0.0, y - y_min)
                        dist_right = max(0.0, y_max - y)
                        if dist_left < 0.4 or dist_right < 0.4:
                            wall_penalty = -0.5
                    
                    # Backward heading penalty
                    heading_penalty = 0.0
                    if hasattr(sim.robot, 'nav') and hasattr(sim.robot.nav, 'orientation'):
                        heading = float(sim.robot.nav.orientation)
                        if (-math.pi < heading < -math.pi/3) or (math.pi/3 < heading < math.pi):
                            heading_penalty = -1.0
                    
                    total_reward = progress_reward + overlap_reward + wall_penalty + heading_penalty
                    
                    # print(f"  Step {t} | Total: {total_reward:.4f} | "
                    #       f"Progress: {progress_reward:.4f} | "
                    #       f"Overlap({overlap_type}): {overlap_reward:.4f} | "
                    #       f"Wall: {wall_penalty:.4f} | "
                    #       f"Heading: {heading_penalty:.4f} | "
                    #       f"Distance: {prev_dist:.3f}m")

                    # Stuck detection: check if robot made progress (same logic as train_v9.py)
                    if stuck_detection_enabled:
                        if prev_dist < best_dist - stuck_min_progress:
                            # Made progress: reset counters and update best distance
                            best_dist = prev_dist
                            consecutive_progress_steps += 1
                            steps_since_last_progress = 0
                            # Reset stuck counter after sustained progress
                            if consecutive_progress_steps >= stuck_progress_reset_steps:
                                steps_without_progress = 0
                        else:
                            # No progress this step
                            steps_since_last_progress += 1
                            consecutive_progress_steps = 0
                            # Only count as "stuck" if we haven't made progress for a while (allows oscillations)
                            if steps_since_last_progress >= stuck_no_progress_window:
                                steps_without_progress += 1
                                # Print when stuck threshold is reached
                                if steps_without_progress >= stuck_threshold_steps:
                                    print(f"[STUCK] Episode {ep+1} | Step {t} | "
                                          f"Distance: {prev_dist:.3f}m | "
                                          f"Best distance: {best_dist:.3f}m | "
                                          f"Steps without progress: {steps_without_progress}")
                                    stuck_terminated = True

                    overlap_counts[info["overlap_type"]] += 1
                    if np.isfinite(info.get("door_dy", float("nan"))):
                        door_dy_history.append(float(info["door_dy"]))

                    if render:
                        screen.fill((255, 255, 255))
                        sim.draw_v0(screen, state_input=feat)
                        import pygame
                        pygame.display.flip()

                    t += 1
                    if done:
                        break

                episode_return += float(reward_sum)
                if done:
                    break

            total_steps = t + 1
            overlap_pct = {k: 100.0 * v / total_steps for k, v in overlap_counts.items()}
            collisions = getattr(sim, "collision_count", 0)
            avg_abs_offset = float(np.mean(offset_history)) if offset_history else 0.0
            max_abs_offset = float(np.max(offset_history)) if offset_history else 0.0
            avg_door_dy = float(np.mean(door_dy_history)) if door_dy_history else float("nan")

            print(f"[{algo.upper()}] Episode {ep+1}/{episodes} | "
                  f"Return: {episode_return:.2f} | Steps: {total_steps} | "
                  f"Collisions: {collisions}")
            print(f"  Overlaps - Free: {overlap_pct['none']:.1f}% | "
                  f"Person: {overlap_pct['person']:.1f}% | "
                  f"Door: {overlap_pct['door']:.1f}% | "
                  f"Both: {overlap_pct['both']:.1f}%")
            if planner_mode == "ts_dwa":
                max_alpha_possible = sim.corridor_width / 1.25
                print(f"  Alpha - Avg: {avg_abs_offset:.3f}m | Max: {max_abs_offset:.3f}m | (range: [0, {max_alpha_possible:.3f}]m)")
            elif planner_mode == "dwa":
                print(f"  w_max - Avg: {avg_abs_offset:.3f} rad/s | Max: {max_abs_offset:.3f} rad/s")

            if writer is not None:
                writer.writerow({
                    "episode": ep + 1,
                    "seed": seed,
                    "algo": algo,
                    "return": episode_return,
                    "steps": total_steps,
                    "collisions": collisions,
                    "overlap_free_pct": overlap_pct["none"],
                    "overlap_person_pct": overlap_pct["person"],
                    "overlap_door_pct": overlap_pct["door"],
                    "overlap_both_pct": overlap_pct["both"],
                    "avg_abs_offset": avg_abs_offset,
                    "max_abs_offset": max_abs_offset,
                    "avg_door_dy": avg_door_dy,
                })
    finally:
        if writer is not None:
            f.close()

    return output_csv if not render else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate navigation algorithms (PPO, TS-DWA, DWA) and log episode stats to CSV"
    )
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo_ts_dwa", "ppo_dwa_door_aware",
                                 "ppo_lstm_ts_dwa", "ppo_lstm_dwa_door_aware",
                                 "td3_ts_dwa", "td3_dwa_door_aware",
                                 "ts_dwa", "dwa", "dwa_door_aware"],
                        help="Algorithm to evaluate: ppo_ts_dwa (PPO controls TS-DWA agent_offset), "
                             "ppo_dwa_door_aware (PPO controls DWA nav.w_max), "
                             "ppo_lstm_ts_dwa / ppo_lstm_dwa_door_aware (same but with recurrent PPO-LSTM), "
                             "td3_ts_dwa / td3_dwa_door_aware (same control but with TD3), "
                             "or baselines: ts_dwa, dwa, dwa_door_aware.")
    parser.add_argument("--w-max-min", type=float, default=-0.2 * math.pi,
                        help="(dwa) Minimum DWA nav.w_max (rad/s) when controlled by RL.")
    parser.add_argument("--w-max-max", type=float, default=math.pi,
                        help="(dwa) Maximum DWA nav.w_max (rad/s) when controlled by RL.")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=3000,
                        help="Maximum steps per episode.")
    parser.add_argument("--model", type=str, default="checkpoints/theta_qnet.pt",
                        help="Path to trained RL model checkpoint (PPO/PPO_LSTM/TD3 depending on --algo).")
    parser.add_argument("--action-select-interval", type=int, default=5,
                        help="Select a new PPO action every N steps (ppo only).")
    parser.add_argument("--macro-step", action="store_true",
                        help="If set, treat action_select_interval as a true frame-skip: apply one action for N sim steps "
                             "and accumulate reward across those N steps.")
    parser.add_argument("--seed", type=int, default=123,
                        help="Base random seed; each episode uses seed+episode_index.")
    parser.add_argument("--render", action="store_true",
                        help="Enable pygame rendering.")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Optional path for the output CSV (defaults to simulation_data/algo_eval_*.csv).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = run_evaluation(
        algo=args.algo,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        model_path=args.model,
        action_select_interval=args.action_select_interval,
        base_seed=args.seed,
        output_csv=args.output_csv,
        macro_step=bool(args.macro_step),
        w_max_min=args.w_max_min,
        w_max_max=args.w_max_max,
    )
    print(f"\nSaved evaluation results to: {csv_path}")


if __name__ == "__main__":
    main()


