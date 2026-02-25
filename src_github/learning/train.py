import os
import sys
import math
import random
from collections import deque
import argparse
import time
from typing import Any, Dict, Optional
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sim.sim import Simulation
from agents.ppo import PPO
from agents.ppo_lstm import PPO_LSTM
#from agents.td3 import TD3, ReplayBuffer as TD3ReplayBuffer
from agents.td3_original import TD3, ReplayBuffer as TD3ReplayBuffer
from agents.td3_lstm import TD3_LSTM

# Optional third-party integrations

import wandb  
import optuna  

def _overwrite_last_ppo_action(agent: PPO, state_feat: np.ndarray, executed_action_value: float) -> None:
    """Make PPO's rollout buffer consistent with the action we *actually executed*.

    `PPO.select_action()` samples an action and immediately stores (state, action, logprob, value)
    into the buffer. If we later clamp/transform that action before applying it to the environment,
    PPO will learn from the wrong action -> training becomes unstable and often gets worse.

    This helper recomputes logprob/value under `policy_old` for the executed action and overwrites
    the last buffer entries so PPO updates match the environment interaction.
    """
    # buffer must already have the state appended by select_action()
    if not agent.buffer.states:
        return

    policy_device = next(agent.policy_old.parameters()).device
    state_tensor_no_batch = torch.as_tensor(state_feat, dtype=torch.float32, device=policy_device)
    state_tensor_batch = state_tensor_no_batch.unsqueeze(0)
    action_tensor_batch = torch.tensor([[float(executed_action_value)]], dtype=torch.float32, device=policy_device)

    with torch.no_grad():
        logprob, state_value, _ = agent.policy_old.evaluate(state_tensor_batch, action_tensor_batch)

    # Overwrite the last stored transition
    agent.buffer.states[-1] = state_tensor_no_batch.detach()
    agent.buffer.actions[-1] = action_tensor_batch.squeeze(0).detach()
    agent.buffer.logprobs[-1] = logprob.detach()
    agent.buffer.state_values[-1] = state_value.squeeze(0).detach()

def configure_nav(sim: Simulation, algo: str) -> None:
    """Configure the robot's local planner to match the requested algorithm."""
    algo = str(algo).lower().strip()

    # Use Robot's helper so TS-DWA gets the correct update() signature (global_path).
    if hasattr(sim.robot, "set_nav_type"):
        sim.robot.set_nav_type(algo)
    else:
        sim.robot.nav_type = algo

    # Ensure the new planner gets a valid goal after swapping planners.
    if getattr(sim.robot, "goal", None) is not None and hasattr(sim.robot, "set_goal"):
        sim.robot.set_goal(tuple(sim.robot.goal))
    elif getattr(sim.robot, "goal", None) is not None and hasattr(sim.robot, "nav") and hasattr(sim.robot.nav, "set_goal"):
        sim.robot.nav.set_goal(tuple(sim.robot.goal))

    # Provide door info to DWA variants (used for door-aware sampling angle/distance).
    if hasattr(sim.robot, "nav") and hasattr(sim.robot.nav, "set_door_info") and hasattr(sim, "get_door_position"):
        try:
            sim.robot.nav.set_door_info(sim.get_door_position(), getattr(sim, "door_side", None))
        except Exception:
            pass

    # Explicitly toggle door-aware sampling for these two modes.
    # if hasattr(sim.robot, "nav") and hasattr(sim.robot.nav, "door_aware_sampling"):
    #     sim.robot.nav.door_aware_sampling = (algo == "dwa")

def extract_nav_features(sim) -> np.ndarray:
    robot_pos = np.asarray(sim.robot.position, dtype=float)
    # Get robot heading (yaw)
    robot_yaw = float(sim.robot.nav.orientation) 
    
    # Helper to rotate global dx, dy into robot's local frame
    def to_local(dx_global, dy_global, yaw):
        # Rotation matrix [ cos  sin]
        #                 [-sin  cos]
        c, s = math.cos(yaw), math.sin(yaw)
        x_local = c * dx_global + s * dy_global
        y_local = -s * dx_global + c * dy_global
        return x_local, y_local

    # --- Goal (Local Frame) ---
    goal_dx_g = float(sim.robot.goal[0] - robot_pos[0])
    goal_dy_g = float(sim.robot.goal[1] - robot_pos[1])
    goal_x, goal_y = to_local(goal_dx_g, goal_dy_g, robot_yaw)

    # --- Door (Local Frame) ---
    door_dx_g = float(sim.robot.door_position[0] - robot_pos[0])
    door_dy_g = float(sim.robot.door_position[1] - robot_pos[1])
    door_x, door_y = to_local(door_dx_g, door_dy_g, robot_yaw)

    v = sim.robot.nav.velocity[0]
    w = sim.robot.nav.velocity[1]
    

    # --- People (Local Frame + Validity Mask) ---
    rel_people = []
    if hasattr(sim.robot, 'people'):
        for person in sim.robot.people:
            if getattr(person, 'active', True):
                dx_g = float(person.position[0] - robot_pos[0])
                dy_g = float(person.position[1] - robot_pos[1])
                dist = math.hypot(dx_g, dy_g)
                # Convert to local x (forward), y (left)
                lx, ly = to_local(dx_g, dy_g, robot_yaw)
                rel_people.append((dist, lx, ly))

    rel_people.sort(key=lambda t: t[0]) # Sort by distance
    
    people_feats = []
    for i in range(3):
        if i < len(rel_people):
            # Format: [local_x, local_y, is_present=1.0]
            _, lx, ly = rel_people[i]
            people_feats.extend([lx, ly, 1.0])
        else:
            # Padding: [0.0, 0.0, is_present=0.0]
            # Zeros with a 0-flag is better than (10,10)
            people_feats.extend([0.0, 0.0, 0.0]) 

    # Distances to corridor left/right sides using world y-bounds
    dist_left = 0
    dist_right = 0
    if hasattr(sim.robot, 'corridor_bounds'):
        bounds = sim.robot.corridor_bounds
        y_min = float(bounds['y_min'])
        y_max = float(bounds['y_max'])
        y = float(robot_pos[1])
        dist_left = max(0.0, y - y_min)
        dist_right = max(0.0, y_max - y)

    # --- Walls ---
    # Wall distances are naturally local if they are just left/right distance
    # Keep your existing logic for walls if it works for your corridor alignment
    # ... (rest of your wall logic)

    # Construct final vector
    # Note: We don't strictly need robot_orientation anymore since 
    # everything is relative to it, but it can help if global alignment matters.
    feat = [
        goal_x, goal_y,       # Goal relative to body
        door_x, door_y,       # Door relative to body
    ]
    feat.extend(people_feats) # 3 people (3 vals each: x, y, valid)
    feat.append(dist_left)
    feat.append(dist_right)
    feat.extend([v, w])
    
    return np.asarray(feat, dtype=np.float32)


def extract_nav_features_v2(sim) -> np.ndarray:
    """
    Feature vector from current simulation state (robot-centric in world frame).
    Includes:
      - robot orientation (heading) in radians
      - relative goal position in world frame: (goal_dx, goal_dy)
      - relative door position in world frame: (door_dx, door_dy)
      - three closest people relative positions wrt robot: [(dx, dy) x 3] (pad with large value if <3)
      - distances to corridor left and right boundaries (y - y_min, y_max - y)
    """
    robot_pos = np.asarray(sim.robot.position, dtype=float)
    robot_orientation = np.asarray(sim.robot.nav.orientation, dtype=float)
    goal_pos = np.asarray(sim.robot.goal, dtype=float)
    door_pos = np.asarray(sim.robot.door_position, dtype=float)
    large_val = 10.0

    # Relative goal and door positions (world frame)
    goal_dx = float(goal_pos[0] - robot_pos[0])
    goal_dy = float(goal_pos[1] - robot_pos[1])
    door_dx = float(door_pos[0] - robot_pos[0])
    door_dy = float(door_pos[1] - robot_pos[1])

    # Compute three closest people relative positions (dx, dy) in world frame
    rel_people: list[tuple[float, float, float]] = []  # (dist, dx, dy)
    if hasattr(sim.robot, 'people'):
        for person in sim.robot.people:
            if getattr(person, 'active', True):
                p = np.asarray(person.position, dtype=float)
                dx = float(p[0] - robot_pos[0])
                dy = float(p[1] - robot_pos[1])
                d = math.hypot(dx, dy)
                rel_people.append((d, dx, dy))
    
    rel_people.sort(key=lambda t: t[0])
    # Take up to 3, pad if fewer
    rel_feats: list[float] = []
    for i in range(3):
        if i < len(rel_people):
            _, dx, dy = rel_people[i]
            rel_feats.extend([dx, dy])
        else:
            rel_feats.extend([large_val, large_val])

    # Distances to corridor left/right sides using world y-bounds
    dist_left = float(large_val)
    dist_right = float(large_val)
    if hasattr(sim.robot, 'corridor_bounds'):
        bounds = sim.robot.corridor_bounds
        y_min = float(bounds['y_min'])
        y_max = float(bounds['y_max'])
        y = float(robot_pos[1])
        dist_left = max(0.0, y - y_min)
        dist_right = max(0.0, y_max - y)

    feat: list[float] = []
    # Robot heading first
    feat.append(robot_orientation)
    # Goal relative position (most important)
    feat.append(goal_dx)
    feat.append(goal_dy)
    # Door relative position
    feat.append(door_dx)
    feat.append(door_dy)
    # People relative positions
    feat.extend(rel_feats)
    # Wall distances
    feat.append(dist_left)
    feat.append(dist_right)

    return np.asarray(feat, dtype=np.float32)

def extract_nav_features_v1(sim) -> np.ndarray:
    """
    Feature vector from current simulation state (robot-centric in world frame).
    Includes:
      - relative goal position in world frame: (goal_dx, goal_dy)
      - relative door position in world frame: (door_dx, door_dy)
      - three closest people relative positions wrt robot: [(dx, dy) x 3] (pad with large value if <3)
      - distances to corridor left and right boundaries (y - y_min, y_max - y)
    """
    robot_pos = np.asarray(sim.robot.position, dtype=float)
    goal_pos = np.asarray(sim.robot.goal, dtype=float)
    door_pos = np.asarray(sim.robot.door_position, dtype=float)
    large_val = 10.0

    # Relative goal and door positions (world frame)
    goal_dx = float(goal_pos[0] - robot_pos[0])
    goal_dy = float(goal_pos[1] - robot_pos[1])
    door_dx = float(door_pos[0] - robot_pos[0])
    door_dy = float(door_pos[1] - robot_pos[1])

    # Compute three closest people relative positions (dx, dy) in world frame
    rel_people: list[tuple[float, float, float]] = []  # (dist, dx, dy)
    if hasattr(sim.robot, 'people'):
        for person in sim.robot.people:
            if getattr(person, 'active', True):
                p = np.asarray(person.position, dtype=float)
                dx = float(p[0] - robot_pos[0])
                dy = float(p[1] - robot_pos[1])
                d = math.hypot(dx, dy)
                rel_people.append((d, dx, dy))
    
    rel_people.sort(key=lambda t: t[0])
    # Take up to 3, pad if fewer
    rel_feats: list[float] = []
    for i in range(3):
        if i < len(rel_people):
            _, dx, dy = rel_people[i]
            rel_feats.extend([dx, dy])
        else:
            rel_feats.extend([large_val, large_val])

    # Distances to corridor left/right sides using world y-bounds
    dist_left = float(large_val)
    dist_right = float(large_val)
    if hasattr(sim.robot, 'corridor_bounds'):
        bounds = sim.robot.corridor_bounds
        y_min = float(bounds['y_min'])
        y_max = float(bounds['y_max'])
        y = float(robot_pos[1])
        dist_left = max(0.0, y - y_min)
        dist_right = max(0.0, y_max - y)

    feat: list[float] = []
    # Goal relative position (most important)
    feat.append(goal_dx)
    feat.append(goal_dy)
    # Door relative position
    feat.append(door_dx)
    feat.append(door_dy)
    # People relative positions
    feat.extend(rel_feats)
    # Wall distances
    feat.append(dist_left)
    feat.append(dist_right)

    return np.asarray(feat, dtype=np.float32)

def extract_nav_features_v0(sim) -> np.ndarray:
    """
    Feature vector from current simulation state.
    Includes:
      - goal_angle(1), goal_distance(1) - CRITICAL for navigation
      - waypoint(2), door_position(2), door_angle(1)
      - linear_velocity(1), angular_velocity(1)
      - three closest people relative positions wrt robot: [(dx, dy) x 3] (pad with large value if <3)
      - distances to corridor left and right boundaries (y - y_min, y_max - y)
    """
    nav = sim.robot.get_navigation_info(2)

    
    robot_pos = np.asarray(sim.robot.position, dtype=float)
    robot_orientation = float(getattr(sim.robot, 'orientation', 0.0))
    goal_pos = np.asarray(sim.robot.goal, dtype=float)
    large_val = 10.0

    # Goal direction in robot frame (CRITICAL for agent to know where to go!)
    goal_vec = goal_pos - robot_pos
    goal_angle_world = math.atan2(goal_vec[1], goal_vec[0])
    goal_angle_robot = goal_angle_world - robot_orientation
    # Normalize to [-π, π]
    while goal_angle_robot > math.pi:
        goal_angle_robot -= 2 * math.pi
    while goal_angle_robot < -math.pi:
        goal_angle_robot += 2 * math.pi
    goal_dist = float(np.linalg.norm(goal_vec))

    # Compute three closest people relative positions (dx, dy) in robot frame
    rel_people: list[tuple[float, float, float]] = []  # (dist, dx, dy)
    if hasattr(sim.robot, 'people'):
        for person in sim.robot.people:
            if getattr(person, 'active', True):
                p = np.asarray(person.position, dtype=float)
                # Transform to robot frame
                rel_x = p[0] - robot_pos[0]
                rel_y = p[1] - robot_pos[1]
                # Rotate into robot's reference frame
                cos_theta = math.cos(-robot_orientation)
                sin_theta = math.sin(-robot_orientation)
                rel_x_robot = rel_x * cos_theta - rel_y * sin_theta
                rel_y_robot = rel_x * sin_theta + rel_y * cos_theta
                d = math.hypot(rel_x_robot, rel_y_robot)
                rel_people.append((d, rel_x_robot, rel_y_robot))
    
    rel_people.sort(key=lambda t: t[0])
    # Take up to 3, pad if fewer
    rel_feats: list[float] = []
    for i in range(3):
        if i < len(rel_people):
            _, dx, dy = rel_people[i]
            rel_feats.extend([dx, dy])
        else:
            rel_feats.extend([large_val, large_val])

    # Distances to corridor left/right sides using y-bounds
    dist_left = float(large_val)
    dist_right = float(large_val)
    if hasattr(sim.robot, 'corridor_bounds'):
        bounds = sim.robot.corridor_bounds
        y_min = float(bounds['y_min'])
        y_max = float(bounds['y_max'])
        y = float(robot_pos[1])
        dist_left = max(0.0, y - y_min)
        dist_right = max(0.0, y_max - y)

    feat = []
    # Add goal information first (most important!)
    feat.append(float(goal_angle_robot))
    feat.append(float(goal_dist))
    # Rest of features
    feat.extend(list(map(float, nav['waypoint'])))
    feat.extend(list(map(float, nav['door_position'])))
    feat.append(float(nav['door_angle']))
    feat.append(float(nav['linear_velocity']))
    feat.append(float(nav['angular_velocity']))
    feat.extend(rel_feats)              # (dx, dy) x 3 in robot frame
    feat.append(dist_left)
    feat.append(dist_right)
    return np.asarray(feat, dtype=np.float32)


def check_robot_overlap(sim) -> dict:
    """
    Check if the robot overlaps with person inflation zones or door inflation zone.
    
    Returns:
        dict with keys:
            - 'person_overlap': bool, True if robot is in any person's inflation zone
            - 'door_overlap': bool, True if robot is in door's inflation zone
            - 'overlap_type': str, one of 'none', 'person', 'door', 'both'
    """
    robot_pos = sim.robot.position
    robot_radius = sim.robot.radius
    
    # Check person overlaps
    person_overlap = False
    person_inflation_radius = 0.2  # Same as used in costmap
    
    for person in sim.robot.people:
        if not person.active:
            continue
        
        # Get person's proxemic ellipse parameters
        axes = getattr(person, 'proxemic_axes', np.array([person.radius, person.radius], dtype=float)).astype(float)
        a = max(float(axes[0]), 1e-4)  # semi-minor axis (width)
        b = max(float(axes[1]), 1e-4)  # semi-major axis (length)
        
        # Ellipse offset (person is at rear of ellipse)
        ellipse_offset_ratio = 0.3
        ellipse_offset = b * ellipse_offset_ratio
        
        # Get person heading
        heading_angle = getattr(person, 'heading_angle', 0.0)
        
        # Vector from person to robot
        rel_x = robot_pos[0] - person.position[0]
        rel_y = robot_pos[1] - person.position[1]
        
        # Rotate into ellipse-aligned frame
        cos_t = math.cos(heading_angle)
        sin_t = math.sin(heading_angle)
        local_x = rel_x * cos_t - rel_y * sin_t
        local_y = rel_x * sin_t + rel_y * cos_t
        
        # Apply offset (shift ellipse forward)
        local_x_shifted = local_x + ellipse_offset
        
        # Check if robot center is inside the ellipse
        norm = math.sqrt((local_x_shifted / b) ** 2 + (local_y / a) ** 2)
        
        # Calculate distance to ellipse boundary
        if norm > 1e-6:
            scale = 1.0 / norm
            boundary_x_shifted = local_x_shifted * scale
            boundary_y = local_y * scale
            boundary_x = boundary_x_shifted - ellipse_offset
            diff_x = local_x - boundary_x
            diff_y = local_y - boundary_y
            dist_to_boundary = math.hypot(diff_x, diff_y)
            if norm < 1.0:
                dist_to_boundary = -dist_to_boundary
        else:
            dist_to_boundary = -min(a, b)
        
        # Check if robot overlaps with inflated ellipse
        clearance = dist_to_boundary - robot_radius
        if clearance <= person_inflation_radius:
            person_overlap = True
            break
    
    # Check door overlap
    door_overlap = False
    if hasattr(sim.robot, 'door_position') and hasattr(sim.robot, 'corridor_bounds'):
        door_pos = np.array(sim.robot.door_position, dtype=float)
        bounds = sim.robot.corridor_bounds
        
        # Door halo radius
        door_inflation_radius = float(getattr(sim.robot.global_planner, 'door_halo_radius', 1.0))
        
        # Distance from robot to door
        dist_to_door = np.linalg.norm(robot_pos - door_pos)
        
        # Determine door side and inward normal
        corridor_mid_y = (bounds['y_min'] + bounds['y_max']) * 0.5
        door_side = "left" if door_pos[1] < corridor_mid_y else "right"
        n_world = np.array([0.0, 1.0]) if door_side == "left" else np.array([0.0, -1.0])
        
        # Vector from door to robot
        v_x = robot_pos[0] - door_pos[0]
        v_y = robot_pos[1] - door_pos[1]
        
        # Check if robot is on the inward-facing side (semicircle)
        dot_product = n_world[0] * v_x + n_world[1] * v_y
        
        # Robot overlaps if within door halo radius AND on inward-facing side
        if dist_to_door <= (door_inflation_radius + robot_radius) and dot_product > 0.0:
            door_overlap = True
    
    # Determine overlap type
    if person_overlap and door_overlap:
        overlap_type = 'both'
    elif person_overlap:
        overlap_type = 'person'
    elif door_overlap:
        overlap_type = 'door'
    else:
        overlap_type = 'none'
    
    return {
        'person_overlap': person_overlap,
        'door_overlap': door_overlap,
        'overlap_type': overlap_type
    }

def compute_reward_v3(sim, progress_prev_dist: float, offset: float = 0.0, door_dy: float | None = None) -> tuple[float, float, dict]:
    """
    Reward focused on obstacle avoidance with better credit assignment.
    
    Key improvements:
    1. Progress component (small) to ensure episodes complete
    2. Obstacle-based rewards (primary signal)
    3. Offset regularization to encourage meaningful corrections
    
    Returns (reward, new_progress_distance, info)
    """
    robot_pos = sim.robot.position
    goal_pos = sim.robot.goal
    dist = float(np.linalg.norm(goal_pos - robot_pos))

    # Start with progress-based shaping so the policy gets a dense signal.
    # (Positive when moving toward the goal.)
    progress = float(progress_prev_dist) - float(dist)
    reward =  np.sign(progress) * ((10.0 * progress) ** 2)
    # Check overlap with inflation zones
    overlap_info = check_robot_overlap(sim)
    
    # PRIMARY SIGNAL: Obstacle-based rewards
    overlap_type = overlap_info['overlap_type']
    if overlap_type == 'none':
        # Small time penalty to encourage finishing sooner (but don't swamp progress).
        reward += -0.2
    elif overlap_type == 'person':
        # Penalty for being in person's proxemic zone
        reward += -2.0*0
    elif overlap_type == 'door':
        reward += -0.2
    elif overlap_type == 'both':
        # Higher penalty for being in both zones
        reward += -2.0*0
    else:
        reward += -0.2

    # Hard collision penalty (critical - must avoid)
    collisions = sim.collision_count
    info = {
        'distance': dist,
        'collisions': collisions,
        'overlap_type': overlap_type,
        'person_overlap': overlap_info['person_overlap'],
        'door_overlap': overlap_info['door_overlap'],
        'offset': offset,
        'offset_abs': abs(offset),
        'door_dy': door_dy if door_dy is not None else float('nan'),
    }
    
    # Penalty for being too close to corridor walls
    if hasattr(sim.robot, 'corridor_bounds'):
        bounds = sim.robot.corridor_bounds
        y_min = float(bounds['y_min'])
        y_max = float(bounds['y_max'])
        y = float(robot_pos[1])
        dist_left = max(0.0, y - y_min)
        dist_right = max(0.0, y_max - y)
        if dist_left < 0.4 or dist_right < 0.4:
            reward += -0.5
    
    # Penalty for facing backwards (heading between -π and -π/2, or between π/2 and π)
    # if hasattr(sim.robot, 'nav') and hasattr(sim.robot.nav, 'orientation'):
    #     heading = float(sim.robot.nav.orientation)
    #     if (-math.pi < heading < -math.pi/3) or (math.pi/3 < heading < math.pi):
    #         reward += -1.0

    # Compute alignment (cosine similarity)
    # 1.0 = facing goal directly, -1.0 = facing away

    
    return reward, dist, info



def compute_reward(sim, progress_prev_dist: float, offset: float = 0.0, door_dy: float | None = None) -> tuple[float, float, dict]:
    """
    Reward focused on obstacle avoidance with better credit assignment.
    
    Key improvements:
    1. Progress component (small) to ensure episodes complete
    2. Obstacle-based rewards (primary signal)
    3. Offset regularization to encourage meaningful corrections
    
    Returns (reward, new_progress_distance, info)
    """
    robot_pos = sim.robot.position
    goal_pos = sim.robot.goal
    dist = float(np.linalg.norm(goal_pos - robot_pos))

    # Start with progress-based shaping so the policy gets a dense signal.
    # (Positive when moving toward the goal.)
    progress = float(progress_prev_dist) - float(dist)
    reward =  np.sign(progress) * ((10.0 * progress) ** 2)
    # Check overlap with inflation zones
    overlap_info = check_robot_overlap(sim)
    
    # PRIMARY SIGNAL: Obstacle-based rewards
    overlap_type = overlap_info['overlap_type']
    if overlap_type == 'none':
        # Small time penalty to encourage finishing sooner (but don't swamp progress).
        reward += -0.2
    elif overlap_type == 'person':
        # Penalty for being in person's proxemic zone
        reward += -0.5
    elif overlap_type == 'door':
        reward += -0.2
    elif overlap_type == 'both':
        # Higher penalty for being in both zones
        reward += -0.5
    else:
        reward += -0.2

    # Hard collision penalty (critical - must avoid)
    collisions = sim.collision_count
    info = {
        'distance': dist,
        'collisions': collisions,
        'overlap_type': overlap_type,
        'person_overlap': overlap_info['person_overlap'],
        'door_overlap': overlap_info['door_overlap'],
        'offset': offset,
        'offset_abs': abs(offset),
        'door_dy': door_dy if door_dy is not None else float('nan'),
    }
    
    # Penalty for being too close to corridor walls
    if hasattr(sim.robot, 'corridor_bounds'):
        bounds = sim.robot.corridor_bounds
        y_min = float(bounds['y_min'])
        y_max = float(bounds['y_max'])
        y = float(robot_pos[1])
        dist_left = max(0.0, y - y_min)
        dist_right = max(0.0, y_max - y)
        if dist_left < 0.4 or dist_right < 0.4:
            reward += -0.5
    
    # Penalty for facing backwards (heading between -π and -π/2, or between π/2 and π)
    # if hasattr(sim.robot, 'nav') and hasattr(sim.robot.nav, 'orientation'):
    #     heading = float(sim.robot.nav.orientation)
    #     if (-math.pi < heading < -math.pi/3) or (math.pi/3 < heading < math.pi):
    #         reward += -1.0

    # Compute alignment (cosine similarity)
    # 1.0 = facing goal directly, -1.0 = facing away
    heading = float(sim.robot.nav.orientation)
    to_goal = sim.robot.goal - sim.robot.position
    to_goal = to_goal / np.linalg.norm(to_goal)
    robot_heading_vec = np.array([math.cos(heading), math.sin(heading)])
    alignment = np.dot(to_goal, robot_heading_vec)
    reward += 0.05 * alignment

    return reward, dist, info


def compute_reward_v0(sim, progress_prev_dist: float, offset: float = 0.0) -> tuple[float, float, dict]:
    """
    Reward focused on obstacle avoidance with better credit assignment.
    
    Key improvements:
    1. Progress component (small) to ensure episodes complete
    2. Obstacle-based rewards (primary signal)
    3. Offset regularization to encourage meaningful corrections
    
    Returns (reward, new_progress_distance, info)
    """
    robot_pos = sim.robot.position
    goal_pos = sim.robot.goal
    dist = float(np.linalg.norm(goal_pos - robot_pos))

    reward = 0
    # Check overlap with inflation zones
    overlap_info = check_robot_overlap(sim)
    
    # PRIMARY SIGNAL: Obstacle-based rewards
    overlap_type = overlap_info['overlap_type']
    if overlap_type == 'none':
        # Positive reward for staying in free space
        reward = -0.05
    elif overlap_type == 'person':
        # Penalty for being in person's proxemic zone
        reward = -1.0  # Increased penalty to make signal stronger
    elif overlap_type == 'both':
        # Higher penalty for being in both zones
        reward = -1.0  # Increased penalty
    else:
        reward = -0.05

    # Hard collision penalty (critical - must avoid)
    collisions = sim.collision_count
    info = {
        'distance': dist,
        'collisions': collisions,
        'overlap_type': overlap_type,
        'person_overlap': overlap_info['person_overlap'],
        'door_overlap': overlap_info['door_overlap'],
        'offset': offset,
        'offset_abs': abs(offset),
    }
    
    # If a collision occurred in this step
    if sim.collision_history:
        if abs(sim.collision_history[-1]['timestamp'] - __import__('datetime').datetime.now().timestamp()) < 0.2:
            reward += -1.0  # Large penalty for actual collision
    

    return reward, dist, info


def evaluate_policy(agent: Any, agent_type: str, algo: str, num_episodes: int = 5, 
                    max_steps: int = 3000, dt: float = 1/60.0, 
                    w_max_min: float = -0.2 * math.pi, w_max_max: float = math.pi) -> float:
    """
    Evaluate the current policy with exploration turned off.
    Returns the average return over num_episodes.
    """
    eval_returns = []
    # Get device from agent (works for PPO and TD3)
    if hasattr(agent, 'policy_old'):
        device = next(agent.policy_old.parameters()).device
    elif hasattr(agent, 'ac') and hasattr(agent.ac, 'pi'):
        device = next(agent.ac.pi.parameters()).device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for eval_ep in range(num_episodes):
        # Create evaluation environment (same as training)
        corridor_width_ep = random.uniform(2.0, 4.0)
        sim = Simulation(
            corridor_width=corridor_width_ep,
            corridor_length=10.0,
            num_people=random.randint(2, 4),
            people_speeds=[random.uniform(0.6, 1.0) for _ in range(10)],
            spawn_interval=random.uniform(0.5, 2.0),
        )
        configure_nav(sim, algo)
        
        # Reset recurrent hidden state if applicable
        if hasattr(agent, "reset_hidden"):
            try:
                agent.reset_hidden()
            except Exception:
                pass
        
        # Initialize
        _, _, _ = sim.step(dt)
        robot_pos = sim.robot.position
        goal_pos = sim.robot.goal
        prev_dist = float(np.linalg.norm(goal_pos - robot_pos))
        
        episode_return = 0.0
        prev_offset = None
        lstm_hidden = None
        
        if agent_type == "ppo_lstm":
            try:
                lstm_hidden = agent.policy_old.init_hidden(batch_size=1)
            except Exception:
                lstm_hidden = None
        
        t = 0
        while t < max_steps:
            obs = extract_nav_features(sim)
            
            # Select action deterministically (no exploration)
            if agent_type in ("td3", "td3_lstm"):
                # TD3: use noise_scale=0.0 for deterministic action
                action = agent.select_action(obs, noise_scale=0.0)
                prev_offset = max(-1.0, min(1.0, float(np.asarray(action).reshape(-1)[0])))
            elif agent_type == "ppo_lstm":
                # PPO-LSTM: deterministic action (use mean, not sample)
                st = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    if lstm_hidden is None:
                        lstm_hidden = agent.policy_old.init_hidden(batch_size=1)
                    # Get action mean directly from actor head (deterministic)
                    lstm_out, lstm_hidden = agent.policy_old._forward_seq(st.view(1, -1), lstm_hidden)
                    feat = lstm_out[:, -1, :].squeeze(0)
                    action_mean = agent.policy_old.actor_head(feat)
                    prev_offset = max(-1.0, min(1.0, float(action_mean.view(-1)[0].item())))
            else:
                # PPO: deterministic action (use mean directly from actor)
                st = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action_mean = agent.policy_old.actor(st)
                    prev_offset = max(-1.0, min(1.0, float(action_mean.view(-1)[0].item())))
            
            # Apply action to planner
            if algo == "ts_dwa":
                # Map action from [-1, 1] to [0, -corridor_width/1.25] for alpha
                # Then pass alpha=-alpha to the function (so final alpha is in [0, corridor_width/1.25])
                action_clamped = max(-1.0, min(1.0, float(prev_offset)))
                alpha_raw = (action_clamped + 1.0) / 2.0 * (-sim.corridor_width / 1.25)
                sim.robot.add_gaussian_bump_to_path(alpha=alpha_raw, sigma=2.0)
                offset = 0.0  # No offset used anymore
            else:
                a = max(-1.0, min(1.0, float(prev_offset)))
                a01 = (float(a) + 1.0) * 0.5
                w_max = float(w_max_min + a01 * (w_max_max - w_max_min))
                if hasattr(sim.robot, "nav") and hasattr(sim.robot.nav, "w_max"):
                    sim.robot.nav.w_max = float(w_max)
                offset = 0.0
            
            # Step simulation
            _, _, done = sim.step(dt)
            reward, prev_dist, info = compute_reward(sim, prev_dist, offset)
            episode_return += float(reward)
            
            t += 1
            if done:
                break
        
        eval_returns.append(episode_return)
    
    return float(np.mean(eval_returns)) if eval_returns else 0.0


def log_training_step(csv_writer, episode: int, step: int, nav_features: np.ndarray, 
                     action: float, reward: float, distance: float, sim: Simulation) -> None:
    """Log a single training step to CSV."""
    robot_pos = sim.robot.position
    goal_pos = sim.robot.goal
    door_pos = sim.get_door_position()
    
    # Get robot orientation
    robot_orientation = float(sim.robot.nav.orientation) if hasattr(sim.robot, 'nav') and hasattr(sim.robot.nav, 'orientation') else 0.0
    
    # Get people positions (up to 3)
    people_positions = []
    for i, person in enumerate(sim.people[:3]):
        if person.active:
            people_positions.extend([float(person.position[0]), float(person.position[1])])
        else:
            people_positions.extend([float('nan'), float('nan')])
    # Pad to 3 people (6 values)
    while len(people_positions) < 6:
        people_positions.extend([float('nan'), float('nan')])
    
    # Write row: episode, step, nav_features (all), action, reward, distance, 
    # robot_x, robot_y, robot_orientation, goal_x, goal_y, door_x, door_y,
    # corridor_width, door_side, num_people, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y
    row = [
        episode, step,
        *nav_features.tolist(),  # All navigation features
        action,
        reward,
        distance,
        float(robot_pos[0]), float(robot_pos[1]),
        robot_orientation,
        float(goal_pos[0]), float(goal_pos[1]),
        float(door_pos[0]), float(door_pos[1]),
        float(sim.corridor_width),
        str(sim.door_side),
        len([p for p in sim.people if p.active]),
        *people_positions,
    ]
    csv_writer.writerow(row)


def train(config: Dict[str, Any], use_wandb: bool = True, run_name: Optional[str] = None) -> float:
    seed = int(config.get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup training data logging
    log_training_data = bool(config.get('log_training_data', False))
    csv_file = None
    csv_writer = None
    if log_training_data:
        os.makedirs('simulation_training', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join('simulation_training', f'training_data_{timestamp}.csv')
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        
        # Write header: need to know feature dimension first
        # We'll write it after we create the first simulation

    # Action space: single offset value for heading adjustment
    num_actions = 1

    # Environment / training hyperparameters (algorithm-agnostic where possible)
    gamma = float(config.get('gamma', 0.99))
    # PPO-specific (but genericizable via agent config)
    k_epochs = int(config.get('k_epochs', 10))
    eps_clip = float(config.get('eps_clip', 0.2))
    update_timestep = int(config.get('update_timestep', 1024))
    action_select_interval = int(config.get('action_select_interval', 1))  # select new action every N steps
    # If true: treat `action_select_interval` as a real frame-skip/macro-step.
    # We will apply the same action for N sim steps, accumulate reward over those N steps,
    # and store exactly ONE (state, action, reward_sum, done) transition into PPO per macro-step.
    macro_step = bool(config.get("macro_step", False))
    lr_actor = float(config.get('lr_actor', config.get('learning_rate', 1e-3)))
    lr_critic = float(config.get('lr_critic', config.get('learning_rate', 1e-3)))
    # Action smoothness (jerkiness) penalty: coef * (a_t - a_{t-1})^2 in normalized action space.
    # Applied once per decision / macro-step for both PPO and TD3.
    action_smooth_coef = float(config.get("action_smooth_coef", 0.0))
    
    # Stuck detection: terminate episode early if robot doesn't make progress for N steps
    stuck_detection_enabled = bool(config.get("stuck_detection_enabled", False))
    stuck_threshold_steps = int(config.get("stuck_threshold_steps", 400))  # steps without progress before termination
    stuck_min_progress = float(config.get("stuck_min_progress", 0.005))  # minimum distance improvement (m) to count as progress
    stuck_penalty = float(config.get("stuck_penalty", -100.0))  # penalty when episode terminates due to stuck
    stuck_progress_reset_steps = int(config.get("stuck_progress_reset_steps", 10))  # consecutive progress steps needed to reset stuck counter
    stuck_no_progress_window = int(config.get("stuck_no_progress_window", 200))  # steps without progress before starting to count as stuck

    # Env defaults (also used to infer feature dimension)
    algo = str(config.get("algo", "ts_dwa")).lower()
    if algo not in ("ts_dwa", "dwa"):
        raise ValueError(f"Unsupported algo={algo!r}. Expected 'ts_dwa' or 'dwa'.")

    # For dwa_door_aware we control the DWA angular sampling limit directly:
    # map a clamped action a in [-1.0, 1.0] to nav.w_max in [w_max_min, w_max_max].
    # Default range requested: [-0.2*pi, pi].
    w_max_min = float(config.get("w_max_min", -0.2 * math.pi))
    w_max_max = float(config.get("w_max_max", math.pi))

    dt = float(config.get('dt', 1/60.0))
    corridor_width = float(config.get('corridor_width', 4.0))
    door_side = str(config.get('door_side', 'right'))
    num_people = int(config.get('num_people', 3))
    people_speed_min = float(config.get('people_speed_min', 0.6))
    people_speed_max = float(config.get('people_speed_max', 1.0))

    # Infer input feature dimension using a temporary simulation
    tmp_sim = Simulation(
        corridor_width=corridor_width,
        door_side=door_side,
        num_people=num_people,
        people_speeds=[random.uniform(people_speed_min, people_speed_max) for _ in range(10)],
    )
    configure_nav(tmp_sim, algo)
    _ = tmp_sim.step(dt)
    input_dim = int(len(extract_nav_features(tmp_sim)))

    # Build agent
    hidden_size = int(config.get('hidden_size', 128))
    # If we store only one transition per macro-step (N sim steps), adjust the discount so the
    # effective horizon is roughly preserved: gamma_macro = gamma ** N.
    gamma_for_agent = float(gamma)
    if macro_step and action_select_interval > 1:
        gamma_for_agent = float(gamma) ** float(action_select_interval)
    agent_type = str(config.get("agent", "ppo")).lower().strip()
    if agent_type == "td3" or agent_type == "td3_lstm":
        # TD3 hyperparams (configurable, but with safe defaults)
        td3_replay_size = int(config.get("td3_replay_size", 200000))
        td3_start_steps = int(config.get("td3_start_steps", 9000))
        td3_update_after = int(config.get("td3_update_after", 2000))
        td3_update_every = int(config.get("td3_update_every", 50))
        td3_batch_size = int(config.get("td3_batch_size", 256))
        td3_policy_delay = int(config.get("td3_policy_delay", 2))
        td3_polyak = float(config.get("td3_polyak", 0.995))
        td3_act_noise = float(config.get("td3_act_noise", 0.15))
        td3_target_noise = float(config.get("td3_target_noise", 0.3))
        td3_noise_clip = float(config.get("td3_noise_clip", 0.5))
        td3_max_hist_len = int(config.get("td3_max_hist_len", 10))

        if agent_type == "td3_lstm":
            agent = TD3_LSTM(
                obs_dim=input_dim,
                act_dim=num_actions,
                act_limit=1.0,
                gamma=gamma_for_agent,
                polyak=td3_polyak,
                pi_lr=lr_actor,
                q_lr=lr_critic,
                policy_delay=td3_policy_delay,
                act_noise=td3_act_noise,
                target_noise=td3_target_noise,
                noise_clip=td3_noise_clip,
                max_hist_len=td3_max_hist_len,
                hist_with_past_act=True,
            )
        else:
            agent = TD3(
                obs_dim=input_dim,
                act_dim=num_actions,
                act_limit=1.0,
                gamma=gamma_for_agent,
                polyak=td3_polyak,
                pi_lr=lr_actor,
                q_lr=lr_critic,
                policy_delay=td3_policy_delay,
                act_noise=td3_act_noise,
                target_noise=td3_target_noise,
                noise_clip=td3_noise_clip,
                use_history=False,
            )

        replay = TD3ReplayBuffer(obs_dim=input_dim, act_dim=num_actions, max_size=td3_replay_size)
    elif agent_type == "ppo_lstm":
        agent = PPO_LSTM(
            state_dim=input_dim,
            action_dim=num_actions,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma_for_agent,
            K_epochs=k_epochs,
            eps_clip=eps_clip,
            has_continuous_action_space=True,
            action_std_init=0.4,
        )
    else:
        agent = PPO(
            state_dim=input_dim,
            action_dim=num_actions,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma_for_agent,
            K_epochs=k_epochs,
            eps_clip=eps_clip,
            has_continuous_action_space=True,
            action_std_init=0.4,
        )

    if use_wandb and wandb is not None:
        wandb.init(project=str(config.get('wandb_project', 'PredictiveDWA')),
                   name=run_name, config=config, allow_val_change=True)
        # Watch networks if available
        try:
            if hasattr(agent, "policy") and hasattr(agent.policy, "actor"):
                wandb.watch(agent.policy.actor, log='all', log_freq=200)
            if hasattr(agent, "policy") and hasattr(agent.policy, "critic"):
                wandb.watch(agent.policy.critic, log='all', log_freq=200)
            if hasattr(agent, "ac") and hasattr(agent.ac, "pi"):
                wandb.watch(agent.ac.pi, log='all', log_freq=200)
            if hasattr(agent, "ac") and hasattr(agent.ac, "q1"):
                wandb.watch(agent.ac.q1, log='all', log_freq=200)
        except Exception:
            pass

    # Update counters:
    # - PPO: counts rollout transitions in the PPO buffer
    # - TD3: counts stored transitions in replay buffer
    time_step = 0

    # Training loop (episodes of the headless simulation)
    episodes = int(config.get('episodes', 50))
    max_steps = int(config.get('max_steps', 3000))

    global_step = 0
    returns = []
    start_time = time.time()
    
    # Write CSV header after we know feature dimension
    if log_training_data and csv_writer is not None:
        # Create a temporary sim to get feature dimension
        tmp_sim = Simulation(corridor_width=4.0, corridor_length=10.0, num_people=1)
        configure_nav(tmp_sim, algo)
        _ = tmp_sim.step(dt)
        tmp_features = extract_nav_features(tmp_sim)
        feature_dim = len(tmp_features)
        
        # Write header
        header = ['episode', 'step']
        header.extend([f'nav_feat_{i}' for i in range(feature_dim)])
        header.extend(['action', 'reward', 'distance', 'robot_x', 'robot_y', 'robot_orientation',
                      'goal_x', 'goal_y', 'door_x', 'door_y', 'corridor_width', 'door_side', 'num_people',
                      'p1_x', 'p1_y', 'p2_x', 'p2_y', 'p3_x', 'p3_y'])
        csv_writer.writerow(header)
        print(f"[Logging] Writing training data to simulation_training/training_data_*.csv")

    for ep in range(episodes):
        num_people = random.randint(2, 4) # Random number of people between 0 and 10
        # Randomize corridor width per-episode (uniform in [2.0, 4.0])
        corridor_width_ep = random.uniform(2.0, 4.0)
        # sim = Simulation(
        #     corridor_width=corridor_width,
        #     door_side=door_side,
        #     num_people=num_people,
        #     people_speeds=[random.uniform(people_speed_min, people_speed_max) for _ in range(10)],
        # )
        sim = Simulation(
            corridor_width=corridor_width_ep,
            corridor_length=10.0,
            num_people = random.randint(2, 4),
            people_speeds=[random.uniform(0.6, 1.0) for _ in range(10)],
            spawn_interval = random.uniform(0.5, 2.0),
        )
        configure_nav(sim, algo)
        # Reset recurrent hidden state at episode start (PPO_LSTM only)
        if hasattr(agent, "reset_hidden"):
            try:
                agent.reset_hidden()
            except Exception:
                pass

        # Reset progress tracker
        _, _, _ = sim.step(dt)  # advance once to initialize internal state
        robot_pos = sim.robot.position
        goal_pos = sim.robot.goal
        prev_dist = float(np.linalg.norm(goal_pos - robot_pos))

        episode_return = 0.0
        overlap_counts = {'none': 0, 'person': 0, 'door': 0, 'both': 0}
        offset_history = []  # TS-DWA: abs(offset) rad ; DWA-door-aware: w_max (rad/s)

        # Storage for action repetition
        prev_offset = None
        prev_action_norm: Optional[float] = None
        prev_collision_count = int(getattr(sim, "collision_count", 0))
        prev_applied_bias: Optional[float] = None
        
        # Stuck detection: track best distance and steps without progress
        best_dist = prev_dist
        steps_without_progress = 0
        consecutive_progress_steps = 0  # Track consecutive steps with progress
        steps_since_last_progress = 0  # Track steps since last progress (allows for oscillations)
        stuck_terminated = False

        # --- Main interaction loop ---
        if agent_type in ("td3", "td3_lstm"):
            # TD3 / TD3-LSTM loop (off-policy)
            hist = None
            if agent_type == "td3_lstm" and hasattr(agent, "make_history_buffer"):
                hist = agent.make_history_buffer()
                hist.reset()

            t = 0
            while t < max_steps:
                # Decision-level observation
                obs = extract_nav_features(sim)
                if hist is not None:
                    hist.push_obs(obs)

                # Action selection
                if replay.size < td3_start_steps:
                    action = np.array([random.uniform(-1.0, 1.0)], dtype=np.float32)
                else:
                    if hist is not None and hasattr(agent, "select_action_with_history"):
                        action = agent.select_action_with_history(obs, hist, noise_scale=None)
                    else:
                        action = agent.select_action(obs, noise_scale=None)
                    action = np.asarray(action, dtype=np.float32).reshape(1)
                action = np.clip(action, -2.0, 2.0)
                prev_offset = float(action[0])
                # Action jerkiness penalty (once per decision / macro-step)
                action_smooth_pen = 0.0
                if action_smooth_coef > 0.0 and prev_action_norm is not None:
                    da = float(prev_offset) - float(prev_action_norm)
                    action_smooth_pen = float(action_smooth_coef) * (da * da)
                prev_action_norm = float(prev_offset)
                if hist is not None:
                    hist.push_act(action)

                # Apply mapping to planner param
                smooth_pen = 0.0
                if algo == "ts_dwa":
                    # Map action from [-1, 1] to [0, -corridor_width/1.25] for alpha
                    # Then pass alpha=-alpha to the function (so final alpha is in [0, corridor_width/1.25])
                    action_clamped = max(-1.0, min(1.0, float(prev_offset)))
                    alpha_raw = (action_clamped + 1.0) / 2.0 * (-sim.corridor_width / 1.25)
                    sim.robot.add_gaussian_bump_to_path(alpha=alpha_raw, sigma=2.0)
                    offset = 0.0  # No offset used anymore
                    offset_history.append(abs(alpha_raw))  # Track the alpha value instead
                else:
                    a = max(-1.0, min(1.0, float(prev_offset)))
                    # Clamp action to [-1, 1], renormalize to [0, 1], then map to w_max range.
                    a01 = (float(a) + 1.0) * 0.5  # [-1,1] -> [0,1]
                    w_max = float(w_max_min + a01 * (w_max_max - w_max_min))

                    if hasattr(sim.robot, "nav") and hasattr(sim.robot.nav, "w_max"):
                        sim.robot.nav.w_max = float(w_max)

                    if prev_applied_bias is not None:
                        smooth_pen = 0.05 * abs(float(w_max) - float(prev_applied_bias))
                    prev_applied_bias = float(w_max)
                    offset = 0.0
                    offset_history.append(float(w_max))

                # Step sim (macro-step optionally)
                inner_steps = action_select_interval if (macro_step and action_select_interval > 1) else 1
                reward_sum = 0.0
                done_flag = False
                last_info = None

                if not macro_step:
                    # single-step transition
                    _, _, done_flag = sim.step(dt)
                    reward, prev_dist, info = compute_reward(sim, prev_dist, offset)
                    
                    # Log training data (log obs before step, action, reward after step)
                    if log_training_data and csv_writer is not None:
                        # Get observation after step for logging
                        obs_after = extract_nav_features(sim)
                        log_training_step(csv_writer, ep, t, obs_after, prev_offset, float(reward), prev_dist, sim)
                    
                    # collision delta penalty
                    # cur_collisions = int(getattr(sim, "collision_count", 0))
                    # if cur_collisions > prev_collision_count:
                    #     reward -= 5.0 * float(cur_collisions - prev_collision_count)
                    # prev_collision_count = cur_collisions
                    
                    # Stuck detection: check if robot made progress
                    if stuck_detection_enabled:
                        if prev_dist < best_dist - stuck_min_progress:
                            # Made progress: increment consecutive progress counter
                            consecutive_progress_steps += 1
                            best_dist = prev_dist
                            # Only reset stuck counter after sustained progress
                            if consecutive_progress_steps >= stuck_progress_reset_steps:
                                steps_without_progress = 0
                        else:
                            # No progress: reset consecutive progress counter and increment stuck counter
                            consecutive_progress_steps = 0
                            steps_without_progress += 1
                            # Terminate if stuck for too long
                            if steps_without_progress >= stuck_threshold_steps:
                                reward += stuck_penalty
                                done_flag = True
                                stuck_terminated = True
                    
                    reward_sum = float(reward)
                    last_info = info
                    overlap_counts[info['overlap_type']] += 1
                    episode_return += float(reward_sum)
                    t += 1
                    global_step += 1
                else:
                    # N-step accumulated transition
                    for inner_idx in range(inner_steps):
                        if t >= max_steps:
                            break
                        _, _, done_flag = sim.step(dt)
                        reward, prev_dist, info = compute_reward(sim, prev_dist, offset)
                        
                        # Log training data (only at decision steps, not every micro-step)
                        if log_training_data and csv_writer is not None and inner_idx == 0:
                            # Get current observation for logging
                            current_obs = extract_nav_features(sim)
                            log_training_step(csv_writer, ep, t, current_obs, prev_offset, float(reward), prev_dist, sim)
                        
                        # cur_collisions = int(getattr(sim, "collision_count", 0))
                        # if cur_collisions > prev_collision_count:
                        #     reward -= 5.0 * float(cur_collisions - prev_collision_count)
                        # prev_collision_count = cur_collisions
                        
                        # Stuck detection: check if robot made progress
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
                                    # Terminate if stuck for too long
                                    if steps_without_progress >= stuck_threshold_steps:
                                        reward += stuck_penalty
                                        done_flag = True
                                        stuck_terminated = True
                        
                        reward_sum += float(reward)
                        last_info = info
                        overlap_counts[info['overlap_type']] += 1
                        episode_return += float(reward)
                        t += 1
                        global_step += 1
                        if done_flag:
                            break

                if algo != "ts_dwa":
                    reward_sum -= float(smooth_pen)
                    episode_return -= float(smooth_pen)
                if action_smooth_pen != 0.0:
                    reward_sum -= float(action_smooth_pen)
                    episode_return -= float(action_smooth_pen)

                # Store transition in replay buffer
                next_obs = extract_nav_features(sim)
                replay.store(obs, np.array([prev_offset], dtype=np.float32), float(reward_sum), next_obs, float(done_flag))

                # Updates
                time_step += 1
                if replay.size >= td3_update_after and (time_step % td3_update_every == 0):
                    last_stats = None
                    for _ in range(td3_update_every):
                        last_stats = agent.update(replay, batch_size=td3_batch_size)
                    if use_wandb and wandb is not None and last_stats is not None:
                        wandb.log({
                            "td3_loss_q": last_stats.loss_q,
                            "td3_loss_pi": last_stats.loss_pi,
                            "td3_q1_mean": last_stats.q1_mean,
                            "td3_q2_mean": last_stats.q2_mean,
                            "train_time_step": time_step,
                        })

                if done_flag:
                    break

        else:
            # PPO / PPO-LSTM loop (on-policy)
            t = 0
            while t < max_steps:
                # Build state features at decision boundary
                state_feat = extract_nav_features(sim)

                # Sample a new action and store (state, action, logprob, value) once per macro-step.
                if hasattr(agent, "select_action_clamped"):
                    offset_normalized = agent.select_action_clamped(state_feat, clamp=(-1.0, 1.0))
                    prev_offset = float(offset_normalized[0])
                else:
                    offset_normalized = agent.select_action(state_feat)  # Returns array
                    prev_offset = max(-1.0, min(1.0, float(offset_normalized[0])))
                    _overwrite_last_ppo_action(agent, state_feat, prev_offset)

                # Action jerkiness penalty (once per decision / macro-step)
                action_smooth_pen = 0.0
                if action_smooth_coef > 0.0 and prev_action_norm is not None:
                    da = float(prev_offset) - float(prev_action_norm)
                    action_smooth_pen = float(action_smooth_coef) * (da * da)
                prev_action_norm = float(prev_offset)

                # Apply action mapping to the planner parameter once per macro-step
                smooth_pen = 0.0
                if algo == "ts_dwa":
                    # Map action from [-1, 1] to [0, -corridor_width/1.25] for alpha
                    # Then pass alpha=-alpha to the function (so final alpha is in [0, corridor_width/1.25])
                    action_clamped = max(-1.0, min(1.0, float(prev_offset)))
                    alpha_raw = (action_clamped + 1.0) / 2.0 * (-sim.corridor_width / 1.25)
                    sim.robot.add_gaussian_bump_to_path(alpha=alpha_raw, sigma=2.0)
                    offset = 0.0  # No offset used anymore
                    offset_history.append(abs(alpha_raw))  # Track the alpha value instead
                else:
                    a = max(-1.0, min(1.0, float(prev_offset)))
                    # Clamp action to [-1, 1], renormalize to [0, 1], then map to w_max range.
                    a01 = (float(a) + 1.0) * 0.5  # [-1,1] -> [0,1]
                    w_max = float(w_max_min + a01 * (w_max_max - w_max_min))

                    if hasattr(sim.robot, "nav") and hasattr(sim.robot.nav, "w_max"):
                        sim.robot.nav.w_max = float(w_max)

                    if prev_applied_bias is not None:
                        smooth_pen = 0.05 * abs(float(w_max) - float(prev_applied_bias))
                    prev_applied_bias = float(w_max)

                    offset = 0.0
                    offset_history.append(float(w_max))

                # Execute N micro-steps with the same applied action; accumulate reward.
                reward_sum = 0.0
                done_flag = False
                for _ in range(action_select_interval if macro_step else 1):
                    if t >= max_steps:
                        break

                    _, _, done_flag = sim.step(dt)

                    reward, prev_dist, info = compute_reward(sim, prev_dist, offset)
                    
                    # Log training data
                    if log_training_data and csv_writer is not None:
                        # Get current observation for logging
                        current_obs = extract_nav_features(sim)
                        log_training_step(csv_writer, ep, t, current_obs, prev_offset, float(reward), prev_dist, sim)

                    # Penalize collisions as a *delta* so it is a clear learning signal.
                    # cur_collisions = int(getattr(sim, "collision_count", 0))
                    # if cur_collisions > prev_collision_count:
                    #     reward -= 5.0 * float(cur_collisions - prev_collision_count)
                    # prev_collision_count = cur_collisions
                    
                    # Stuck detection: check if robot made progress
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
                                # Terminate if stuck for too long
                                if steps_without_progress >= stuck_threshold_steps:
                                    reward += stuck_penalty
                                    done_flag = True
                                    stuck_terminated = True

                    reward_sum += float(reward)
                    episode_return += float(reward)

                    # Track overlap stats at the micro-step resolution (keeps the same interpretation as before)
                    overlap_counts[info['overlap_type']] += 1

                    global_step += 1
                    t += 1

                    if done_flag:
                        break

                # Apply smoothness penalty ONCE per macro-step (parameter changed once)
                if algo != "ts_dwa":
                    reward_sum -= float(smooth_pen)
                    episode_return -= float(smooth_pen)
                if action_smooth_pen != 0.0:
                    reward_sum -= float(action_smooth_pen)
                    episode_return -= float(action_smooth_pen)

                # PPO bookkeeping: store one reward+terminal per macro-step
                agent.buffer.rewards.append(float(reward_sum))
                agent.buffer.is_terminals.append(bool(done_flag))

                # Trigger update at fixed *macro-step* timesteps (buffer entries)
                time_step += 1
                if time_step % update_timestep == 0:
                    update_stats = agent.update()
                    if use_wandb and wandb is not None and update_stats is not None:
                        wandb.log({
                            'ppo_loss': update_stats['loss'],
                            'ppo_policy_loss': update_stats['policy_loss'],
                            'ppo_value_loss': update_stats['value_loss'],
                            'ppo_entropy': update_stats['entropy'],
                            'train_time_step': time_step,
                        })

                if done_flag:
                    # Update at episode boundary as well
                    update_stats = agent.update()
                    if use_wandb and wandb is not None and update_stats is not None:
                        wandb.log({
                            'ppo_loss': update_stats['loss'],
                            'ppo_policy_loss': update_stats['policy_loss'],
                            'ppo_value_loss': update_stats['value_loss'],
                            'ppo_entropy': update_stats['entropy'],
                            'train_time_step': time_step,
                        })
                    break

        # Episode metrics
        total_steps = t + 1
        overlap_pct = {k: 100 * v / total_steps for k, v in overlap_counts.items()}
        returns.append(episode_return)
        # Total collisions this episode (from simulation)
        episode_collisions = getattr(sim, 'collision_count', 0)
        
        # Offset statistics
        avg_abs_offset = np.mean(offset_history) if offset_history else 0.0
        max_abs_offset = np.max(offset_history) if offset_history else 0.0
        if algo == "ts_dwa":
            # offset_history now stores alpha values (in meters)
            avg_alpha = avg_abs_offset
            max_alpha = max_abs_offset
            max_alpha_possible = sim.corridor_width / 1.25
        else:
            avg_alpha = float("nan")
            max_alpha = float("nan")
            max_alpha_possible = float("nan")

        termination_reason = "stuck" if stuck_terminated else ("goal" if prev_dist < 1.0 else "max_steps")
        print(f"Episode {ep+1}/{episodes} | Return: {episode_return:.2f} | Steps: {total_steps} | Term: {termination_reason}")
        print(f"  Overlaps - Free: {overlap_pct['none']:.1f}% | Person: {overlap_pct['person']:.1f}% | Door: {overlap_pct['door']:.1f}% | Both: {overlap_pct['both']:.1f}%")
        if algo == "ts_dwa":
            print(f"  Alpha - Avg: {avg_alpha:.3f}m | Max: {max_alpha:.3f}m | (range: [0, {max_alpha_possible:.3f}]m)")
        else:
            print(f"  w_max - Avg: {avg_abs_offset:.3f} rad/s | Max: {max_abs_offset:.3f} rad/s | (range: [{w_max_min:.3f}, {w_max_max:.3f}])")

        if use_wandb and wandb is not None:
            log_row = {
                'episode': ep + 1,
                'algo': algo,
                'return': episode_return,
                'steps': total_steps,
                'collisions': episode_collisions,
                'overlap_free_pct': overlap_pct['none'],
                'overlap_person_pct': overlap_pct['person'],
                'overlap_door_pct': overlap_pct['door'],
                'overlap_both_pct': overlap_pct['both'],
                'elapsed_min': (time.time() - start_time) / 60.0,
                'stuck_terminated': int(stuck_terminated),
            }
            if algo == "ts_dwa":
                log_row.update({
                    'avg_alpha': float(avg_alpha),
                    'max_alpha': float(max_alpha),
                })
            else:
                log_row.update({
                    'avg_w_max': float(avg_abs_offset),
                    'max_w_max': float(max_abs_offset),
                })
            wandb.log(log_row)
        
        # Periodic evaluation with exploration turned off
        if use_wandb and wandb is not None and (ep + 1) % 50 == 0:
            print(f"\n[Evaluation] Running 5 episodes with exploration turned off...")
            eval_avg_return = evaluate_policy(
                agent=agent,
                agent_type=agent_type,
                algo=algo,
                num_episodes=5,
                max_steps=max_steps,
                dt=dt,
                w_max_min=w_max_min,
                w_max_max=w_max_max,
            )
            print(f"[Evaluation] Average return (no exploration): {eval_avg_return:.2f}")
            wandb.log({
                'eval_avg_return': eval_avg_return,
                'eval_episode': ep + 1,
            })

    avg_return = float(np.mean(returns)) if returns else 0.0

    # Save policy
    os.makedirs('checkpoints', exist_ok=True)
    if agent_type in ("td3", "td3_lstm"):
        ckpt_name = f"{agent_type}_{algo}.pt"
    else:
        ckpt_name = "theta_qnet.pt"
    agent.save(os.path.join('checkpoints', ckpt_name))
    print(f"Saved policy to checkpoints/{ckpt_name}")

    if use_wandb and wandb is not None:
        wandb.summary['avg_return'] = avg_return
        wandb.finish()

    return avg_return


def run_optuna(study_name: str, num_trials: int, base_config: Dict[str, Any]) -> None:
    if optuna is None:
        raise RuntimeError("Optuna is not installed. Please install optuna to use hyperparameter search.")

    def objective(trial: 'optuna.trial.Trial') -> float:
        # Suggest hyperparameters
        config = dict(base_config)
        config.update({
            'lr_actor': trial.suggest_float('lr_actor', 1e-5, 5e-3, log=True),
            'lr_critic': trial.suggest_float('lr_critic', 1e-5, 5e-3, log=True),
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 384]),
            'gamma': trial.suggest_float('gamma', 0.90, 0.999),
            'k_epochs': trial.suggest_int('k_epochs', 5, 20),
            'eps_clip': trial.suggest_float('eps_clip', 0.1, 0.3),
            'update_timestep': trial.suggest_int('update_timestep', 1000, 5000, step=500),
        })
        # Shorter training for objective
        config['episodes'] = int(base_config.get('optuna_episodes', 12))
        avg_return = train(config, use_wandb=bool(base_config.get('wandb_during_optuna', False)),
                           run_name=f"optuna_trial_{trial.number}")
        # We want to maximize avg_return
        return avg_return

    storage = None  # In-memory; customize with SQLite if desired
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage)
    study.optimize(objective, n_trials=num_trials, gc_after_trial=True)

    print("Best trial:")
    print(f"  value: {study.best_trial.value}")
    print("  params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train navigation policy with PPO (agent-agnostic structure)')
    parser.add_argument('--algo', type=str, default='ts_dwa',
                        choices=['ts_dwa', 'dwa'],
                        help='Which local planner PPO is controlling: ts_dwa learns a heading offset; dwa learns nav.w_max.')
    parser.add_argument('--agent', type=str, default='ppo',
                        choices=['ppo', 'ppo_lstm', 'td3', 'td3_lstm'],
                        help='Policy model: feedforward PPO (ppo) or recurrent PPO-LSTM (ppo_lstm).')
    parser.add_argument('--w-max-min', type=float, default=-0.2 * math.pi,
                        help='(dwa) Minimum DWA nav.w_max (rad/s) when controlled by RL.')
    parser.add_argument('--w-max-max', type=float, default=math.pi,
                        help='(dwa) Maximum DWA nav.w_max (rad/s) when controlled by RL.')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-actor', type=float, default=None)
    parser.add_argument('--lr-critic', type=float, default=None)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--k-epochs', type=int, default=10)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--update-timestep', type=int, default=2000)
    parser.add_argument('--action-select-interval', type=int, default=1, help='Select a new action every N steps (default 1 = every step)')
    parser.add_argument(
        '--action-smooth-coef',
        type=float,
        default=0.0,
        help='Action jerkiness penalty coefficient: coef * (a_t - a_{t-1})^2 in normalized action space (applied once per decision/macro-step).',
    )
    parser.add_argument('--macro-step', action='store_true',
                        help='If set, treat action_select_interval as a true frame-skip: apply one action for N sim steps, '
                             'accumulate reward over those N steps, and store ONE PPO transition per N steps.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='PredictiveDWA')
    # Optuna options
    parser.add_argument('--optuna-trials', type=int, default=0)
    parser.add_argument('--optuna-study', type=str, default='theta_qnet_study')
    parser.add_argument('--optuna-episodes', type=int, default=12)
    parser.add_argument('--wandb-during-optuna', action='store_true')
    # Stuck detection arguments
    parser.add_argument('--stuck-detection-enabled', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Enable early termination when robot gets stuck (no progress for N steps). Default: True.')
    parser.add_argument('--stuck-threshold-steps', type=int, default=400,
                        help='Number of steps without progress before terminating episode (default: 200).')
    parser.add_argument('--stuck-min-progress', type=float, default=0.005,
                        help='Minimum distance improvement (meters) to count as progress (default: 0.1).')
    parser.add_argument('--stuck-penalty', type=float, default=-100.0,
                        help='Reward penalty when episode terminates due to stuck (default: -10.0).')
    parser.add_argument('--stuck-progress-reset-steps', type=int, default=10,
                        help='Consecutive progress steps needed to reset stuck counter (default: 10).')
    parser.add_argument('--stuck-no-progress-window', type=int, default=200,
                        help='Steps without progress before starting to count as stuck (allows oscillations, default: 20).')
    # TD3-specific arguments
    parser.add_argument('--td3-act-noise', type=float, default=0.15,
                        help='TD3 action noise for exploration during training (default: 0.15). Set to 0.0 for no exploration noise.')
    parser.add_argument('--td3-target-noise', type=float, default=0.3,
                        help='TD3 target policy smoothing noise (default: 0.3).')
    parser.add_argument('--td3-noise-clip', type=float, default=0.5,
                        help='TD3 noise clipping range (default: 0.5).')
    # Training data logging
    parser.add_argument('--log-training-data', action='store_true',
                        help='Log navigation features and actions to CSV files in simulation_training/ folder.')
    return parser.parse_args()


def main():
    args = parse_args()

    base_config: Dict[str, Any] = {
        'algo': args.algo,
        'agent': args.agent,
        'w_max_min': float(args.w_max_min),
        'w_max_max': float(args.w_max_max),
        'episodes': args.episodes,
        'max_steps': args.max_steps,
        'learning_rate': args.lr,
        'lr_actor': args.lr_actor if args.lr_actor is not None else args.lr,
        'lr_critic': args.lr_critic if args.lr_critic is not None else args.lr,
        'hidden_size': args.hidden,
        'gamma': args.gamma,
        'k_epochs': args.k_epochs,
        'eps_clip': args.eps_clip,
        'update_timestep': args.update_timestep,
        'action_select_interval': args.action_select_interval,
        'action_smooth_coef': float(args.action_smooth_coef),
        'macro_step': bool(args.macro_step),
        'seed': args.seed,
        'wandb_project': args.wandb_project,
        # Stuck detection
        'stuck_detection_enabled': bool(args.stuck_detection_enabled),
        'stuck_threshold_steps': int(args.stuck_threshold_steps),
        'stuck_min_progress': float(args.stuck_min_progress),
        'stuck_penalty': float(args.stuck_penalty),
        'stuck_progress_reset_steps': int(args.stuck_progress_reset_steps),
        'stuck_no_progress_window': int(args.stuck_no_progress_window),
        # Env defaults
        'dt': 1/60.0,
        'corridor_width': 4.0,
        'door_side': 'right',
        'num_people': 3,
        'people_speed_min': 1.1,
        'people_speed_max': 1.0,
        # Optuna-specific
        'optuna_episodes': args.optuna_episodes,
        'wandb_during_optuna': args.wandb_during_optuna,
        # TD3-specific
        'td3_act_noise': float(args.td3_act_noise),
        'td3_target_noise': float(args.td3_target_noise),
        'td3_noise_clip': float(args.td3_noise_clip),
        'log_training_data': bool(args.log_training_data),
    }

    if args.optuna_trials and args.optuna_trials > 0:
        run_optuna(args.optuna_study, args.optuna_trials, base_config)
    else:
        train(base_config, use_wandb=args.use_wandb)


if __name__ == '__main__':
    main()


