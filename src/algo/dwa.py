import pygame
import numpy as np
import random
from typing import List, Tuple
import math

from sim.person import Person
from copy import deepcopy


class DWA:
    def __init__(self, position, velocity, max_speed, goal, radius, corridor_bounds):
        self.position = position
        self.velocity = velocity
        self.max_speed = max_speed
        self.goal = goal
        self.radius = radius
        self.corridor_bounds = corridor_bounds
        self.door_aware_sampling = False

        # DWA
        if np.linalg.norm(velocity) > 1e-3:
            self.orientation = math.atan2(velocity[1], velocity[0])
        else:
            self.orientation = 0.0  # Angle in radians (0 points to right)
        
        # DWA parameters
        self.max_rotation = math.pi / 2  # Max angular velocity (rad/s)
        self.max_accel = 1.0 * 4  # Max linear acceleration (m/s²)
        self.max_angular_accel = math.pi * 2  # Max angular acceleration (rad/s²)
        self.dt = 0.167  # Time step for simulation
        self.predict_time = 2.0  # How far ahead to predict (seconds)
        self.goal = None
        self.trajectories = []  # For visualization
        self.best_trajectory = None  # For visualization
        
        # Sample space parameters
        self.v_samples = 8  # Number of linear velocity samples
        self.w_samples = 8  # Number of angular velocity samples
        self.v_min = 0.0  # Minimum linear velocity
        self.v_max = max_speed  # Maximum linear velocity
        self.w_min = -math.pi  # Minimum angular velocity
        self.w_max = math.pi  # Maximum angular velocity
        
        # Door-aware sampling parameters
        self.door_position = None  # Will be set when door position is known
        self.door_side = None  # Will be set when door side is known
        self.door_influence_radius = 7.5  # Distance in meters where door affects sampling
        self.door_sampling_bias = 0.8  # How strongly to bias sampling (0-1)
        
        # Scoring weights
        self.weights = {
            'heading': 0.2,      # Higher weight on reaching goal (like FORWARDWEIGHT in reference)
            'goal': 0.4,      # Higher weight on reaching goal (like FORWARDWEIGHT in reference)
            'clearance': 0.3, # Moderate weight on clearance 
            'velocity': 0.1,  # Lower weight on velocity
        }
        
        # Robot dynamics
        self.v = 0.0  # Current linear velocity
        self.w = 0.0  # Current angular velocity
        
        # Door angle in robot frame (set by robot.py)
        self.door_angle_robot_frame = None

        # Wall checking parameters
        self.wall_check_points = 6  # Default value, will be updated dynamically

    def set_goal(self, goal: Tuple[float, float]):
        self.goal = np.array(goal)

    def set_door_info(self, door_position: Tuple[float, float], door_side: str):
        """Set the door position and side for door-aware sampling.
        
        Args:
            door_position: (x,y) position of the door in world coordinates
            door_side: "left" or "right" indicating which side of corridor the door is on
        """
        self.door_position = np.array(door_position)
        self.door_side = door_side

    def set_door_angle_robot_frame(self, angle_rad: float):
        """Set the door angle in robot reference frame (radians)."""
        self.door_angle_robot_frame = angle_rad

    def get_door_aware_sampling_params_v0(self):
        """Calculate sampling parameters based on proximity to door.
        
        Returns:
            tuple: (w_min, w_max) angular velocity limits biased away from door
        """
        if self.door_position is None or self.door_side is None:
            return self.w_min, self.w_max
            
        # Calculate euclidean distance to door
        dist_to_door = np.linalg.norm(self.position - self.door_position)
        # If far from door, use normal sampling
        if dist_to_door > self.door_influence_radius:
            return self.w_min, self.w_max
        # Use door angle in robot frame (set by robot.py)
        if self.door_angle_robot_frame is not None:
            door_angle_rad = float(self.normalize_angle(self.door_angle_robot_frame))
        else:
            # Fallback: Calculate door-relative angle
            door_direction = self.door_position - self.position
            door_angle_rad = math.atan2(door_direction[1], door_direction[0])
            door_angle_rad = self.normalize_angle(door_angle_rad - self.orientation)
        door_angle = math.degrees(door_angle_rad)

        # Compute a bounded, smooth influence in [0, 1] to avoid over-shrinking the
        # angular window (which can cause the robot to get stuck, especially in
        # narrow corridors).
        #
        # - dist_factor: 1.0 at the door, 0.0 at influence_radius
        # - ahead_factor: 1.0 if door is in front (cos(angle)>0), 0.0 if behind
        # - width_factor: reduce bias in narrow corridors (more likely to need "escape" turns)
        clamp01 = lambda x: max(0.0, min(1.0, float(x)))
        dist_factor = clamp01(1.0 - (dist_to_door / max(1e-6, float(self.door_influence_radius))))
        ahead_factor = clamp01(math.cos(door_angle_rad))  # door behind -> ~0

        width_factor = 1.0
        if hasattr(self, "corridor_bounds") and self.corridor_bounds is not None:
            try:
                corridor_width = float(self.corridor_bounds["y_max"] - self.corridor_bounds["y_min"])
                # No extra bias at/below 2.5m, full bias by 4.0m.
                width_factor = clamp01((corridor_width - 2.5) / 1.5)
            except Exception:
                width_factor = 1.0

        # Base influence: multiply (instead of add) so it doesn't exceed 1.0.
        base = dist_factor * ahead_factor
        # Smoothstep for stability (reduces sensitivity around the threshold).
        base = base * base * (3.0 - 2.0 * base)
        bias = clamp01(getattr(self, "door_sampling_bias", 0.0))
        influence = clamp01(base * width_factor * bias)

        # Guarantee we never shrink the opposite-turn side below this minimum.
        # This prevents "no escape" situations where the robot needs a small turn
        # toward the door-side to re-align in a narrow corridor.
        min_opposite_turn = math.radians(20.0)  # rad/s
        min_window_width = math.radians(30.0)   # rad/s

        # Determine which way to bias based on door side
        if self.door_side == "right":
            # Door on right -> prefer left turns by shrinking the *positive* side,
            # but never remove it completely.
            w_min = self.w_min
            w_max = self.w_max * (1.0 - influence)
            w_max = max(float(w_max), float(min_opposite_turn))
        else:
            # Door on left -> prefer right turns by shrinking the *negative* side,
            # but never remove it completely.
            w_min = self.w_min * (1.0 - influence)
            w_min = min(float(w_min), -float(min_opposite_turn))
            w_max = self.w_max

        # Ensure the sampling window isn't pathologically narrow.
        if (w_max - w_min) < min_window_width:
            mid = 0.5 * (w_min + w_max)
            half = 0.5 * min_window_width
            w_min = mid - half
            w_max = mid + half
            # Clamp to absolute limits
            w_min = max(float(self.w_min), float(w_min))
            w_max = min(float(self.w_max), float(w_max))
            
        return w_min, w_max

    def get_door_aware_sampling_params(self):
        """Calculate sampling parameters based on proximity to door.
        
        Returns:
            tuple: (w_min, w_max) angular velocity limits biased away from door
        """
        if self.door_position is None or self.door_side is None:
            return self.w_min, self.w_max
            
        # Calculate euclidean distance to door
        dist_to_door = np.linalg.norm(self.position - self.door_position)
        # If far from door, use normal sampling
        if dist_to_door > self.door_influence_radius:
            return self.w_min, self.w_max
        # Use door angle in robot frame (set by robot.py)
        if self.door_angle_robot_frame is not None:
            door_angle = math.degrees(self.door_angle_robot_frame)
        else:
            # Fallback: Calculate door-relative angle
            door_direction = self.door_position - self.position
            door_angle_rad = math.atan2(door_direction[1], door_direction[0])
            door_angle_rad = self.normalize_angle(door_angle_rad - self.orientation)
            door_angle = math.degrees(door_angle_rad)

        # Calculate influence factor (1 when at door, 0 at influence radius)
        # Use distance to door and orientation reletive to the door
        influence = (1.0 - (dist_to_door / self.door_influence_radius)) + (1 - (abs(door_angle) / 180.0))
        influence *= self.door_sampling_bias  # Scale by bias factor

        # Function to clamp values to a range
        clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

        # Determine which way to bias based on door side
        if self.door_side == "right":
            # Bias towards negative angles (left turns) when door is on right
            w_min = -math.pi
            #w_max = clamp(math.pi -(math.pi + math.pi/2)*influence, -math.pi, math.pi)
            #w_max = clamp(math.pi * (1 - influence) + math.radians(abs(door_angle)), -math.pi, math.pi)
            #w_max = (math.pi + math.radians(abs(door_angle))) * (1 - influence)
            if math.radians(door_angle) > 0 and math.radians(door_angle) < math.pi:
                #w_max = (math.pi + math.radians(door_angle)) * max(0, (1 - influence))
                w_min = -math.pi/2
                w_max = (math.pi + math.radians(door_angle)) * (1 - influence)
            else:
                w_max = math.pi
            #w_max = (math.pi + math.radians(abs(door_angle))) * max(0, (1 - influence))
            #w_max = math.pi * (1 - influence)
            #print(f"influence: {influence}")
        else:
            # Bias towards positive angles (right turns) when door is on left
            w_min = -math.pi * (1 - influence)
            w_max = math.pi
            
        return w_min, w_max


    def update(self, dt: float, people: List[Person]):
        if self.goal is None:
            return
        #print(f"self.door_sampling_bias: {self.door_sampling_bias}")
        # Generate dynamic window
        dw = self.dynamic_window()
        
        # Get door-aware sampling parameters
        if self.door_aware_sampling == True:
            w_min, w_max = self.get_door_aware_sampling_params()  # Door aware sampling
        else:
            #w_min, w_max = -math.pi, -math.pi*0.1
            w_min, w_max = self.w_min, self.w_max
        # Sample velocities and evaluate
        best_score = -float('inf')
        best_v, best_w = 0.0, 0.0
        self.trajectories = []  # Reset for visualization
        
        # Sample velocities like in the reference implementation
        # Create arrays of possible velocity changes (not absolute velocities)
        v_samples = np.linspace(max(dw[0], self.v_min), min(dw[1], self.v_max), num=self.v_samples)
        w_samples = np.linspace(max(dw[2], w_min), min(dw[3], w_max), num=self.w_samples)
        
        for v in v_samples:
            for w in w_samples:
                trajectory = self.predict_trajectory(v, w)
                self.trajectories.append(trajectory)  # For visualization
                
                # Calculate scores
                heading_score = self.heading_score(trajectory, v, w)
                goal_score = self.goal_score(trajectory)
                clearance_score = self.clearance_score(trajectory, people)
                
                # Velocity score: prefer higher speeds but don't penalize too much for lower speeds
                velocity_score = v / self.max_speed if v > 0 else 0
                
                # Total weighted score
                total_score = (self.weights['goal'] * goal_score +
                              self.weights['clearance'] * clearance_score +
                              self.weights['velocity'] * velocity_score)
                
                if total_score > best_score:
                    best_score = total_score
                    best_v, best_w = v, w
                    self.best_trajectory = trajectory  # For visualization
        
        # Update velocities with constraints
        self.v = best_v
        self.w = best_w
        
        # Update position and orientation
        self.orientation += self.w * dt
        self.orientation = self.normalize_angle(self.orientation)
        
        # Calculate velocity vector
        self.velocity = np.array([
            self.v * math.cos(self.orientation),
            self.v * math.sin(self.orientation)
        ])
        self.position += self.velocity * dt

        return self.velocity, self.position, self.goal

    def update_real(
        self,
        people: List[Person],
        costmap,
        position,
        orientation: float,
        global_path=None,
    ):
        """Real-world update: compute the best (v, w) and trajectories using a ROS costmap.

        This mirrors `update()` scoring, but:
        - `self.position` and `self.orientation` are set directly from sensors.
        - The robot state is NOT integrated forward in time (no pose propagation).
        - Clearance is evaluated from the costmap via `clearance_score_costmap()`.

        Parameters
        ----------
        people : List[Person]
            Currently unused for clearance in real mode (kept for API parity).
        costmap : object
            ROS costmap (OccupancyGrid-like or costmap_2d-like wrapper).
        position : array-like
            Current robot (x, y) in world/map frame.
        orientation : float
            Current robot yaw in radians (world/map frame).
        global_path : optional
            Ignored for vanilla DWA; accepted so Robot can call this uniformly.
        """
        if self.goal is None:
            return

        # Update pose from sensors
        self.position[0] = float(position[0])
        self.position[1] = float(position[1])
        self.orientation = float(orientation)

        # Generate dynamic window (uses internal v/w history + internal dt constraints)
        dw = self.dynamic_window()

        # Door-aware sampling params (optional)
        if self.door_aware_sampling is True:
            w_min, w_max = self.get_door_aware_sampling_params()
        else:
            w_min, w_max = -math.pi, math.pi

        best_score = -float("inf")
        best_v, best_w = 0.0, 0.0
        self.trajectories = []

        v_samples = np.linspace(max(dw[0], self.v_min), min(dw[1], self.v_max), num=self.v_samples)
        w_samples = np.linspace(max(dw[2], w_min), min(dw[3], w_max), num=self.w_samples)

        for v in v_samples:
            for w in w_samples:
                trajectory = self.predict_trajectory(v, w)
                self.trajectories.append(trajectory)

                heading_score = self.heading_score(trajectory, v, w)
                goal_score = self.goal_score(trajectory)
                clearance_score = self.clearance_score_costmap(trajectory, costmap)
                velocity_score = v / self.max_speed if v > 0 else 0.0

                total_score = (
                    self.weights["goal"] * goal_score
                    + self.weights["clearance"] * clearance_score
                    + self.weights["velocity"] * velocity_score
                )

                if total_score > best_score:
                    best_score = total_score
                    best_v, best_w = float(v), float(w)
                    self.best_trajectory = trajectory

        # Store selected command; do NOT integrate pose here
        self.v = best_v
        self.w = best_w
        self.velocity = np.array([self.v * math.cos(self.orientation), self.v * math.sin(self.orientation)])
        return self.velocity, self.position, self.goal
    
    def dynamic_window(self):
        # Calculate the dynamic window based on current velocities and constraints  
        # Velocity limits
        vs = [self.v_min, self.v_max, -self.max_rotation, self.max_rotation]
        
        # Add acceleration constraints
        vd = [
            self.v - self.max_accel * self.dt,
            self.v + self.max_accel * self.dt,
            self.w - self.max_angular_accel * self.dt,
            self.w + self.max_angular_accel * self.dt
        ]
        
        # Combine and clamp to absolute limits
        dw = [
            max(vs[0], vd[0]),  # Min linear velocity
            min(vs[1], vd[1]),  # Max linear velocity
            max(vs[2], vd[2]),  # Min angular velocity
            min(vs[3], vd[3])   # Max angular velocity
        ]
        
        return dw
    
    def predict_trajectory(self, v: float, w: float):
        # Simulate trajectory with given velocities
        time = 0
        trajectory = [self.position.copy()]
        current_pos = self.position.copy()
        current_theta = self.orientation
        
        while time < self.predict_time:
            current_theta += w * self.dt
            current_theta = self.normalize_angle(current_theta)
            
            current_pos[0] += v * math.cos(current_theta) * self.dt
            current_pos[1] += v * math.sin(current_theta) * self.dt
            trajectory.append(current_pos.copy())
            time += self.dt
            
        return np.array(trajectory)

    def heading_score(self, trajectory, v, w):
        """Score based on how well the trajectory heads toward the goal.
        
        Classic DWA uses the angle between current heading and goal direction.
        We also penalize excessive rotation to prevent wrap-around exploits.
        """
        if v < 0.01:  # Penalize near-zero velocity heavily
            return -1.0
        
        # Goal direction from current position
        goal_direction = self.goal - self.position
        desired_theta = math.atan2(goal_direction[1], goal_direction[0])
        
        # Heading after ONE time step (not full predict_time to avoid wrap-around)
        next_theta = self.orientation + w * self.dt
        next_theta = self.normalize_angle(next_theta)
        
        # Angular difference between next heading and goal direction
        angle_diff = abs(self.normalize_angle(desired_theta - next_theta))
        
        # Base score: 1.0 when aligned, 0.0 when perpendicular, negative when opposite
        base_score = 1.0 - (angle_diff / math.pi)
        
        # Penalize high angular velocities to prevent spinning (diminishing returns)
        # Trajectories with |w| > pi/2 get progressively penalized
        rotation_penalty = max(0.0, (abs(w) - math.pi/2) / (math.pi/2))  # 0 to 1 for w from pi/2 to pi
        
        return base_score - 0.3 * rotation_penalty
    
    def goal_score(self, trajectory):
        # Score based on distance to goal
        final_pos = trajectory[-1]
        distance_to_goal = np.linalg.norm(self.goal - final_pos)
        
        # Calculate progress towards goal (like reference implementation)
        previous_distance = np.linalg.norm(self.goal - self.position)
        progress = previous_distance - distance_to_goal
        
        # Normalize progress (positive for moving towards goal)
        if previous_distance == 0:
            return 1.0
             
        return progress / previous_distance  # Normalized progress score
    
    def clearance_score_v0(self, trajectory, people):
        if not people:
            return 1.0
            
        min_distance = float('inf')
        
        for person in people:
            if not person.active:
                continue
                
            for point in trajectory:
                distance = np.linalg.norm(point - person.position) - self.radius - person.radius
                if distance < min_distance:
                    min_distance = distance
                    if min_distance <= 0:  # Collision
                        return -float('inf')
        
        # Normalize to [0, 1] range with 1 being safe distance
        safe_distance = 1.0  # 1 meter is considered safe
        return min(min_distance / safe_distance, 1.0)

    def clearance_score(self, trajectory, people):
        """Calculate clearance score considering both people and corridor boundaries"""
        if not people and not hasattr(self, 'corridor_bounds'):
            return 1.0  # No obstacles or boundaries to consider
        #people = deepcopy(people)
        #for person in people:
        #    if np.linalg.norm(self.position - person.position) < 4.0:
        #        person.position = np.array([20.0, 20.0])

        min_distance = float('inf')
        
        # Check distance to people
        for person in people:
            if not person.active:
                continue
                
            for point in trajectory:
                distance = np.linalg.norm(point - person.position) - self.radius - person.radius
                if distance < min_distance:
                    min_distance = distance
                    if min_distance <= 0:  # Collision
                        return -float('inf')
        
        # Update wall check points based on trajectory length and corridor width
        corridor_width = self.corridor_bounds['y_max'] - self.corridor_bounds['y_min']
        self.wall_check_points = max(1, int(len(trajectory) / (0.5 * corridor_width)))
        
        # Check distance to corridor boundaries if they exist
        
        if hasattr(self, 'corridor_bounds'):
            bounds = self.corridor_bounds  # Need to take in account the we need to extract the walls from the cost map
            for point in trajectory[:self.wall_check_points]:  # Use dynamic parameter
                # Distance to left wall (x_min)
                dist_left = point[0] - bounds['x_min'] - self.radius
                # Distance to right wall (x_max)
                dist_right = bounds['x_max'] - point[0] - self.radius
                # Distance to bottom wall (y_min)
                dist_bottom = point[1] - bounds['y_min'] - self.radius
                # Distance to top wall (y_max)
                dist_top = bounds['y_max'] - point[1] - self.radius
                
                # Find the minimum distance to any boundary
                current_min = min(dist_left, dist_right, dist_bottom, dist_top)
                
                if current_min < min_distance:
                    min_distance = current_min
                    if min_distance <= 0:  # Collision with boundary
                        return -float('inf')
        
        # Normalize to [0, 1] range with 1 being safe distance
        safe_distance = 1.0  # 1 meter is considered safe
        return min(min_distance / safe_distance, 1.0)

    def clearance_score_from_costmap(self, trajectory, costmap):
        """
        Clearance score that uses a ROS1 costmap (e.g. global/local costmap_2d)
        instead of explicit Person objects, in a way that is closer to common
        DWA implementations:
          - Reject a trajectory if it collides with obstacles (here: any lethal /
            unknown costmap cell within `self.radius` of any trajectory point).
          - Otherwise compute clearance from the *minimum distance* to the nearest
            obstacle cell along the trajectory (minus `self.radius`), then map it
            into a [0,1] score.

        Args:
            trajectory: np.ndarray of shape (N, 2) in *world coordinates*.
            costmap:   Either:
                       - A `nav_msgs/OccupancyGrid`-style object (common in Python):
                           * costmap.info.resolution
                           * costmap.info.width / height
                           * costmap.info.origin.position.x / y
                           * costmap.data (flat, row-major, costs in [0,100], -1 unknown)
                       - Or a costmap_2d-like wrapper that exposes:
                           * costmap.worldToMap(wx, wy) -> (mx, my) or (ok, mx, my)
                           * costmap.getCost(mx, my) -> int
                           * costmap.getSizeInCellsX(), costmap.getSizeInCellsY()

        Behaviour:
            - Any obstacle cell within `self.radius` of the trajectory is treated
              as collision -> returns -inf immediately.
            - Otherwise we compute the min obstacle distance along the trajectory,
              subtract `self.radius`, and normalize.
            - Optionally, if self.corridor_bounds is defined, we also enforce
              hard corridor-wall constraints (like in clearance_score_v2):
              any point that crosses the bounds returns -inf.
        """
        # Fallback if no costmap is provided
        if costmap is None:
            return 1.0

        # --- 1. Optional hard corridor wall constraint ------------------------
        if hasattr(self, "corridor_bounds") and self.corridor_bounds is not None:
            b = self.corridor_bounds
            for p in trajectory:
                if (p[0] - b["x_min"] - self.radius) <= 0:  # left wall
                    return -float("inf")
                if (b["x_max"] - p[0] - self.radius) <= 0:  # right wall
                    return -float("inf")
                if (p[1] - b["y_min"] - self.radius) <= 0:  # bottom wall
                    return -float("inf")
                if (b["y_max"] - p[1] - self.radius) <= 0:  # top wall
                    return -float("inf")

        # --- 2. Costmap-based clearance (radius collision + min obstacle distance) ---
        lethal_threshold = 95
        treat_unknown_as_obstacle = True

        # Support both costmap_2d-like APIs and OccupancyGrid-like objects.
        has_costmap2d_api = hasattr(costmap, "worldToMap") and hasattr(costmap, "getCost")

        if has_costmap2d_api:
            try:
                width = int(costmap.getSizeInCellsX())
                height = int(costmap.getSizeInCellsY())
            except Exception:
                # If size can't be read, we can still try to query costs via worldToMap/getCost.
                width, height = None, None

            def world_to_map(wx: float, wy: float):
                out = costmap.worldToMap(wx, wy)
                # common patterns: (mx, my) or (ok, mx, my)
                if isinstance(out, tuple) and len(out) == 2:
                    return True, int(out[0]), int(out[1])
                if isinstance(out, tuple) and len(out) == 3:
                    return bool(out[0]), int(out[1]), int(out[2])
                return False, 0, 0

            def get_cost(mx: int, my: int) -> int:
                return int(costmap.getCost(mx, my))

            # Resolution is needed for distance conversion; if unavailable, fall back to 0.05m.
            res = float(getattr(costmap, "resolution", 0.05))

            def cell_center_world(mx: int, my: int):
                # If a mapToWorld exists, prefer it; else approximate with origin/res not available.
                if hasattr(costmap, "mapToWorld"):
                    out = costmap.mapToWorld(mx, my)
                    if isinstance(out, tuple) and len(out) == 2:
                        return float(out[0]), float(out[1])
                # Fallback: treat (mx,my) as already in meters (best-effort)
                return float(mx) * res, float(my) * res
        else:
            try:
                res = float(costmap.info.resolution)
                width = int(costmap.info.width)
                height = int(costmap.info.height)
                origin_x = float(costmap.info.origin.position.x)
                origin_y = float(costmap.info.origin.position.y)
                data = costmap.data
            except AttributeError:
                return 1.0

            def world_to_map(wx: float, wy: float):
                mx = int((wx - origin_x) / res)
                my = int((wy - origin_y) / res)
                ok = (0 <= mx < width) and (0 <= my < height)
                return ok, mx, my

            def get_cost(mx: int, my: int) -> int:
                return int(data[my * width + mx])

            def cell_center_world(mx: int, my: int):
                cx = origin_x + (mx + 0.5) * res
                cy = origin_y + (my + 0.5) * res
                return cx, cy

        # How far we search for obstacles around each trajectory point (meters).
        # If no obstacle is found within this range, we treat it as "very safe".
        max_search_dist_m = max(2.0, float(self.radius) + 1.0)
        search_r_cells = max(1, int(math.ceil(max_search_dist_m / res)))

        min_clearance_m = float("inf")

        for p in trajectory:
            wx, wy = float(p[0]), float(p[1])
            ok, mx0, my0 = world_to_map(wx, wy)
            if not ok:
                # Outside known map -> reject
                return -float("inf")

            local_min_obst_dist_m = float("inf")

            # Scan a local window for lethal/unknown cells and compute nearest distance
            for dy in range(-search_r_cells, search_r_cells + 1):
                my = my0 + dy
                if height is not None and (my < 0 or my >= height):
                    continue
                for dx in range(-search_r_cells, search_r_cells + 1):
                    mx = mx0 + dx
                    if width is not None and (mx < 0 or mx >= width):
                        continue

                    cell_cost = get_cost(mx, my)
                    is_unknown = cell_cost < 0
                    is_lethal = (cell_cost >= lethal_threshold) or (treat_unknown_as_obstacle and is_unknown)
                    if not is_lethal:
                        continue

                    cx, cy = cell_center_world(mx, my)
                    dist_m = math.hypot(wx - cx, wy - cy)
                    if dist_m < local_min_obst_dist_m:
                        local_min_obst_dist_m = dist_m

            # If no obstacles in the search window, treat as far away.
            if local_min_obst_dist_m == float("inf"):
                local_min_obst_dist_m = max_search_dist_m

            # Radius-based collision check
            if local_min_obst_dist_m <= float(self.radius):
                return -float("inf")

            clearance_here = local_min_obst_dist_m - float(self.radius)
            if clearance_here < min_clearance_m:
                min_clearance_m = clearance_here

        if min_clearance_m == float("inf"):
            return 1.0

        # Normalize clearance to [0,1] (1.0 means >= safe_distance away from obstacles)
        safe_distance = 1.0
        return min(max(min_clearance_m / safe_distance, 0.0), 1.0)

    def clearance_score_costmap(self, trajectory, costmap):
        """Costmap-based clearance score (DWA-style), with corridor bounds enforced.

        This is the public name used by `update_real()`. It delegates to
        `clearance_score_from_costmap()` which implements a radius-based collision
        check + minimum obstacle distance normalization.
        """
        return self.clearance_score_from_costmap(trajectory, costmap)

    def clearance_score_v2(self, trajectory, people):
        """Return a clearance score that:
        1. Considers *people* distances for grading (like clearance_score_v0).
        2. Treats *corridor walls* as hard constraints – a trajectory that
           intersects a wall is rejected (-inf) but proximity to the wall is
           **not** penalised.

        This lets the robot skim alongside walls when that is the safest route
        around dynamic obstacles, while still preventing actual collisions.
        """
        # --- 1. collision / scoring against dynamic obstacles (people) --------
        if not people:
            min_dist_people = float('inf')
        else:
            min_dist_people = float('inf')
            for person in people:
                if not person.active:
                    continue
                for p in trajectory:
                    dist = np.linalg.norm(p - person.position) - self.radius - person.radius
                    if dist < min_dist_people:
                        min_dist_people = dist
                        if min_dist_people <= 0:  # collision -> reject
                            return -float('inf')
        # --- 2. hard boundary check for corridor walls -----------------------
        if hasattr(self, 'corridor_bounds') and self.corridor_bounds is not None:
            b = self.corridor_bounds
            for p in trajectory:
                if (p[0] - b['x_min'] - self.radius) <= 0:  # left wall collision
                    return -float('inf')
                if (b['x_max'] - p[0] - self.radius) <= 0:  # right wall collision
                    return -float('inf')
                if (p[1] - b['y_min'] - self.radius) <= 0:  # bottom wall
                    return -float('inf')
                if (b['y_max'] - p[1] - self.radius) <= 0:  # top wall
                    return -float('inf')
        # --- 3. convert people clearance to a [0,1] score --------------------
        if min_dist_people == float('inf'):
            return 1.0  # no people encountered
        safe_distance = 1.0  # 1 m considered comfortable around people
        return min(min_dist_people / safe_distance, 1.0)

    def clearance_score_v3(self, trajectory, people):
        """Calculate wall clearance score that penalizes proximity to walls but allows getting close.
        
        This method treats walls as soft constraints - the robot can get close to walls
        but will prefer trajectories that maintain some distance from walls.
        """
        if not hasattr(self, 'corridor_bounds') or self.corridor_bounds is None:
            return 1.0  # No walls to consider
        
        min_wall_distance = float('inf')
        bounds = self.corridor_bounds
        
        # Check distance to corridor boundaries
        for point in trajectory:
            # Distance to left wall (x_min)
            dist_left = point[0] - bounds['x_min'] - self.radius
            # Distance to right wall (x_max)
            dist_right = bounds['x_max'] - point[0] - self.radius
            # Distance to bottom wall (y_min)
            dist_bottom = point[1] - bounds['y_min'] - self.radius
            # Distance to top wall (y_max)
            dist_top = bounds['y_max'] - point[1] - self.radius
            
            # Find the minimum distance to any boundary
            current_min = min(dist_left, dist_right, dist_bottom, dist_top)
            
            if current_min < min_wall_distance:
                min_wall_distance = current_min
                if min_wall_distance <= 0:  # Collision with boundary
                    return -float('inf')
        
        # Convert to a score: prefer some distance from walls but don't penalize too much
        # 0.3m is considered comfortable distance from walls
        comfortable_distance = 0.5
        if min_wall_distance >= comfortable_distance:
            return 1.0  # Full score when maintaining comfortable distance
        else:
            # Gradual penalty as we get closer to walls
            return min_wall_distance / comfortable_distance
    
    def normalize_angle(self, angle):
        # Normalize angle to [-π, π]
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def set_orientation(self, orientation: float):
        """Set the robot's orientation (in radians)."""
        self.orientation = self.normalize_angle(orientation)


