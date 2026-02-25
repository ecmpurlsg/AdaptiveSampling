import pygame
import numpy as np
import random
from typing import List, Tuple, Optional
import math
from copy import deepcopy, copy

from sim.person import Person
from algo.simple import Simple
from algo.dwa import DWA
from algo.ts_dwa import TSDWA
from algo.ts_dwa_Try import TSDWA_2
from algo.global_planner import StraightLineGlobalPlanner
from algo.global_planner import AStarGlobalPlanner


def add_gaussian_bump_to_path(start: Tuple[float, float], goal: Tuple[float, float],
                               door_x: float, alpha: float, corridor_width: float, 
                               y_min: float = 0.0, sigma: float = 1.5, 
                               resolution: float = 0.25) -> List[np.ndarray]:
    """Fast standalone function to create a path with a Gaussian upward bump around the door.
    
    Creates a new straight-line path from start to goal, then applies a Gaussian bump.
    
    Parameters
    ----------
    start : Tuple[float, float]
        Starting position (x, y).
    goal : Tuple[float, float]
        Goal position (x, y).
    door_x : float
        X-coordinate of the door.
    alpha : float
        Maximum height of the bump (in meters). Will be clamped to stay within bounds.
    corridor_width : float
        Width of the corridor (used for bounds checking).
    y_min : float, optional
        Minimum y-coordinate (default: 0.0).
    sigma : float, optional
        Standard deviation of the Gaussian (controls width of the bump).
        Default is 1.5 meters.
    resolution : float, optional
        Resolution for waypoint spacing along the path (default: 0.25m).
    
    Returns
    -------
    List[np.ndarray]
        New path with Gaussian bump applied to straight-line path.
    """
    start_np = np.array(start, dtype=float)
    goal_np = np.array(goal, dtype=float)
    
    # Generate straight-line path from start to goal
    total_distance = np.linalg.norm(goal_np - start_np)
    if total_distance < 1e-6:
        return [start_np, goal_np]
    
    num_waypoints = max(2, int(total_distance / resolution))
    straight_path = []
    for i in range(num_waypoints):
        t = i / (num_waypoints - 1) if num_waypoints > 1 else 0.0
        waypoint = start_np + t * (goal_np - start_np)
        straight_path.append(waypoint)
    
    # Apply Gaussian bump to the straight path
    y_max = y_min + corridor_width
    modified_path = []
    
    for waypoint in straight_path:
        wp = np.array(waypoint, dtype=float)
        x = wp[0]
        y = wp[1]
        
        # Calculate distance from door along x-axis
        dx = x - door_x
        
        # Apply Gaussian bump: bump_height = alpha * exp(-0.5 * (dx/sigma)^2)
        bump_height = alpha * math.exp(-0.5 * (dx / sigma) ** 2)
        
        # Add bump upward (increase y-coordinate)
        new_y = y + bump_height
        
        # Clamp to corridor bounds
        new_y = max(y_min, min(y_max, new_y))
        
        modified_path.append(np.array([x, new_y], dtype=float))
    
    return modified_path


class Robot:
    def __init__(self, position: Tuple[float, float], radius: float, corridor_bounds: dict, door_position: Tuple[float, float]):
        self.position = np.array(position, dtype=float)
        self.radius = radius
        self.velocity = np.array([0.0, 0.0])
        self.max_speed = 1.2
        self.goal = None
        self.corridor_bounds = corridor_bounds
        self.door_position = door_position
        self.people = None
        # History of robot positions for rendering traversed path
        self.path_points: list[np.ndarray] = [self.position.copy()]

        # Visual proxemic footprint (semi-major/minor axes in meters)
        self.proxemic_axes = np.array([radius * 1.8, radius * 1.2], dtype=float)
        self.proxemic_color = (150, 150, 255, 80)

        # ------ Global planner (A* with door avoidance) ------
        corridor_length = self.corridor_bounds['x_max'] - self.corridor_bounds['x_min']
        corridor_width  = self.corridor_bounds['y_max'] - self.corridor_bounds['y_min']

        # Get door information from simulation
        door_pos = self.door_position
        # Determine door side based on door position
        door_side = "right"  # Default
        if door_pos[1] < corridor_width / 2:
            door_side = "left"

        self.global_planner = AStarGlobalPlanner(
            corridor_length, 
            corridor_width, 
            door_pos, 
            door_side,
            resolution=0.25,
            door_halo_radius=1.0,  # 1.0m radius around door (1.0m + 0.2m robot = 1.2m total threshold)
            consider_doors=False   # Force straight-line until door handling is desired
        )
        self.global_path = None  # will be initialised after goal is set

        # ------ Choosing the local planner type ------
    
        #self.nav = Simple(self.position, self.velocity, self.max_speed, self.goal, self.radius)
        self.nav = DWA(self.position, self.velocity, self.max_speed, self.goal, self.radius, self.corridor_bounds)
        #self.nav = TSDWA(self.position, self.velocity, self.max_speed, self.goal, self.radius, self.corridor_bounds)
        #self.nav = TSDWA_2(self.position, self.velocity, self.max_speed, self.goal, self.radius, self.corridor_bounds)


        self.nav_type = "ts_dwa"
        #self.nav_type = "ts_dwa"
        if self.nav_type == "ts_dwa":
            self.nav = TSDWA(self.position, self.velocity, self.max_speed, self.goal, self.radius, self.corridor_bounds)
        elif self.nav_type == "dwa":
            self.nav = DWA(self.position, self.velocity, self.max_speed, self.goal, self.radius, self.corridor_bounds)
        elif self.nav_type == "dwa_door_aware":
            self.nav = DWA(self.position, self.velocity, self.max_speed, self.goal, self.radius, self.corridor_bounds)
            self.nav.door_aware_sampling = True
        
        # ------ Choosing the local planner type ------

        # Learning
        self.state = None
        self.reward = None
        self.done = False

    def set_nav_type(self, nav_type: str):
        self.nav_type = nav_type
        if self.nav_type == "ts_dwa":
            self.nav = TSDWA(self.position, self.velocity, self.max_speed, self.goal, self.radius, self.corridor_bounds)
        elif self.nav_type == "dwa":
            self.nav = DWA(self.position, self.velocity, self.max_speed, self.goal, self.radius, self.corridor_bounds)
        elif self.nav_type == "dwa_door_aware":
            self.nav = DWA(self.position, self.velocity, self.max_speed, self.goal, self.radius, self.corridor_bounds)
            self.nav.door_aware_sampling = True
    
    def set_goal(self, goal: Tuple[float, float]):
        self.goal = np.array(goal)
        self.nav.set_goal(goal)
        # Compute (or recompute) global path whenever a new goal is set
        self.global_path = self.global_planner.plan(tuple(self.position), tuple(goal))
    
    def add_gaussian_bump_to_path(self, alpha: float, sigma: float = 1.5, 
                                   resolution: float = 0.25,
                                   start: Optional[Tuple[float, float]] = None) -> None:
        """Create a new path with a Gaussian upward bump around the door.
        
        Generates a fresh straight-line path from start (or current position) to goal,
        then applies a Gaussian bump. This replaces the existing global_path.
        
        Parameters
        ----------
        alpha : float
            Maximum height of the bump (in meters). The bump will be bounded by
            corridor_width, so alpha is clamped to [0, corridor_width - current_y].
        sigma : float, optional
            Standard deviation of the Gaussian (controls width of the bump).
            Default is 1.5 meters.
        resolution : float, optional
            Resolution for waypoint spacing along the path (default: 0.25m).
        start : Tuple[float, float], optional
            Starting position (x, y). If None, uses robot's current position.
        
        Modifies
        --------
        self.global_path : List[np.ndarray]
            The global path is replaced with a new path that has the Gaussian bump.
        """
        if self.goal is None:
            return
        
        # Use provided start or robot's current position
        start_pos = tuple(start) if start is not None else tuple(self.position)
        
        door_x = float(self.door_position[0])
        corridor_width = float(self.corridor_bounds['y_max'] - self.corridor_bounds['y_min'])
        y_min = float(self.corridor_bounds['y_min'])
        
        # Generate new path from start to goal with Gaussian bump
        self.global_path = add_gaussian_bump_to_path(
            start=start_pos,
            goal=tuple(self.goal),
            door_x=door_x,
            alpha=alpha,
            corridor_width=corridor_width,
            y_min=y_min,
            sigma=sigma,
            resolution=resolution
        )
    
    def update(self, dt: float, people: List[Person]):
        self.people = people
        if self.goal is None:
            return
        # Pass global path to TS-DWA; other planners ignore the extra arg
        if self.nav_type == "ts_dwa":
            self.velocity, self.position, self.goal = self.nav.update(dt, people, self.global_path)
        else:
            self.velocity, self.position, self.goal = self.nav.update(dt, people)

        # Append current position to path history (limit length to avoid unbounded growth)
        if not hasattr(self, 'path_points'):
            self.path_points = []
        self.path_points.append(self.position.copy())
        if len(self.path_points) > 5000:
            self.path_points = self.path_points[-5000:]

        # Get state, reward and done
        nav_info = self.get_navigation_info(2)
        self.state = {"nav_info": nav_info, "costmap": self.get_egocentric_costmap()}
        self.reward = nav_info["closest_obstacle_distance"]
        self.done = np.linalg.norm(self.position - self.goal) < 1.0

        return deepcopy(self.state), self.reward, self.done     

    def get_egocentric_costmap(self, size=4.0, resolution=0.1, inflation_radius=0.2):
        """
        Generate an egocentric costmap centered on the robot.
        
        Args:
            size: Size of the costmap in meters (will be square)
            resolution: Meters per pixel
            inflation_radius: How far obstacles influence the costmap
        
        Returns:
            A 2D numpy array (grayscale 0-255) where:
            - 0 = free space
            - 255 = obstacle
            - Values in between represent inflated obstacle areas
        """
        # Calculate grid dimensions
        grid_size = int(size / resolution)
        center_pixel = grid_size // 2
        costmap = np.zeros((grid_size, grid_size), dtype=np.uint8)
        
        # Get robot's current orientation
        robot_angle = self.nav.orientation
        
        # Transform function from world to robot-centric coordinates
        def world_to_robot(p):
            # Translate to robot center
            translated = p - self.position
            # Rotate to align with robot orientation
            rotated = np.array([
                -translated[0] * np.sin(robot_angle) + translated[1] * np.cos(robot_angle),
                translated[0] * np.cos(robot_angle) + translated[1] * np.sin(robot_angle)
            ])
            # Convert to grid coordinates
            grid_x = int(rotated[0] / resolution + center_pixel)
            grid_y = int(rotated[1] / resolution + center_pixel)
            return grid_x, grid_y
        
        # Add corridor boundaries to costmap
        if hasattr(self, 'corridor_bounds'):
            bounds = self.corridor_bounds
            # Check all four walls
            walls = [
                (bounds['x_min'], bounds['y_min'], bounds['x_max'], bounds['y_min']),  # Bottom
                (bounds['x_min'], bounds['y_max'], bounds['x_max'], bounds['y_max']),  # Top
                (bounds['x_min'], bounds['y_min'], bounds['x_min'], bounds['y_max']),  # Left
                (bounds['x_max'], bounds['y_min'], bounds['x_max'], bounds['y_max'])   # Right
            ]
            
            for wall in walls:
                # Sample points along the wall
                if wall[0] == wall[2]:  # Vertical wall
                    y_values = np.linspace(wall[1], wall[3], num=int(size/resolution)*2)
                    x_values = np.full_like(y_values, wall[0])
                else:  # Horizontal wall
                    x_values = np.linspace(wall[0], wall[2], num=int(size/resolution)*2)
                    y_values = np.full_like(x_values, wall[1])
                
                for x, y in zip(x_values, y_values):
                    grid_x, grid_y = world_to_robot(np.array([x, y]))
                    if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                        # Calculate distance from wall to robot
                        dist = np.linalg.norm([x - self.position[0], y - self.position[1]])
                        if dist <= size/2:
                            # Inflate the obstacle
                            for dx in range(-int(inflation_radius/resolution), int(inflation_radius/resolution)+1):
                                for dy in range(-int(inflation_radius/resolution), int(inflation_radius/resolution)+1):
                                    nx, ny = grid_x + dx, grid_y + dy
                                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                                        # Linear decay with distance
                                        obstacle_dist = np.sqrt(dx**2 + dy**2) * resolution
                                        if obstacle_dist <= inflation_radius:
                                            cost = int(255 * (1 - min(obstacle_dist / inflation_radius, 1)))
                                            costmap[ny, nx] = max(costmap[ny, nx], cost)
        
        # Add people to costmap
        for person in self.people:
            if not person.active:
                continue

            # Person position in robot-centric coordinates (float precision)
            translated = person.position - self.position
            rotated = np.array([
                -translated[0] * np.sin(robot_angle) + translated[1] * np.cos(robot_angle),
                translated[0] * np.cos(robot_angle) + translated[1] * np.sin(robot_angle)
            ])
            person_grid = rotated / resolution + center_pixel
            grid_x = int(round(person_grid[0]))
            grid_y = int(round(person_grid[1]))

            # Skip if completely outside
            if grid_x < 0 or grid_x >= grid_size or grid_y < 0 or grid_y >= grid_size:
                continue

            axes = getattr(person, 'proxemic_axes', np.array([person.radius, person.radius], dtype=float)).astype(float)
            a = max(float(axes[0]), 1e-4)
            b = max(float(axes[1]), 1e-4)
            
            # Offset ratio: 0.0 = person at center, 0.5 = person at rear edge
            ellipse_offset_ratio = 0.3  # Person is 30% back from center
            ellipse_offset = b * ellipse_offset_ratio  # Offset along major axis (forward direction)

            heading_world = getattr(person, 'heading_angle', 0.0)
            ellipse_angle = heading_world - robot_angle
            cos_t = math.cos(ellipse_angle)
            sin_t = math.sin(ellipse_angle)

            extent = max(a, b) + self.radius + inflation_radius
            cells_extent = int(math.ceil(extent / resolution))

            nx_min = max(0, grid_x - cells_extent)
            nx_max = min(grid_size - 1, grid_x + cells_extent)
            ny_min = max(0, grid_y - cells_extent)
            ny_max = min(grid_size - 1, grid_y + cells_extent)

            if nx_min > nx_max or ny_min > ny_max:
                continue

            for nx in range(nx_min, nx_max + 1):
                cell_x = (nx - center_pixel) * resolution
                for ny in range(ny_min, ny_max + 1):
                    cell_y = (ny - center_pixel) * resolution

                    rel_x = cell_x - rotated[0]
                    rel_y = cell_y - rotated[1]

                    # Rotate into ellipse-aligned frame
                    local_x = rel_x * cos_t - rel_y * sin_t
                    local_y = rel_x * sin_t + rel_y * cos_t
                    
                    # Apply offset so person is toward the rear of the ellipse
                    local_x_shifted = local_x + ellipse_offset

                    norm = math.sqrt((local_x_shifted / b) ** 2 + (local_y / a) ** 2)

                    if norm > 1e-6:
                        scale = 1.0 / norm
                        boundary_x_shifted = local_x_shifted * scale
                        boundary_y = local_y * scale
                        # Convert boundary back to unshifted frame for distance calculation
                        boundary_x = boundary_x_shifted - ellipse_offset
                        diff_x = local_x - boundary_x
                        diff_y = local_y - boundary_y
                        dist_to_boundary = math.hypot(diff_x, diff_y)
                        if norm < 1.0:
                            dist_to_boundary = -dist_to_boundary
                    else:
                        dist_to_boundary = -min(a, b)

                    clearance = dist_to_boundary - self.radius

                    if clearance <= inflation_radius:
                        cost = int(255 * (1 - min(max(clearance, 0.0) / inflation_radius, 1.0)))
                        costmap[ny, nx] = max(costmap[ny, nx], cost)

        # Add door semicircular inflation to costmap (visible only in costmap)
        if hasattr(self, 'door_position') and hasattr(self, 'corridor_bounds'):
            # Door position in world coordinates
            door_world = np.array(self.door_position, dtype=float)

            # Convert door position to robot frame (continuous)
            translated = door_world - self.position
            door_rotated = np.array([
                -translated[0] * np.sin(robot_angle) + translated[1] * np.cos(robot_angle),
                translated[0] * np.cos(robot_angle) + translated[1] * np.sin(robot_angle)
            ])

            # Determine door side relative to corridor centerline
            bounds = self.corridor_bounds
            corridor_mid_y = (bounds['y_min'] + bounds['y_max']) * 0.5
            door_side = "left" if door_world[1] < corridor_mid_y else "right"

            # Inward normal of the wall in WORLD frame (points into corridor)
            # Left wall (y near 0): +Y points inward; Right wall (y near max): -Y points inward
            n_world = np.array([0.0, 1.0]) if door_side == "left" else np.array([0.0, -1.0])

            # Rotate inward normal into ROBOT frame
            n_robot = np.array([
                -n_world[0] * np.sin(robot_angle) + n_world[1] * np.cos(robot_angle),
                n_world[0] * np.cos(robot_angle) + n_world[1] * np.sin(robot_angle)
            ])
            norm_n = np.linalg.norm(n_robot)
            if norm_n > 1e-9:
                n_robot = n_robot / norm_n

            # Inflation radius for the door halo (meters)
            door_inflation_radius = float(getattr(self.global_planner, 'door_halo_radius', 1.0))
            cells_extent = int(math.ceil(door_inflation_radius / resolution))

            # Door center in grid coordinates
            door_grid_x = int(round(door_rotated[0] / resolution + center_pixel))
            door_grid_y = int(round(door_rotated[1] / resolution + center_pixel))

            nx_min = max(0, door_grid_x - cells_extent)
            nx_max = min(grid_size - 1, door_grid_x + cells_extent)
            ny_min = max(0, door_grid_y - cells_extent)
            ny_max = min(grid_size - 1, door_grid_y + cells_extent)

            # Iterate cells around the door; inflate only the inward-facing half (semicircle)
            for nx in range(nx_min, nx_max + 1):
                cell_x = (nx - center_pixel) * resolution
                for ny in range(ny_min, ny_max + 1):
                    cell_y = (ny - center_pixel) * resolution

                    v_robot_x = cell_x - door_rotated[0]
                    v_robot_y = cell_y - door_rotated[1]
                    dist = math.hypot(v_robot_x, v_robot_y)

                    # Keep only the inward-facing half using dot(n_robot, v_robot) > 0
                    if dist <= door_inflation_radius and (n_robot[0] * v_robot_x + n_robot[1] * v_robot_y) > 0.0:
                        # Softer, less aggressive falloff that preserves higher intensity at the edge
                        t = min(max(dist / door_inflation_radius, 0.0), 1.0)
                        edge_intensity_ratio = 0.3  # Intensity at the boundary (0..1)
                        decay = (1.0 - t) ** 0.5      # Non-linear decay for smoother gradient
                        cost_ratio = edge_intensity_ratio + decay * (1.0 - edge_intensity_ratio)
                        cost = int(255 * cost_ratio)
                        costmap[ny, nx] = max(costmap[ny, nx], cost)
        
        return costmap

    def get_navigation_info(self, lookahead_distance: float = 2.0):
        """
        Returns key navigation information in the robot's egocentric frame.
        
        Args:
            door_position: (x,y) position of the door in world coordinates
            lookahead_distance: Distance along trajectory to find waypoint (meters)
            
        Returns:
            dict containing:
                - waypoint: (x,y) position 6m ahead in robot frame (forward is +x)
                - door_position: (x,y) of door in robot frame
                - linear_velocity: Current forward velocity (m/s)
                - angular_velocity: Current angular velocity (rad/s)
        """
        # Get current velocities
        linear_vel = self.velocity[0]
        angular_vel = self.velocity[1]
        
        # Convert door position to robot frame
        door_world = np.array(self.door_position)
        door_relative = door_world - self.position
        door_robot_frame = np.array([
            door_relative[0] * math.cos(self.nav.orientation) + door_relative[1] * math.sin(self.nav.orientation),
            -door_relative[0] * math.sin(self.nav.orientation) + door_relative[1] * math.cos(self.nav.orientation)
        ])

        # Angle to door in robot frame (radians)
        door_angle_rad = math.atan2(door_robot_frame[1], door_robot_frame[0])
        
        # Find waypoint N meters ahead along best trajectory
        waypoint_robot_frame = np.array([lookahead_distance, 0.0])  # Default straight ahead
        
        if self.nav.best_trajectory is not None and len(self.nav.best_trajectory) > 1:
            # Find point along trajectory that's approximately 6m ahead
            cumulative_dist = 0.0
            prev_point = self.nav.best_trajectory[0]
            
            for i in range(1, len(self.nav.best_trajectory)):
                current_point = self.nav.best_trajectory[i]
                segment_length = np.linalg.norm(current_point - prev_point)
                cumulative_dist += segment_length
                
                if cumulative_dist >= lookahead_distance:
                    # Interpolate to get exact point
                    alpha = (lookahead_distance - (cumulative_dist - segment_length)) / segment_length
                    waypoint_world = prev_point + alpha * (current_point - prev_point)
                    
                    # Convert to robot frame
                    waypoint_relative = waypoint_world - self.position
                    waypoint_robot_frame = np.array([
                        waypoint_relative[0] * math.cos(self.nav.orientation) + waypoint_relative[1] * math.sin(self.nav.orientation),
                        -waypoint_relative[0] * math.sin(self.nav.orientation) + waypoint_relative[1] * math.cos(self.nav.orientation)
                    ])
                    break
                prev_point = current_point
    
        # ----- Closest obstacle distance -----
        min_distance = float('inf')

        # Check people
        for person in self.people:
            if not person.active:
                continue
            dist = np.linalg.norm(person.position - self.position) - person.radius - self.radius
            min_distance = min(min_distance, max(0.0, dist))

        # Check corridor bounds (walls are line segments)
        bounds = self.corridor_bounds
        walls = [
            ((bounds['x_min'], bounds['y_min']), (bounds['x_max'], bounds['y_min'])),  # Bottom
            ((bounds['x_min'], bounds['y_max']), (bounds['x_max'], bounds['y_max'])),  # Top
            ((bounds['x_min'], bounds['y_min']), (bounds['x_min'], bounds['y_max'])),  # Left
            ((bounds['x_max'], bounds['y_min']), (bounds['x_max'], bounds['y_max']))   # Right
        ]

        def point_line_distance(p, a, b):
            """Shortest distance from point p to line segment ab."""
            p, a, b = np.array(p), np.array(a), np.array(b)
            ap = p - a
            ab = b - a
            t = np.dot(ap, ab) / np.dot(ab, ab)
            t = np.clip(t, 0, 1)
            closest = a + t * ab
            return np.linalg.norm(p - closest)

        for wall_start, wall_end in walls:
            dist_to_wall = point_line_distance(self.position, wall_start, wall_end) - self.radius
            min_distance = min(min_distance, max(0.0, dist_to_wall))
        
        return {
            'waypoint': waypoint_robot_frame,
            'door_position': door_robot_frame,
            'door_angle': door_angle_rad,
            'linear_velocity': linear_vel,
            'angular_velocity': angular_vel,
            'closest_obstacle_distance': min_distance
        }
    
    def draw(self, screen, scale, offset):
        if self.nav_type == "simple":
            pos = (self.position * scale + offset).astype(int)
            pygame.draw.circle(screen, (0, 0, 255), pos, int(self.radius * scale))
            # Draw direction indicator
            end_pos = (self.position + self.velocity * 0.5) * scale + offset
            pygame.draw.line(screen, (0, 255, 0), pos, end_pos.astype(int), 2)
        elif self.nav_type in ("dwa", "ts_dwa", "dwa_door_aware"):
            # Draw traversed path (polyline)
            if hasattr(self, 'path_points') and len(self.path_points) > 1:
                path_pts = [(p * scale + offset).astype(int) for p in self.path_points]
                pygame.draw.lines(screen, (128, 0, 128), False, path_pts, 2)

            # Draw robot
            pos = (self.position * scale + offset).astype(int)
            proxemic_axes = getattr(self, 'proxemic_axes', np.array([self.radius * 1.8, self.radius * 1.2], dtype=float))
            proxemic_color = getattr(self, 'proxemic_color', (150, 150, 255, 80))
            a_pix = max(int(proxemic_axes[0] * scale), 1)
            b_pix = max(int(proxemic_axes[1] * scale), 1)
            halo_surface = pygame.Surface((2 * a_pix, 2 * b_pix), pygame.SRCALPHA)
            pygame.draw.ellipse(halo_surface, proxemic_color, halo_surface.get_rect())
            angle_deg = -math.degrees(self.nav.orientation)
            if abs(angle_deg) > 1e-2:
                halo_surface = pygame.transform.rotate(halo_surface, angle_deg)
            halo_rect = halo_surface.get_rect(center=tuple(pos))
            screen.blit(halo_surface, halo_rect)

            pygame.draw.circle(screen, (0, 0, 255), pos, int(self.radius * scale))
            
            # Draw orientation
            end_pos = (self.position + np.array([
                math.cos(self.nav.orientation),
                math.sin(self.nav.orientation)
            ]) * self.radius * 1.5) * scale + offset
            pygame.draw.line(screen, (0, 255, 0), pos, end_pos.astype(int), 2)
            
            # # Draw global path as green line - COMMENTED OUT
            if self.global_path is not None and len(self.global_path) > 1:
                global_path_points = [(p * scale + offset).astype(int) for p in self.global_path]
                pygame.draw.lines(screen, (0, 255, 0), False, global_path_points, 3)  # Green, thick line
                
                # Draw waypoints along the global path
                for i, waypoint in enumerate(self.global_path):
                    wp_pos = (waypoint * scale + offset).astype(int)
                    if i == 0:  # Start point
                        pygame.draw.circle(screen, (0, 255, 0), wp_pos, 5)  # Green circle
                    elif i == len(self.global_path) - 1:  # End point
                        pygame.draw.circle(screen, (255, 0, 0), wp_pos, 5)  # Red circle
                    else:  # Intermediate waypoints
                        pygame.draw.circle(screen, (0, 200, 0), wp_pos, 3)  # Darker green
            
            # Draw all sampled trajectories (faint)
            for traj in self.nav.trajectories:
                points = [(p * scale + offset).astype(int) for p in traj]
                if len(points) > 1:
                    pygame.draw.lines(screen, (200, 200, 255, 50), False, points, 1)
            
            # Draw best trajectory
            if self.nav.best_trajectory is not None:
                points = [(p * scale + offset).astype(int) for p in self.nav.best_trajectory]
                if len(points) > 1:
                    pygame.draw.lines(screen, (0, 255, 255), False, points, 2)

            # Draw costmap (semi-transparent overlay)
            size = 4.0
            costmap = self.get_egocentric_costmap(size=size, resolution=0.1)
            
            # Create a colored version of the costmap (red for obstacles)
            colored_costmap = np.zeros((*costmap.shape, int(size)), dtype=np.uint8)  # Now with 4 channels (RGBA)
            colored_costmap[..., 0] = costmap  # Red channel
            colored_costmap[..., 1] = 0        # Green channel
            colored_costmap[..., 2] = 0        # Blue channel
            colored_costmap[..., 3] = costmap  # Alpha channel (for transparency)
            
            # Create surface from the costmap
            costmap_surface = pygame.surfarray.make_surface(colored_costmap[..., :3])  # Only use RGB channels
            costmap_surface.set_alpha(None)  # Clear any existing alpha
            alpha_surface = pygame.Surface(costmap_surface.get_size(), pygame.SRCALPHA)
            alpha_surface.blit(costmap_surface, (0, 0))
            alpha_surface.fill((255, 0, 0, 64), special_flags=pygame.BLEND_RGBA_MULT)  # More transparent (64 = ~25% opacity, was 128 = 50%)
            
            # Scale to world coordinates (4m x 4m -> pixels)
            costmap_size_pixels = int(size * scale)
            alpha_surface = pygame.transform.scale(alpha_surface, 
                                                (costmap_size_pixels, costmap_size_pixels))
            
            # Rotate to match robot orientation
            angle_deg = -math.degrees(self.nav.orientation)
            alpha_surface = pygame.transform.rotate(alpha_surface, angle_deg)
            
            # Calculate position (center the costmap on the robot)
            costmap_rect = alpha_surface.get_rect(center=pos)
            
            # Draw the costmap
            screen.blit(alpha_surface, costmap_rect)

            # Visualize navigation info with arrow indicators
            if hasattr(self, 'corridor_bounds'):
                nav_info = self.get_navigation_info(2)
                
                # Convert robot-frame positions to world coordinates for drawing
                def robot_to_world(robot_pos):
                    world_x = (self.position[0] + 
                            robot_pos[0] * math.cos(self.nav.orientation) - 
                            robot_pos[1] * math.sin(self.nav.orientation))
                    world_y = (self.position[1] + 
                            robot_pos[0] * math.sin(self.nav.orientation) + 
                            robot_pos[1] * math.cos(self.nav.orientation))
                    return np.array([world_x, world_y])
                
                # # Draw waypoint (6m ahead) - COMMENTED OUT
                # if 'waypoint' in nav_info:
                #     waypoint_world = robot_to_world(nav_info['waypoint'])
                #     waypoint_screen = (waypoint_world * scale + offset).astype(int)
                #     pygame.draw.circle(screen, (255, 255, 0), waypoint_screen, 8)  # Yellow
                
                # Draw door position
                if 'door_position' in nav_info:
                    door_world = robot_to_world(nav_info['door_position'])
                    door_screen = (door_world * scale + offset).astype(int)
                    pygame.draw.circle(screen, (0, 255, 0), door_screen, 6)  # Green
                
                # Draw velocity arrows
                arrow_length = 1  # Base length in pixels
                head_angle = math.pi/6  # 30 degrees
                head_length = 10  # pixels
                
                # Linear velocity arrow (blue)
                lin_vel = nav_info['linear_velocity']
                # Calculate end point in direction of orientation
                end_point = self.position + np.array([
                    math.cos(self.nav.orientation),
                    math.sin(self.nav.orientation)
                ]) * lin_vel * arrow_length / self.max_speed
                
                start_screen = pos
                end_screen = (end_point * scale + offset).astype(int)
                
                # Draw arrow line
                pygame.draw.line(screen, (0, 0, 255), start_screen, end_screen, 2)
                
                # Draw arrow head
                angle = math.atan2(end_screen[1]-start_screen[1], end_screen[0]-start_screen[0])
                points = [
                    end_screen,
                    (end_screen[0] - head_length * math.cos(angle - head_angle), end_screen[1] - head_length * math.sin(angle - head_angle)),
                    (end_screen[0] - head_length * math.cos(angle + head_angle), end_screen[1] - head_length * math.sin(angle + head_angle))
                ]
                pygame.draw.polygon(screen, (0, 0, 255), points)
                
                # # Angular velocity arrow (red)
                # ang_vel = nav_info['angular_velocity']
                # # Position arrow perpendicular to robot orientation
                # radius = 40  # pixels from center
                # angle_offset = math.pi/2 if ang_vel > 0 else -math.pi/2
                # start_point = self.position + np.array([
                #     math.cos(self.nav.orientation + angle_offset),
                #     math.sin(self.nav.orientation + angle_offset)
                # ]) * radius / scale
                
                # # Calculate arrow direction (tangential)
                # arrow_dir = np.array([
                #     -math.sin(self.nav.orientation + angle_offset),
                #     math.cos(self.nav.orientation + angle_offset)
                # ]) * math.copysign(1, ang_vel)
                
                # end_point = start_point + arrow_dir * abs(ang_vel) * arrow_length / self.nav.max_rotation
                
                # start_screen = (start_point * scale + offset).astype(int)
                # end_screen = (end_point * scale + offset).astype(int)
                
                # # Draw arrow line
                # pygame.draw.line(screen, (255, 0, 0), start_screen, end_screen, 2)
                
                # # Draw arrow head
                # angle = math.atan2(end_screen[1]-start_screen[1], end_screen[0]-start_screen[0])
                # points = [end_screen,
                #     (end_screen[0] - head_length * math.cos(angle - head_angle), end_screen[1] - head_length * math.sin(angle - head_angle)),
                #     (end_screen[0] - head_length * math.cos(angle + head_angle), end_screen[1] - head_length * math.sin(angle + head_angle))
                # ]
                # pygame.draw.polygon(screen, (255, 0, 0), points)

                # # Constants
                # # Red arrow (angular velocity indicator) - COMMENTED OUT
                # if lin_vel > 0:
                #     ang_vel = nav_info['angular_velocity']
                #     radius = 30  # Distance from robot center (pixels)
                #     arrow_length = 25 * (1 + ang_vel)  # Base length (scaled by ang_vel)
                #     head_length = 8
                #     head_angle = math.pi/6  # 30 degrees
                #
                #
                #     lin_vel_dir = np.array([math.cos(self.nav.orientation), 
                #                         math.sin(self.nav.orientation)])
                #
                #     # Compute perpendicular direction (90° rotation)
                #     perp_dir = np.array([-lin_vel_dir[1], lin_vel_dir[0]])  # Rotate 90° CCW
                #
                #     # Position arrow at a fixed radius from robot center
                #     start_point = self.position# + perp_dir * radius / scale
                #
                #     # Arrow direction depends on sign of angular velocity
                #     end_point = start_point + perp_dir * np.sign(ang_vel) * arrow_length * abs(ang_vel) / self.nav.max_rotation
                #
                #     # Convert to screen coordinates
                #     start_screen = (start_point * scale + offset).astype(int)
                #     end_screen = (end_point * scale + offset).astype(int)
                #
                #     # Draw arrow shaft
                #     pygame.draw.line(screen, (255, 0, 0), start_screen, end_screen, 2)
                #
                #     # Draw arrowhead
                #     angle = math.atan2(end_screen[1] - start_screen[1], end_screen[0] - start_screen[0])
                #     points = [
                #         end_screen,
                #         (end_screen[0] - head_length * math.cos(angle - head_angle),
                #         end_screen[1] - head_length * math.sin(angle - head_angle)),
                #         (end_screen[0] - head_length * math.cos(angle + head_angle),
                #         end_screen[1] - head_length * math.sin(angle + head_angle))
                #     ]
                #     pygame.draw.polygon(screen, (255, 0, 0), points)

                    #closest distance
                    #print(nav_info['closest_obstacle_distance'])
                                