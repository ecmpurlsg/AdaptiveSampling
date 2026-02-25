import pygame
import numpy as np
import random
import csv
import os
import math
from datetime import datetime
from typing import List, Tuple

from sim.person_complex import Person
from sim.robot_complex import Robot


class Simulation:
    def __init__(
        self,
        corridor_width: float = 4.0,
        door_side: str = "right",
        corridor_length: float = 15.0,
        num_people: int = 5,
        people_speeds: List[float] = None,
        spawn_interval: float = 0.5,
        spawn_timer: float = 0.5,
        # Spawning process:
        # - "uniform": headway ~ Uniform(spawn_interval_range[0], spawn_interval_range[1])  (default; matches old behavior)
        # - "poisson": headway = spawn_min_gap + Exp(spawn_rate_hz)  (shifted exponential / Poisson arrivals)
        spawn_process: str = "poisson",
        spawn_interval_range: Tuple[float, float] = (0.5, 2.0),
        spawn_min_gap: float = 0.3,
        spawn_rate_hz: float = 0.588,
        door_num: int = 1,
        door_spacing_m: float = 5.0,
    ):
        self.corridor_width = corridor_width
        self.door_num = door_num
        self.door_spacing_m = door_spacing_m
        
        
        # Calculate corridor length based on door_num: door_spacing_m*(door_num+1) meters
        self.corridor_length = door_spacing_m * (door_num + 1)
        
        # Initialize multiple doors
        self.doors = []
        base_spawn_rate = 0.558
        for i in range(door_num):
            door_x = door_spacing_m * (i + 1)  # Doors spaced by door_spacing_m meters
            # # First door (i=0) on right, second door (i=1) on left
            # if i == 0:
            #     door_side_i = 'right'
            # elif i == 1:
            #     door_side_i = 'left'
            # else:
            #     # For additional doors beyond 2, randomly assign
            door_side_i = "right"#random.choice(['left', 'right'])
            spawn_rate_hz_i = random.uniform(base_spawn_rate, 2.0 * base_spawn_rate)
            turn_dist_alpha_i = random.choice([1.0, 2.0, 3.0])
            
            door_info = {
                'x': door_x,
                'side': door_side_i,
                'spawn_rate_hz': spawn_rate_hz_i,
                'turn_dist_alpha': turn_dist_alpha_i,
                'spawn_timer': 0.0,
                'spawn_interval': 0.0,
            }
            self.doors.append(door_info)
        
        # For backwards compatibility, keep door_side and door_position
        # Use first door as default
        self.door_side = self.doors[0]['side'] if self.doors else door_side
        self.door_position = self.doors[0]['x'] if self.doors else (0.5 * self.corridor_length)
        
        self.num_people = num_people
        self.people_speeds = people_speeds if people_speeds else [random.uniform(0.6, 1.2) for _ in range(num_people)]
        
        self.corridor_bounds = {
            'x_min': 0,
            'x_max': self.corridor_length,
            'y_min': 0,
            'y_max': self.corridor_width
        }
        
        # Initialize agents - use first door position for robot initialization
        first_door_pos = self.get_door_position_by_index(0) if self.doors else (self.door_position, self.corridor_width)
        robot_start_pos = (0.5, corridor_width/1.25)
        self.robot = Robot(robot_start_pos, 0.2, self.corridor_bounds, first_door_pos)
        robot_goal_pos = (self.corridor_length - 1.5, corridor_width/1.25)
        self.robot.set_goal(robot_goal_pos)
        
        # Store starting position and goal for feature extraction (neural network trained with corridor_length=10)
        self.robot.start_position = np.array(robot_start_pos, dtype=float)
        self.robot.actual_goal = np.array(robot_goal_pos, dtype=float)
        
        # Store all door positions in robot for closest door finding
        self.robot.all_doors = [self.get_door_position_by_index(i) for i in range(len(self.doors))]
        
        # Store door x positions for door-based goal projection
        self.robot.door_x_positions = [door['x'] for door in self.doors]
        
        # Initialize last_passed_door_idx for door-based goal projection
        self.robot.last_passed_door_idx = -1
        
        # Set door information for DWA (use closest door)
        if hasattr(self.robot.nav, 'set_door_info'):
            closest_door_pos, closest_door_side = self.get_closest_door()
            self.robot.nav.set_door_info(closest_door_pos, closest_door_side)
        
        self.people: List[Person] = []
        self.spawn_process = str(spawn_process).lower().strip()
        self.spawn_interval_range = spawn_interval_range
        self.spawn_min_gap = float(spawn_min_gap)
        self.spawn_rate_hz = float(spawn_rate_hz)  # Keep for backwards compat

        # Initialize spawn timers for each door
        for door in self.doors:
            door['spawn_timer'] = float(spawn_timer)
            door['spawn_interval'] = self._sample_next_spawn_interval_for_door(door, fallback=float(spawn_interval))
        
        self.spawn_timer = float(spawn_timer)
        # Keep backwards-compat: allow passing a fixed spawn_interval when using uniform.
        self.spawn_interval = float(spawn_interval)
        # If using the default "uniform" process, we sample the next interval just like before.
        self.spawn_interval = self._sample_next_spawn_interval(fallback=float(spawn_interval))
        
        # For visualization - calculate scale to fit corridor in screen
        # Default screen size: 1000x400 pixels
        default_screen_width = 1000
        default_screen_height = 400
        padding = 50  # Padding on each side
        
        # Calculate scale to fit both length and width
        available_width = max(100, default_screen_width - 2 * padding)
        available_height = max(100, default_screen_height - 2 * padding)
        
        scale_x = available_width / self.corridor_length if self.corridor_length > 0 else 40
        scale_y = available_height / self.corridor_width if self.corridor_width > 0 else 40
        
        # Use the smaller scale to ensure both dimensions fit
        self.scale = min(scale_x, scale_y)
        # Ensure minimum scale for visibility
        if self.scale < 5:
            self.scale = 5
        self.offset = np.array([padding, padding])

        # For learning
        self.done = False
        
        # Data recording
        self.data_recording_enabled = True
        self.simulation_data = []
        self.start_time = None
        self.goal_reached_time = None
        self.total_distance_traveled = 0.0
        self.previous_position = None
        self.collision_count = 0
        self.collision_history = []  # Track collision timestamps and details
        self.collided_people = set()  # Track which people have already been counted as collided (one per person per episode)

    def get_door_position(self) -> Tuple[float, float]:
        """Returns the precise (x,y) world coordinates of the closest door (for backwards compatibility)"""
        return self.get_closest_door()[0]
    
    def get_door_position_by_index(self, index: int) -> Tuple[float, float]:
        """Returns the precise (x,y) world coordinates of a door by index"""
        if index < 0 or index >= len(self.doors):
            return (self.door_position, self.corridor_width if self.door_side == "right" else 0.0)
        door = self.doors[index]
        door_x = door['x']
        if door['side'] == "right":
            door_y = self.corridor_width  # At the right wall
        else:
            door_y = 0.0  # At the left wall
        return (door_x, door_y)
    
    def get_closest_door(self) -> Tuple[Tuple[float, float], str]:
        """Returns the next door (not yet passed) position and side to the robot.
        
        If the robot has passed a door's x-location, it considers the next door.
        If all doors are passed, returns the last door.
        """
        if not self.doors:
            return ((self.door_position, self.corridor_width if self.door_side == "right" else 0.0), self.door_side)
        
        robot_pos = self.robot.position
        robot_x = robot_pos[0]
        
        # Find the next door that the robot hasn't passed yet
        next_door_idx = None
        for i, door in enumerate(self.doors):
            door_x = door['x']
            if robot_x < door_x:  # Robot hasn't passed this door yet
                next_door_idx = i
                break
        
        # If all doors are passed, use the last door
        if next_door_idx is None:
            next_door_idx = len(self.doors) - 1
        
        closest_door = self.doors[next_door_idx]
        return (self.get_door_position_by_index(next_door_idx), closest_door['side'])
    
    def get_projected_goal(self) -> np.ndarray:
        """Compute the projected goal based on doors ahead of the robot.
        
        If there are 2+ doors ahead, the goal is the middle point between the next two doors,
        projected onto the global path. If there's only 1 door ahead, use the actual goal.
        The goal stays fixed until the robot passes the middle of each door.
        
        Returns:
            np.ndarray: The projected goal position (x, y)
        """
        robot_pos = self.robot.position
        robot_x = robot_pos[0]
        
        # Initialize last_passed_door_idx if not exists
        if not hasattr(self.robot, 'last_passed_door_idx'):
            self.robot.last_passed_door_idx = -1
        
        # Update last_passed_door_idx: check if robot has passed the middle of any door
        for i, door in enumerate(self.doors):
            door_x = door['x']
            door_middle_x = door_x  # Middle of door is at door_x
            if robot_x > door_middle_x and i > self.robot.last_passed_door_idx:
                self.robot.last_passed_door_idx = i
        
        # Find doors ahead of the last passed door
        doors_ahead = []
        for i, door in enumerate(self.doors):
            door_x = door['x']
            if i > self.robot.last_passed_door_idx:
                doors_ahead.append((i, door_x))
        
        # Sort by x-coordinate
        doors_ahead.sort(key=lambda t: t[1])
        
        if len(doors_ahead) >= 2:
            # Two or more doors ahead: goal is middle point between next two doors
            next_door_x = doors_ahead[0][1]
            second_door_x = doors_ahead[1][1]
            middle_x = (next_door_x + second_door_x) / 2.0
            
            # Project this x-coordinate onto the global path
            # The global path is a straight line from start to actual goal
            if hasattr(self.robot, 'start_position') and hasattr(self.robot, 'actual_goal'):
                start_pos = self.robot.start_position
                actual_goal = self.robot.actual_goal
                
                # Direction vector from start to actual goal
                direction_vec = actual_goal - start_pos
                direction_len = np.linalg.norm(direction_vec)
                
                if direction_len > 1e-6:
                    direction_unit = direction_vec / direction_len
                    
                    # Project middle_x onto the line from start to goal
                    # Find the parameter t such that start_x + t * direction_x = middle_x
                    if abs(direction_unit[0]) > 1e-6:
                        t = (middle_x - start_pos[0]) / direction_unit[0]
                        projected_goal = start_pos + t * direction_unit
                    else:
                        # Vertical line case (shouldn't happen in corridor)
                        projected_goal = actual_goal
                else:
                    projected_goal = actual_goal
            else:
                # Fallback: use y-coordinate from robot's current position
                projected_goal = np.array([middle_x, robot_pos[1]], dtype=float)
        elif len(doors_ahead) == 1:
            # Only one door ahead: use actual goal
            if hasattr(self.robot, 'actual_goal'):
                projected_goal = self.robot.actual_goal
            else:
                projected_goal = self.robot.goal
        else:
            # No doors ahead: use actual goal
            if hasattr(self.robot, 'actual_goal'):
                projected_goal = self.robot.actual_goal
            else:
                projected_goal = self.robot.goal
        
        return np.array(projected_goal, dtype=float)
    
    def spawn_person_with_target(self):
        if len(self.people) >= self.num_people:
            return
            
        door_x = self.door_position
        door_pos = self.get_door_position()  # Get door position (at wall)
        # Spawn people slightly inward from the wall to keep them in corridor
        if self.door_side == "right":
            door_y = door_pos[1] - 0.5  # 0.5m inward from right wall
            target = (door_x, -1.0)  # Move down out of corridor
        else:
            door_y = door_pos[1] + 0.5  # 0.5m inward from left wall
            target = (door_x, self.corridor_width + 1.0)  # Move up out of corridor
            
        speed = self.people_speeds[len(self.people)]
        self.people.append(Person((door_x, door_y), 0.3, speed, target))

    def spawn_person(self):
        if len(self.people) >= self.num_people:
            return
        
        # Find a door that's ready to spawn
        ready_doors = []
        for i, door in enumerate(self.doors):
            if door['spawn_timer'] >= door['spawn_interval']:
                ready_doors.append(i)
        
        if not ready_doors:
            return
        
        # Randomly select one of the ready doors
        door_idx = random.choice(ready_doors)
        door = self.doors[door_idx]
        door_pos = self.get_door_position_by_index(door_idx)
        
        # Spawn people slightly inward from the wall to keep them in corridor
        if door['side'] == "right":
            door_y = door_pos[1] - 0.5  # 0.5m inward from right wall
        else:
            door_y = door_pos[1] + 0.5  # 0.5m inward from left wall
            
        speed = self.people_speeds[len(self.people) % len(self.people_speeds)]
        person = Person(
            (door['x'], door_y), 
            0.3, 
            speed, 
            door['side'], 
            self.corridor_width, 
            self.corridor_length,
            turn_dist_alpha=door['turn_dist_alpha']
        )
        self.people.append(person)
        
        # Reset spawn timer for this door
        door['spawn_timer'] = 0.0
        door['spawn_interval'] = self._sample_next_spawn_interval_for_door(door)

    def _sample_next_spawn_interval(self, fallback: float = 0.5) -> float:
        """
        Sample the next spawn headway (seconds) - backwards compatibility method.
        - uniform: Uniform(low, high)
        - poisson: min_gap + Exp(rate)
        """
        proc = str(self.spawn_process).lower().strip()

        if proc == "poisson":
            rate = float(self.spawn_rate_hz)
            min_gap = max(0.0, float(self.spawn_min_gap))
            if rate <= 0.0 or (not math.isfinite(rate)):
                return max(0.0, float(fallback))
            # Exp(rate) in seconds (random.expovariate expects lambda=rate)
            return float(min_gap + random.expovariate(rate))

        # Default: uniform (matches previous behavior of resampling between 0.5 and 2.0 after each spawn)
        try:
            lo, hi = float(self.spawn_interval_range[0]), float(self.spawn_interval_range[1])
        except Exception:
            lo, hi = 0.5, 2.0
        if not math.isfinite(lo):
            lo = 0.5
        if not math.isfinite(hi):
            hi = 2.0
        if hi < lo:
            lo, hi = hi, lo
        lo = max(0.0, lo)
        hi = max(lo, hi)
        return float(random.uniform(lo, hi))
    
    def _sample_next_spawn_interval_for_door(self, door: dict, fallback: float = 0.5) -> float:
        """
        Sample the next spawn headway (seconds) for a specific door.
        - uniform: Uniform(low, high)
        - poisson: min_gap + Exp(rate) using door-specific spawn_rate_hz
        """
        proc = str(self.spawn_process).lower().strip()

        if proc == "poisson":
            rate = float(door['spawn_rate_hz'])
            min_gap = max(0.0, float(self.spawn_min_gap))
            if rate <= 0.0 or (not math.isfinite(rate)):
                return max(0.0, float(fallback))
            # Exp(rate) in seconds (random.expovariate expects lambda=rate)
            return float(min_gap + random.expovariate(rate))

        # Default: uniform
        try:
            lo, hi = float(self.spawn_interval_range[0]), float(self.spawn_interval_range[1])
        except Exception:
            lo, hi = 0.5, 2.0
        if not math.isfinite(lo):
            lo = 0.5
        if not math.isfinite(hi):
            hi = 2.0
        if hi < lo:
            lo, hi = hi, lo
        lo = max(0.0, lo)
        hi = max(lo, hi)
        return float(random.uniform(lo, hi))
    
    def step(self, dt: float):
        # Initialize start time on first step
        if self.start_time is None:
            self.start_time = datetime.now()
            self.previous_position = self.robot.position.copy()
                
        # Spawn people from all doors
        for door in self.doors:
            door['spawn_timer'] += dt
            if door['spawn_timer'] >= door['spawn_interval'] and len(self.people) < self.num_people:
                self.spawn_person()  # spawn_person handles door selection internally
        
        # Keep backwards compatibility timer
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval and len(self.people) < self.num_people:
            self.spawn_timer = 0
            self.spawn_interval = self._sample_next_spawn_interval(fallback=self.spawn_interval)
            
        # Update agents
        self.robot.add_gaussian_bump_to_path(alpha=-self.corridor_width/1.25, sigma=0.0)
        state, reward, done = self.robot.update(dt, self.people)
        for person in self.people:
            person.update(dt, self.people, self.robot, self.corridor_bounds)
        
        # Remove inactive people
        self.people = [p for p in self.people if p.active]
        
        # Check for collisions and record data if enabled
        if self.data_recording_enabled:
            self._check_collisions()
            self._record_simulation_data(dt, done)

        return state, reward, done
    
    def _update_scale_for_screen(self, screen):
        """Update scale and offset to fit the corridor in the screen."""
        if screen is None:
            return
        
        screen_width, screen_height = screen.get_size()
        padding = 50  # Padding on each side
        
        # Calculate scale to fit both length and width
        available_width = max(100, screen_width - 2 * padding)
        available_height = max(100, screen_height - 2 * padding)
        
        scale_x = available_width / self.corridor_length if self.corridor_length > 0 else 40
        scale_y = available_height / self.corridor_width if self.corridor_width > 0 else 40
        
        # Use the smaller scale to ensure both dimensions fit
        self.scale = min(scale_x, scale_y)
        # Ensure minimum scale for visibility
        if self.scale < 5:
            self.scale = 5
        self.offset = np.array([padding, padding])
    
    def draw(self, screen):
        # Update scale to fit screen
        self._update_scale_for_screen(screen)
        
        # Draw corridor
        corridor_rect = pygame.Rect(
            self.offset[0],
            self.offset[1],
            int(self.corridor_length * self.scale),
            int(self.corridor_width * self.scale)
        )
        pygame.draw.rect(screen, (0, 0, 0), corridor_rect, 3)  # Bolder black border (was 1)
        
        # Draw all doors
        for door in self.doors:
            door_pos = int(door['x'] * self.scale) + self.offset[0]
            if door['side'] == "right":
                door_y = int(self.corridor_width * self.scale) + self.offset[1] - 10
                pygame.draw.line(screen, (0, 255, 0), (door_pos, door_y), (door_pos, door_y + 10), 3)
            else:
                door_y = self.offset[1]
                pygame.draw.line(screen, (0, 255, 0), (door_pos, door_y), (door_pos, door_y + 10), 3)
        
        # Draw agents
        for person in self.people:
            person.draw(screen, self.scale, self.offset)
        self.robot.draw(screen, self.scale, self.offset)

    def draw_v0(self, screen, state_input=None):
        # Different from 'draw' function: Print the number of people, robot's speed, and robot's position
        # on the screen
        
        # Update scale to fit screen
        self._update_scale_for_screen(screen)

        # Draw corridor
        corridor_rect = pygame.Rect(
            self.offset[0],
            self.offset[1],
            int(self.corridor_length * self.scale),
            int(self.corridor_width * self.scale)
        )
        pygame.draw.rect(screen, (0, 0, 0), corridor_rect, 3)  # Bolder black border (was 1)
        
        # Draw all doors
        for door in self.doors:
            door_pos = int(door['x'] * self.scale) + self.offset[0]
            if door['side'] == "right":
                door_y = int(self.corridor_width * self.scale) + self.offset[1] - 10
                pygame.draw.line(screen, (0, 255, 0), (door_pos, door_y), (door_pos, door_y + 10), 3)
            else:
                door_y = self.offset[1]
                pygame.draw.line(screen, (0, 255, 0), (door_pos, door_y), (door_pos, door_y + 10), 3)
        
        # Draw actual goal if set (as black cross)
        if self.robot.goal is not None:
            goal_pos = (self.robot.goal * self.scale + self.offset).astype(int)
            cross_size = 10  # Size of cross arms in pixels
            # Draw horizontal line
            pygame.draw.line(screen, (0, 0, 0), 
                           (goal_pos[0] - cross_size, goal_pos[1]), 
                           (goal_pos[0] + cross_size, goal_pos[1]), 2)
            # Draw vertical line
            pygame.draw.line(screen, (0, 0, 0), 
                           (goal_pos[0], goal_pos[1] - cross_size), 
                           (goal_pos[0], goal_pos[1] + cross_size), 2)
        
        # Draw projected goal (used by neural network) as a blue circle
        if hasattr(self.robot, 'projected_goal') and self.robot.projected_goal is not None:
            proj_goal_pos = (self.robot.projected_goal * self.scale + self.offset).astype(int)
            pygame.draw.circle(screen, (0, 0, 255), proj_goal_pos, 8, 2)  # Blue circle, 8px radius, 2px width
        
        # Draw people
        for person in self.people:
            person.draw(screen, self.scale, self.offset)
        
        # Draw robot (with trajectories)
        self.robot.draw(screen, self.scale, self.offset)
        
        # Display info
        font = pygame.font.SysFont(None, 24)
        # info_text = [
        #     f"Position: ({self.robot.position[0]:.1f}, {self.robot.position[1]:.1f})",
        # ]

        # Optional: display state_input feature vector for debugging.
        # Assumes layout from learning.extract_nav_features:
        # [num_people,
        #  heading,
        #  goal_dx, goal_dy, door_dx, door_dy,
        #  p1_dx, p1_dy, p2_dx, p2_dy, p3_dx, p3_dy,
        #  dist_left, dist_right]
        if False:#state_input is not None:
            feat = np.asarray(state_input, dtype=float).tolist()
            if len(feat) >= 13:
                heading = feat[0]
                gdx, gdy = feat[1], feat[2]
                door_dx, door_dy = feat[3], feat[4]
                p1_dx, p1_dy = feat[5], feat[6]
                p2_dx, p2_dy = feat[7], feat[8]
                p3_dx, p3_dy = feat[9], feat[10]
                dist_left = feat[11]
                dist_right = feat[12]

                info_text.extend([
                    f"heading:    {heading:5.2f} rad ({heading * 180 / math.pi:5.1f}°)",
                    f"goal_rel:   ({gdx:5.2f}, {gdy:5.2f})",
                    f"door_rel:   ({door_dx:5.2f}, {door_dy:5.2f})",
                    f"p1_rel:     ({p1_dx:5.2f}, {p1_dy:5.2f})",
                    f"p2_rel:     ({p2_dx:5.2f}, {p2_dy:5.2f})",
                    f"p3_rel:     ({p3_dx:5.2f}, {p3_dy:5.2f})",
                    f"dist_left:  {dist_left:5.2f} m",
                    f"dist_right: {dist_right:5.2f} m",
                ])
            else:
                info_text.append(f"state_input (len={len(feat)}): {feat}")
        
        # for i, text in enumerate(info_text):
        #     text_surface = font.render(text, True, (0, 0, 0))
        #     screen.blit(text_surface, (10, 10 + i * 25))

    def draw_v1(self, screen):
        # Different from 'draw' function: Print the number of people, robot's speed, and robot's position
        # on the screen
        
        # Update scale to fit screen
        self._update_scale_for_screen(screen)

        # Draw corridor
        corridor_rect = pygame.Rect(
            self.offset[0],
            self.offset[1],
            int(self.corridor_length * self.scale),
            int(self.corridor_width * self.scale)
        )
        pygame.draw.rect(screen, (0, 0, 0), corridor_rect, 3)  # Bolder black border (was 1)
        
        # Draw all doors
        for door in self.doors:
            door_pos = int(door['x'] * self.scale) + self.offset[0]
            if door['side'] == "right":
                door_y = int(self.corridor_width * self.scale) + self.offset[1] - 10
                pygame.draw.line(screen, (0, 255, 0), (door_pos, door_y), (door_pos, door_y + 10), 3)
            else:
                door_y = self.offset[1]
                pygame.draw.line(screen, (0, 255, 0), (door_pos, door_y), (door_pos, door_y + 10), 3)
        
        # Draw actual goal if set (as black cross)
        if self.robot.goal is not None:
            goal_pos = (self.robot.goal * self.scale + self.offset).astype(int)
            cross_size = 10  # Size of cross arms in pixels
            # Draw horizontal line
            pygame.draw.line(screen, (0, 0, 0), 
                           (goal_pos[0] - cross_size, goal_pos[1]), 
                           (goal_pos[0] + cross_size, goal_pos[1]), 2)
            # Draw vertical line
            pygame.draw.line(screen, (0, 0, 0), 
                           (goal_pos[0], goal_pos[1] - cross_size), 
                           (goal_pos[0], goal_pos[1] + cross_size), 2)
        
        # Draw projected goal (used by neural network) as a blue circle
        if hasattr(self.robot, 'projected_goal') and self.robot.projected_goal is not None:
            proj_goal_pos = (self.robot.projected_goal * self.scale + self.offset).astype(int)
            pygame.draw.circle(screen, (0, 0, 255), proj_goal_pos, 8, 2)  # Blue circle, 8px radius, 2px width
        
        # Draw people
        for person in self.people:
            person.draw(screen, self.scale, self.offset)
        
        # Draw robot (with trajectories)
        self.robot.draw(screen, self.scale, self.offset)
        
        # Display info
        font = pygame.font.SysFont(None, 24)
        info_text = [
            #f"People: {len(self.people)}/{self.num_people}",
            #f"Robot Vel: {np.linalg.norm(self.robot.velocity):.2f} m/s",
            f"Position: ({self.robot.position[0]:.1f}, {self.robot.position[1]:.1f})",
            #f"Distance to door: {self.robot.door_position:.1f}"
        ]
        
        for i, text in enumerate(info_text):
            text_surface = font.render(text, True, (0, 0, 0))
            screen.blit(text_surface, (10, 10 + i * 25))
    
    def _check_collisions(self):
        """Check for collisions between robot and people"""
        robot_pos = self.robot.position
        robot_radius = self.robot.radius
        
        for person in self.people:
            if not person.active:
                continue
                
            person_pos = person.position
            person_radius = person.radius
            
            # Calculate distance between centers
            distance = np.linalg.norm(robot_pos - person_pos)
            
            # Check if collision occurs (distance < sum of radii)
            if distance < (robot_radius + person_radius):
                # Check if this person has already been counted as collided in this episode
                person_id = id(person)
                
                # Only count if this is the first collision with this person in this episode
                if person_id not in self.collided_people:
                    self.collided_people.add(person_id)
                    self.collision_count += 1
                    collision_id = f"robot_person_{person_id}"
                    collision_info = {
                        'collision_id': collision_id,
                        'timestamp': datetime.now().timestamp(),
                        'robot_position': robot_pos.copy(),
                        'person_position': person_pos.copy(),
                        'distance': distance,
                        'collision_count': self.collision_count
                    }
                    self.collision_history.append(collision_info)
                    print(f"Collision detected! Total collisions: {self.collision_count}")
    
    def _record_simulation_data(self, dt: float, done: bool):
        """Record simulation data for each time step"""
        current_time = datetime.now()
        elapsed_time = (current_time - self.start_time).total_seconds()
        
        # Calculate distance traveled
        if self.previous_position is not None:
            distance_step = np.linalg.norm(self.robot.position - self.previous_position)
            self.total_distance_traveled += distance_step
        
        # Calculate velocity magnitude
        velocity_magnitude = np.linalg.norm(self.robot.velocity)
        
        # Record goal reached time
        if done and self.goal_reached_time is None:
            self.goal_reached_time = elapsed_time
        
        # Store data point
        data_point = {
            'timestamp': current_time.isoformat(),
            'elapsed_time': elapsed_time,
            'robot_x': float(self.robot.position[0]),
            'robot_y': float(self.robot.position[1]),
            'robot_velocity_x': float(self.robot.velocity[0]),
            'robot_velocity_y': float(self.robot.velocity[1]),
            'robot_velocity_magnitude': velocity_magnitude,
            'total_distance_traveled': self.total_distance_traveled,
            'goal_reached': done,
            'num_people': len(self.people),
            'collision_count': self.collision_count,
            'dt': dt
        }
        
        self.simulation_data.append(data_point)
        
        # Update previous position for next iteration
        self.previous_position = self.robot.position.copy()
    
    def enable_data_recording(self, enabled: bool = True):
        """Enable or disable data recording"""
        self.data_recording_enabled = enabled
    
    def get_simulation_summary(self):
        """Get a summary of the simulation results"""
        if not self.simulation_data:
            return None
        
        # Calculate average velocity
        velocities = [point['robot_velocity_magnitude'] for point in self.simulation_data]
        avg_velocity = np.mean(velocities) if velocities else 0.0
        
        # Get final distance traveled
        final_distance = self.total_distance_traveled
        
        # Get time to reach goal
        time_to_goal = self.goal_reached_time if self.goal_reached_time else None
        
        # Get total simulation time
        total_time = self.simulation_data[-1]['elapsed_time'] if self.simulation_data else 0.0
        
        return {
            'total_simulation_time': total_time,
            'time_to_reach_goal': time_to_goal,
            'average_velocity': avg_velocity,
            'total_distance_traveled': final_distance,
            'goal_reached': self.goal_reached_time is not None,
            'total_collisions': self.collision_count,
            'total_data_points': len(self.simulation_data)
        }
    
    def export_data_to_csv(self, filename: str = None):
        """Export simulation data to CSV file"""
        if not self.simulation_data:
            print("No simulation data to export")
            return None
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_data_{timestamp}.csv"
        
        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Create data directory if it doesn't exist
        data_dir = "simulation_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        filepath = os.path.join(data_dir, filename)
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = self.simulation_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header
                writer.writeheader()
                
                # Write data
                writer.writerows(self.simulation_data)
            
            print(f"Simulation data exported to: {filepath}")
            
            # Print summary
            summary = self.get_simulation_summary()
            if summary:
                print("\nSimulation Summary:")
                print(f"  Total simulation time: {summary['total_simulation_time']:.2f} seconds")
                print(f"  Time to reach goal: {summary['time_to_reach_goal']:.2f} seconds" if summary['time_to_reach_goal'] else "  Goal not reached")
                print(f"  Average velocity: {summary['average_velocity']:.2f} m/s")
                print(f"  Total distance traveled: {summary['total_distance_traveled']:.2f} meters")
                print(f"  Total collisions: {summary['total_collisions']}")
                print(f"  Goal reached: {'Yes' if summary['goal_reached'] else 'No'}")
                print(f"  Total data points: {summary['total_data_points']}")
            
            return filepath
            
        except Exception as e:
            print(f"Error exporting data to CSV: {e}")
            return None
    
    def reset_data_recording(self):
        """Reset all recorded data"""
        self.simulation_data = []
        self.start_time = None
        self.goal_reached_time = None
        self.total_distance_traveled = 0.0
        self.previous_position = None
        self.collision_count = 0
        self.collision_history = []
        self.collided_people = set()  # Reset collided people tracking