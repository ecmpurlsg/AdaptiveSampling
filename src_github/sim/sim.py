import pygame
import numpy as np
import random
import csv
import os
import math
from datetime import datetime
from typing import List, Tuple

from sim.person import Person
from sim.robot import Robot


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
    ):
        self.corridor_width = corridor_width
        self.door_side = door_side
        self.num_people = num_people
        self.people_speeds = people_speeds if people_speeds else [random.uniform(0.6, 1.2) for _ in range(num_people)]
        
        # Corridor dimensions
        self.corridor_length = corridor_length
        self.door_position = 0.5 * self.corridor_length  # Door is 40% along corridor
        self.corridor_bounds = {
            'x_min': 0,
            'x_max': self.corridor_length,
            'y_min': 0,
            'y_max': self.corridor_width
        }
        
        # Initialize agents
        self.robot = Robot((0.5, self.corridor_width - 1.0), 0.2, self.corridor_bounds, self.get_door_position())
        self.robot.set_goal((self.corridor_length - 1.5, self.corridor_width - 1.0))
        
        # Set door information for DWA
        if hasattr(self.robot.nav, 'set_door_info'):
            self.robot.nav.set_door_info(self.get_door_position(), self.door_side)
        
        self.people: List[Person] = []
        self.spawn_process = str(spawn_process).lower().strip()
        self.spawn_interval_range = spawn_interval_range
        self.spawn_min_gap = float(spawn_min_gap)
        self.spawn_rate_hz = float(spawn_rate_hz)

        self.spawn_timer = float(spawn_timer)
        # Keep backwards-compat: allow passing a fixed spawn_interval when using uniform.
        self.spawn_interval = float(spawn_interval)
        # If using the default "uniform" process, we sample the next interval just like before.
        self.spawn_interval = self._sample_next_spawn_interval(fallback=float(spawn_interval))
        
        # For visualization
        self.scale = 40  # pixels per meter
        self.offset = np.array([50, 50])

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

    def get_door_position(self) -> Tuple[float, float]:
        """Returns the precise (x,y) world coordinates of the door"""
        door_x = self.door_position
        if self.door_side == "right":
            door_y = self.corridor_width  # At the right wall
        else:
            door_y = 0.0  # At the left wall
        return (door_x, door_y)
    
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
            
        door_x = self.door_position
        door_pos = self.get_door_position()  # Get door position (at wall)
        # Spawn people slightly inward from the wall to keep them in corridor
        if self.door_side == "right":
            door_y = door_pos[1] - 0.5  # 0.5m inward from right wall
        else:
            door_y = door_pos[1] + 0.5  # 0.5m inward from left wall
            
        speed = self.people_speeds[len(self.people)]
        person = Person((door_x, door_y), 0.3, speed, self.door_side, self.corridor_width, self.corridor_length)
        self.people.append(person)

    def _sample_next_spawn_interval(self, fallback: float = 0.5) -> float:
        """
        Sample the next spawn headway (seconds).
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
    
    def step(self, dt: float):
        # Initialize start time on first step
        if self.start_time is None:
            self.start_time = datetime.now()
            self.previous_position = self.robot.position.copy()
                
        # Spawn people
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval and len(self.people) < self.num_people:
            self.spawn_person()
            self.spawn_timer = 0
            self.spawn_interval = self._sample_next_spawn_interval(fallback=self.spawn_interval)
            
        # Update agents
        #self.robot.add_gaussian_bump_to_path(alpha=-self.corridor_width/1.25, sigma=2.0, start = (0.5, self.corridor_width/1.25))
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
    
    def draw(self, screen):
        # Draw corridor
        corridor_rect = pygame.Rect(
            self.offset[0],
            self.offset[1],
            int(self.corridor_length * self.scale),
            int(self.corridor_width * self.scale)
        )
        pygame.draw.rect(screen, (0, 0, 0), corridor_rect, 3)  # Bolder black border (was 1)
        
        # Draw door
        door_pos = int(self.door_position * self.scale) + self.offset[0]
        if self.door_side == "right":
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

        # Draw corridor
        corridor_rect = pygame.Rect(
            self.offset[0],
            self.offset[1],
            int(self.corridor_length * self.scale),
            int(self.corridor_width * self.scale)
        )
        pygame.draw.rect(screen, (0, 0, 0), corridor_rect, 3)  # Bolder black border (was 1)
        
        # Draw door
        door_pos = int(self.door_position * self.scale) + self.offset[0]
        if self.door_side == "right":
            door_y = int(self.corridor_width * self.scale) + self.offset[1] - 10
            pygame.draw.line(screen, (0, 255, 0), (door_pos, door_y), (door_pos, door_y + 10), 3)
        else:
            door_y = self.offset[1]
            pygame.draw.line(screen, (0, 255, 0), (door_pos, door_y), (door_pos, door_y + 10), 3)
        
        # Draw goal if set (as black cross)
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

        # Draw corridor
        corridor_rect = pygame.Rect(
            self.offset[0],
            self.offset[1],
            int(self.corridor_length * self.scale),
            int(self.corridor_width * self.scale)
        )
        pygame.draw.rect(screen, (0, 0, 0), corridor_rect, 3)  # Bolder black border (was 1)
        
        # Draw door
        door_pos = int(self.door_position * self.scale) + self.offset[0]
        if self.door_side == "right":
            door_y = int(self.corridor_width * self.scale) + self.offset[1] - 10
            pygame.draw.line(screen, (0, 255, 0), (door_pos, door_y), (door_pos, door_y + 10), 3)
        else:
            door_y = self.offset[1]
            pygame.draw.line(screen, (0, 255, 0), (door_pos, door_y), (door_pos, door_y + 10), 3)
        
        # Draw goal if set (as black cross)
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
                # Check if this is a new collision (not already recorded)
                collision_id = f"robot_person_{id(person)}"
                
                # Only count if this collision hasn't been recorded recently
                # (to avoid counting the same collision multiple times)
                recent_collision = any(
                    abs(coll['timestamp'] - datetime.now().timestamp()) < 0.1  # Within 0.1 seconds
                    for coll in self.collision_history
                    if coll.get('collision_id') == collision_id
                )
                
                if not recent_collision:
                    self.collision_count += 1
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