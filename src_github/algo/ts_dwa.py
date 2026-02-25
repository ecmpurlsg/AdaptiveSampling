import math
import numpy as np
from typing import List, Tuple
import scipy.stats as stats

from sim.person import Person  # keep parity with the original planner
from scipy.stats import beta


class TSDWA:
    """Targeted‑Sampling Dynamic Window Approach (TS‑DWA) local planner.

    This class follows the formulation in Shen & Soh, *J. Mechanisms and Robotics* (2024)
    and is a drop‑in replacement for the legacy :class:`DWA` class located in `dwa.py`.

    Core ideas implemented here
    ---------------------------
    1. **Polar translational sampling** biased towards the global‑plan heading.
    2. **Path‑curvature‑coupled angular samples** (ω = v * κ  + α * θ_offset).
    3. A *minimal* set of lateral / reverse samples for escape behaviour.
    4. Hooks to plug into the existing scoring pipeline (goal, clearance, velocity).
    5. **Weighted asymmetric sampling** with multiple distribution strategies.

    Asymmetric Sampling
    -------------------
    The planner supports multiple sampling strategies for creating biased heading
    distributions around the path heading:

    - **uniform**: Evenly-spaced samples (default, symmetric)
    - **power**: Power-law distribution based on left/right weight ratio
    - **gaussian**: Normal distribution with offset based on weight ratio
    - **beta**: Beta distribution using left/right weights as shape parameters

    Use `left_weight` and `right_weight` parameters to control sampling density:
    - Higher weight on one side = more samples on that side
    - Equal weights (1.0, 1.0) = symmetric sampling
    - Example: left_weight=1.0, right_weight=2.0 = bias towards right

    The public API intentionally mirrors the original DWA so existing simulation
    scripts (GUI, visualisers, episode runners) continue to work unchanged.
    """

    # ---------------------------------------------------------------------
    # ─── INITIALISATION ───────────────────────────────────────────────────
    # ---------------------------------------------------------------------

    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        max_speed: float,
        goal: Tuple[float, float],
        radius: float,
        corridor_bounds: dict,
        *,
        look_ahead_idx: int = 6,        # i_look in the paper
        n_heading: int = 9,             # n_asamp   (angular samples in polar space)
        n_speed: int = 9,               # n_vsamp   (speed magnitude samples)
        theta_range: float = math.pi/3,  # θ_range   (±30° cone)
        alpha_ph: float = 2.0,          # α_ph heading‑bias gain
        n_skip: int = 4,                # spacing between curvature calculation points
        sampling_strategy: str = "beta",  # Strategy: "uniform", "power", "gaussian", "beta"
        left_weight: float = 2,       # Sampling density weight for left side
        right_weight: float = 10,      # Sampling density weight for right side
        verbose: bool = False,          # Enable debug printing
    ) -> None:
        # Save parameters identical to the original DWA planner ------------------
        self.position = position
        self.velocity = velocity
        self.max_speed = max_speed
        self.goal = np.array(goal)
        self.radius = radius
        self.corridor_bounds = corridor_bounds

        if np.linalg.norm(velocity) > 1e-3:
            self.orientation = math.atan2(velocity[1], velocity[0])
        else:
            self.orientation = 0.0  # Angle in radians (0 points to right)

        # -------  TS‑specific parameters  --------------------------------------
        self.look_ahead_idx = look_ahead_idx
        self.n_heading = n_heading
        self.n_speed = n_speed
        self.theta_range = theta_range
        self.alpha_ph = alpha_ph
        self.n_skip = n_skip
        self.sampling_strategy = sampling_strategy
        self.left_weight = left_weight
        self.right_weight = right_weight
        self.verbose = verbose

        # Dynamics / limits replicate original values so scoring remains valid ----
        self.max_rotation = math.pi
        self.max_accel = 1.0 * 4
        self.max_angular_accel = math.pi * 2
        self.dt = 0.1
        self.predict_time = 2.0

        # Stored for visualisation ----------------------------------------------
        self.trajectories: List[np.ndarray] = []
        self.best_trajectory: np.ndarray | None = None

        # Runtime state ----------------------------------------------------------
        self.v = 0.0
        self.w = 0.0

        # Re‑use original scoring weights for now; user may tune externally
        self.weights = {"heading": 0.2, "goal": 0.4, "clearance": 0.3, "velocity": 0.1}

        # Wall checking parameters
        self.wall_check_points = 6  # Default value, will be updated dynamically

    # ---------------------------------------------------------------------
    # ─── PUBLIC API (matches original DWA) ────────────────────────────────
    # ---------------------------------------------------------------------

    def set_goal(self, goal: Tuple[float, float]):
        self.goal = np.array(goal)

    def update(self, dt: float, people: List[Person], global_path: List[np.ndarray]):
        """Main planner step.

        Parameters
        ----------
        dt : float
            Simulation time‐step.
        people : list[Person]
            Crowd for clearance scoring.
        global_path : list[np.ndarray]
            Sequence of way‑points produced by a global planner, **in map frame**.
            The first element is assumed to be the robot pose; the last is the goal.
        """
        if self.goal is None:
            return

        # ------------------------------------------------------------------
        # 1. Dynamic Window (same computation as legacy planner)
        dw = self._dynamic_window()
        
        # Debug: Print DW range and current state
        if self.verbose:
            print(f"\n=== Timestep ===")
            print(f"DW: v=[{dw[0]:5.2f}, {dw[1]:5.2f}], ω=[{dw[2]:6.3f}, {dw[3]:6.3f}] | Current: v={self.v:5.3f}, ω={self.w:6.3f}")

        # ------------------------------------------------------------------
        # 2. Extract path heading θ_ph as baseline, add agent offset
        theta_ph_path = self._extract_heading(global_path)
        agent_offset = getattr(self, 'agent_offset', 0.0)  # Agent's correction
        theta_ph = theta_ph_path + agent_offset  # Combined heading
        #print(f"agent_offset: {agent_offset}")
        
        # Set curvature to 0 (agent controls direction via offset)
        kappa = 0.0

        # ------------------------------------------------------------------
        # 3. Generate *targeted* samples biased along the global path
        samples = self._generate_ts_samples(dw, theta_ph, kappa)

        # ------------------------------------------------------------------
        # 4. Trajectory rollout + scoring (identical to baseline)
        best_score = -float("inf")
        best_v, best_w = 0.0, 0.0
        self.trajectories.clear()

        for v_sample, w_sample in samples:
            traj = self._predict_trajectory(v_sample, w_sample)
            self.trajectories.append(traj)

            h = self._heading_score(v_sample, w_sample)
            g = self._goal_score(traj)
            c = self._clearance_score(traj, people)
            vel_score = v_sample / self.max_speed

            score = (
                self.weights["heading"] * h
                + self.weights["goal"] * g
                + self.weights["clearance"] * c
                + self.weights["velocity"] * vel_score
            )
            if score > best_score:
                best_score = score
                best_v, best_w = v_sample, w_sample
                self.best_trajectory = traj

        # ------------------------------------------------------------------
        # 5. Command update & kinematics integration
        self.v, self.w = best_v, best_w
        
        # Debug: Print trajectory selection info
        if self.verbose:
            omega_samples = [w for v, w in samples]
            print(f"θ_ph_path={theta_ph_path:6.3f}, offset={agent_offset:6.3f}, θ_ph_final={theta_ph:6.3f} | "
                  f"Selected: v={best_v:5.3f}, ω={best_w:6.3f} | "
                  f"ω range=[{min(omega_samples):6.3f}, {max(omega_samples):6.3f}]")
        
        self.orientation = getattr(self, "orientation", 0.0) + self.w * dt
        self.orientation = self._normalize_angle(self.orientation)
        self.velocity = np.array([
            self.v * math.cos(self.orientation),
            self.v * math.sin(self.orientation),
        ])
        self.position += self.velocity * dt
        return self.velocity, self.position, self.goal

    def update_real(
        self,
        people: List[Person],
        costmap,
        position,
        orientation: float,
        global_path: List[np.ndarray],
    ):
        """Real-world update (sensor-driven pose, costmap clearance).

        - Sets `self.position` and `self.orientation` directly from sensors.
        - Uses `clearance_score_costmap()` for clearance (plus corridor bounds).
        - Does NOT integrate pose forward (no dt-based motion update).
        """
        if self.goal is None:
            return

        # Update pose from sensors
        self.position[0] = float(position[0])
        self.position[1] = float(position[1])
        self.orientation = float(orientation)

        # 1) dynamic window
        dw = self._dynamic_window()

        # 2) path heading + agent offset
        theta_ph_path = self._extract_heading(global_path)
        agent_offset = getattr(self, "agent_offset", 0.0)
        theta_ph = theta_ph_path + agent_offset
        kappa = 0.0

        # 3) targeted samples
        samples = self._generate_ts_samples(dw, theta_ph, kappa)

        # 4) rollout + scoring
        best_score = -float("inf")
        best_v, best_w = 0.0, 0.0
        self.trajectories.clear()

        for v_sample, w_sample in samples:
            traj = self._predict_trajectory(v_sample, w_sample)
            self.trajectories.append(traj)

            h = self._heading_score(v_sample, w_sample)
            g = self._goal_score(traj)
            c = self.clearance_score_costmap(traj, costmap)
            vel_score = v_sample / self.max_speed if self.max_speed > 1e-9 else 0.0

            score = (
                self.weights["heading"] * h
                + self.weights["goal"] * g
                + self.weights["clearance"] * c
                + self.weights["velocity"] * vel_score
            )
            if score > best_score:
                best_score = score
                best_v, best_w = float(v_sample), float(w_sample)
                self.best_trajectory = traj

        # 5) store selected command; do NOT integrate pose
        self.v, self.w = best_v, best_w
        self.velocity = np.array([self.v * math.cos(self.orientation), self.v * math.sin(self.orientation)])
        return self.velocity, self.position, self.goal

    def clearance_score_costmap(self, trajectory: np.ndarray, costmap):
        """Costmap-based clearance score (DWA-style), with corridor bounds enforced.

        Behaviour matches `algo.dwa.DWA.clearance_score_from_costmap()`:
        - reject if any lethal/unknown cell is within `self.radius`
        - else score = min_clearance / safe_distance clipped to [0,1]
        """
        if costmap is None:
            return 1.0

        # --- 1) Optional hard corridor wall constraint ---
        if hasattr(self, "corridor_bounds") and self.corridor_bounds is not None:
            b = self.corridor_bounds
            for p in trajectory:
                if (p[0] - b["x_min"] - self.radius) <= 0:
                    return -float("inf")
                if (b["x_max"] - p[0] - self.radius) <= 0:
                    return -float("inf")
                if (p[1] - b["y_min"] - self.radius) <= 0:
                    return -float("inf")
                if (b["y_max"] - p[1] - self.radius) <= 0:
                    return -float("inf")

        lethal_threshold = 95
        treat_unknown_as_obstacle = True

        has_costmap2d_api = hasattr(costmap, "worldToMap") and hasattr(costmap, "getCost")

        if has_costmap2d_api:
            try:
                width = int(costmap.getSizeInCellsX())
                height = int(costmap.getSizeInCellsY())
            except Exception:
                width, height = None, None

            def world_to_map(wx: float, wy: float):
                out = costmap.worldToMap(wx, wy)
                if isinstance(out, tuple) and len(out) == 2:
                    return True, int(out[0]), int(out[1])
                if isinstance(out, tuple) and len(out) == 3:
                    return bool(out[0]), int(out[1]), int(out[2])
                return False, 0, 0

            def get_cost(mx: int, my: int) -> int:
                return int(costmap.getCost(mx, my))

            res = float(getattr(costmap, "resolution", 0.05))

            def cell_center_world(mx: int, my: int):
                if hasattr(costmap, "mapToWorld"):
                    out = costmap.mapToWorld(mx, my)
                    if isinstance(out, tuple) and len(out) == 2:
                        return float(out[0]), float(out[1])
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

        max_search_dist_m = max(2.0, float(self.radius) + 1.0)
        search_r_cells = max(1, int(math.ceil(max_search_dist_m / res)))

        min_clearance_m = float("inf")

        for p in trajectory:
            wx, wy = float(p[0]), float(p[1])
            ok, mx0, my0 = world_to_map(wx, wy)
            if not ok:
                return -float("inf")

            local_min_obst_dist_m = float("inf")

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

            if local_min_obst_dist_m == float("inf"):
                local_min_obst_dist_m = max_search_dist_m

            if local_min_obst_dist_m <= float(self.radius):
                return -float("inf")

            clearance_here = local_min_obst_dist_m - float(self.radius)
            if clearance_here < min_clearance_m:
                min_clearance_m = clearance_here

        if min_clearance_m == float("inf"):
            return 1.0

        safe_distance = 1.0
        return min(max(min_clearance_m / safe_distance, 0.0), 1.0)

    # ------------------------------------------------------------------
    # ─── INTERNAL UTILITIES ─────────────────────────────────────────────
    # ------------------------------------------------------------------

    def _generate_weighted_headings(
        self, theta_ph: float, theta_range: float, n_samples: int
    ) -> np.ndarray:
        """Generate heading angles using the configured sampling strategy.
        
        Parameters
        ----------
        theta_ph : float
            Center heading (path heading in robot frame).
        theta_range : float
            Angular extent to sample on each side of theta_ph.
        n_samples : int
            Number of heading samples to generate.
            
        Returns
        -------
        np.ndarray
            Array of heading angles in radians.
        """
        if self.sampling_strategy == "uniform":
            return self._uniform_sampling(theta_ph, theta_range, n_samples)
        elif self.sampling_strategy == "power":
            return self._power_law_sampling(theta_ph, theta_range, n_samples)
        elif self.sampling_strategy == "gaussian":
            return self._gaussian_sampling(theta_ph, theta_range, n_samples)
        elif self.sampling_strategy == "beta":
            return self._beta_sampling_v0(theta_ph, theta_range, n_samples)
        else:
            raise ValueError(
                f"Unknown sampling_strategy '{self.sampling_strategy}'. "
                f"Expected one of: 'uniform', 'power', 'gaussian', 'beta'."
            )

    def _uniform_sampling(
        self, theta_ph: float, theta_range: float, n_samples: int
    ) -> np.ndarray:
        """Uniform (evenly-spaced) sampling strategy.
        
        This is the baseline symmetric sampling approach. Generates samples
        evenly distributed across the full angular range.
        
        Parameters
        ----------
        theta_ph : float
            Center heading angle.
        theta_range : float
            Angular extent on each side.
        n_samples : int
            Number of samples to generate.
            
        Returns
        -------
        np.ndarray
            Evenly-spaced heading angles.
        """
        return np.linspace(
            theta_ph - theta_range,
            theta_ph + theta_range,
            n_samples
        )

    def _power_law_sampling(
        self, theta_ph: float, theta_range: float, n_samples: int
    ) -> np.ndarray:
        """Power-law biased sampling strategy.
        
        Applies a power transformation to create asymmetric sampling density.
        The power parameter is derived from the ratio of right_weight to left_weight:
        - power > 1: More samples on the left (smaller angles)
        - power < 1: More samples on the right (larger angles)
        - power = 1: Uniform sampling
        
        Parameters
        ----------
        theta_ph : float
            Center heading angle.
        theta_range : float
            Angular extent on each side.
        n_samples : int
            Number of samples to generate.
            
        Returns
        -------
        np.ndarray
            Power-law distributed heading angles.
        """
        # Calculate power from weight ratio
        # right_weight > left_weight means we want more samples on right (power < 1)
        power = self.left_weight / (self.right_weight + 1e-6)
        
        # Generate uniform samples in [0, 1]
        uniform = np.linspace(0, 1, n_samples)
        
        # Apply power transformation
        biased = uniform ** power
        
        # Map to angular range [theta_ph - theta_range, theta_ph + theta_range]
        headings = theta_ph - theta_range + biased * (2 * theta_range)
        
        return headings

    def _gaussian_sampling(
        self, theta_ph: float, theta_range: float, n_samples: int
    ) -> np.ndarray:
        """Gaussian (normal) distribution sampling strategy.
        
        Generates samples from a normal distribution with the mean offset
        based on the weight ratio. Uses deterministic percentile-based sampling
        to avoid randomness.
        
        Weight interpretation:
        - right_weight > left_weight: Mean shifts right (positive offset)
        - left_weight > right_weight: Mean shifts left (negative offset)
        - equal weights: Centered at theta_ph
        
        Parameters
        ----------
        theta_ph : float
            Center heading angle.
        theta_range : float
            Angular extent on each side.
        n_samples : int
            Number of samples to generate.
            
        Returns
        -------
        np.ndarray
            Gaussian-distributed heading angles clipped to range.
        """
        # Calculate offset based on weight ratio
        # Normalized difference gives direction and magnitude of bias
        weight_sum = self.left_weight + self.right_weight + 1e-6
        weight_diff = (self.right_weight - self.left_weight) / weight_sum
        offset = weight_diff * theta_range * 0.5  # Scale to half range
        
        # Standard deviation: use a fraction of range to ensure good coverage
        std_dev = theta_range / 2.5
        
        # Generate deterministic samples using percentiles (inverse CDF)
        # Evenly spaced percentiles from 0.5% to 99.5% to avoid extreme tails
        percentiles = np.linspace(0.005, 0.995, n_samples)
        
        # Convert percentiles to z-scores (inverse CDF of standard normal)
        # Using approximation for inverse error function
        z_scores = np.sqrt(2) * self._erfinv(2 * percentiles - 1)
        
        # Transform to actual heading angles
        headings = theta_ph + offset + z_scores * std_dev
        
        # Clip to stay within the allowed range
        headings = np.clip(
            headings,
            theta_ph - theta_range,
            theta_ph + theta_range
        )
        
        return headings

    def _erfinv(self, x: np.ndarray) -> np.ndarray:
        """Approximate inverse error function for Gaussian sampling.
        
        Uses a polynomial approximation that's accurate enough for sampling.
        
        Parameters
        ----------
        x : np.ndarray
            Input values in range [-1, 1].
            
        Returns
        -------
        np.ndarray
            Approximate inverse error function values.
        """
        # Simple polynomial approximation (accurate to ~0.01)
        a = 0.147
        sign = np.sign(x)
        x = np.abs(x)
        
        ln_term = np.log(1 - x * x + 1e-10)
        term1 = 2 / (np.pi * a) + ln_term / 2
        term2 = ln_term / a
        
        result = sign * np.sqrt(-term1 + np.sqrt(term1 * term1 - term2))
        return result

    def _beta_sampling_v0(
        self, theta_ph: float, theta_range: float, n_samples: int
    ) -> np.ndarray:
        """Beta distribution sampling strategy.
        
        Uses the beta distribution which naturally supports asymmetric shapes
        bounded to [0, 1]. The left_weight and right_weight directly map to
        the beta distribution's alpha and beta parameters.
        
        Weight interpretation (after scaling):
        - left_weight > right_weight: More samples on left (alpha > beta)
        - right_weight > left_weight: More samples on right (beta > alpha)
        - equal weights: Symmetric distribution
        
        Requires scipy.stats. Falls back to power-law if scipy unavailable.
        
        Parameters
        ----------
        theta_ph : float
            Center heading angle.
        theta_range : float
            Angular extent on each side.
        n_samples : int
            Number of samples to generate.
            
        Returns
        -------
        np.ndarray
            Beta-distributed heading angles.
        """
        try:
            from scipy.stats import beta
        except ImportError:
            # Fallback to power-law if scipy not available
            if self.verbose:
                print(
                    "Warning: scipy not available for beta sampling. "
                    "Falling back to power-law sampling."
                )
            return self._power_law_sampling(theta_ph, theta_range, n_samples)
        
        # Map weights to beta distribution parameters
        # Scale to reasonable range (0.5 to 5.0) for good distribution shapes
        min_param = 0.5
        max_param = 5.0
        
        # Normalize weights
        total_weight = self.left_weight + self.right_weight + 1e-6
        left_norm = self.left_weight / total_weight
        right_norm = self.right_weight / total_weight
        if self.verbose:
            print(f"left_norm: {left_norm}, right_norm: {right_norm}")
        
        # Map to beta parameters (higher weight = higher parameter value)
        # This creates bias towards that side
        alpha = min_param + left_norm * (max_param - min_param) * 2
        beta_param = min_param + right_norm * (max_param - min_param) * 2
        
        # Generate deterministic samples using percentiles
        percentiles = np.linspace(0.01, 0.99, n_samples)
        beta_samples = beta.ppf(percentiles, alpha, beta_param)
        
        # Map from [0, 1] to angular range
        headings = theta_ph - theta_range + beta_samples * (2 * theta_range)
        
        return headings

    def _beta_sampling_v1(
        self, theta_ph: float, theta_range: float, n_samples: int
    ) -> np.ndarray:
        """Beta distribution sampling strategy.
        
        Uses the beta distribution which naturally supports asymmetric shapes
        bounded to [0, 1]. The left_weight and right_weight directly map to
        the beta distribution's alpha and beta parameters.
        
        Weight interpretation (after scaling):
        - left_weight > right_weight: More samples on left (alpha > beta)
        - right_weight > left_weight: More samples on right (beta > alpha)
        - equal weights: Symmetric distribution
        
        Requires scipy.stats. 
        
        Parameters
        ----------
        theta_ph : float
            Center heading angle.
        theta_range : float
            Angular extent on each side.
        n_samples : int
            Number of samples to generate.
            
        Returns
        -------
        np.ndarray
            Beta-distributed heading angles.
        """
        
        # Map weights to beta distribution parameters
        # Scale to reasonable range (0 to 10) for good distribution shapes
        def weights_to_beta_params(left_weight, right_weight, pmin=0.5, pmax=5.0, k=1.0):
            # softmax with numerical stability
            a = k * left_weight
            b = k * right_weight
            m = max(a, b)
            ea = np.exp(a - m)
            eb = np.exp(b - m)
            p_left = ea / (ea + eb)     # in (0,1)
            p_right = 1.0 - p_left      # in (0,1)

            alpha = pmin + p_left  * (pmax - pmin)
            beta  = pmin + p_right * (pmax - pmin)
            return alpha, beta
        alpha_param, beta_param = weights_to_beta_params(self.left_weight, self.right_weight)

        # Exclude low-density regions: keep only u where pdf(u) >= lambda * max_pdf
        density_threshold_fraction = 0.1  # keep regions with at least 10% of peak density
        u_grid = np.linspace(0.0, 1.0, 401)
        pdf_vals = beta.pdf(u_grid, alpha_param, beta_param)
        max_pdf = np.max(pdf_vals) + 1e-12
        keep_mask = pdf_vals >= (density_threshold_fraction * max_pdf)
        if not np.any(keep_mask):
            # Fallback: use central mass interval if threshold removed everything
            p_min = 0.01
            p_max = 0.99
        else:
            u_lo = float(u_grid[np.argmax(keep_mask)])
            u_hi = float(u_grid[len(keep_mask) - np.argmax(keep_mask[::-1]) - 1])
            # Convert to CDF bounds to sample deterministically within high-density region
            p_min = float(beta.cdf(u_lo, alpha_param, beta_param))
            p_max = float(beta.cdf(u_hi, alpha_param, beta_param))
            # Guard against numerical collapse
            if p_max - p_min < 1e-6:
                p_min = max(0.0, p_min - 1e-3)
                p_max = min(1.0, p_max + 1e-3)

        percentiles = np.linspace(p_min, p_max, n_samples)
        beta_samples = beta.ppf(percentiles, alpha_param, beta_param)

        # Map from [0, 1] to [theta_ph - theta_range, theta_ph + theta_range]
        headings = theta_ph - theta_range + beta_samples * (2 * theta_range)
        if self.verbose:
            print(f"theta_range: {theta_range}")
            print(f"beta_samples: {beta_samples}")
            print(f"headings: {headings}")
            print(f"theta_ph: {theta_ph}")

        return headings

    def _beta_sampling(
        self, theta_ph: float, theta_range: float, n_samples: int
    ) -> np.ndarray:
        """Beta distribution sampling strategy.
        
        Uses the beta distribution which naturally supports asymmetric shapes
        bounded to [0, 1]. The left_weight and right_weight directly map to
        the beta distribution's alpha and beta parameters.
        
        Weight interpretation (after scaling):
        - left_weight > right_weight: More samples on left (alpha < beta)
        - right_weight > left_weight: More samples on right (alpha > beta)
        - equal weights: Symmetric distribution
        
        The parameter range [1.0, 5.0] ensures unimodal distributions suitable
        for path planning (parameters < 1 create U-shaped distributions).
        
        Requires scipy.stats. 
        
        Parameters
        ----------
        theta_ph : float
            Center heading angle.
        theta_range : float
            Angular extent on each side.
        n_samples : int
            Number of samples to generate.
            
        Returns
        -------
        np.ndarray
            Beta-distributed heading angles.
        """
        # Map left/right weights to Beta shape parameters via clipping and linear interpolation
        # Clip weights to [-10, 10]
        lw = float(np.clip(self.left_weight, -10.0, 10.0))
        rw = float(np.clip(self.right_weight, -10.0, 10.0))
        # Linearly interpolate to [1.0, 5.0] to ensure unimodal distributions
        def to_beta_param(x: float) -> float:
            t = (x + 10.0) / 20.0  # map [-10,10] -> [0,1]
            return 1.0 + t * (5.0 - 1.0)  # map [0,1] -> [1.0, 5.0]
        # In Beta(alpha, beta): alpha > beta skews right (towards 1), alpha < beta skews left (towards 0)
        # Since u=0 maps to left and u=1 maps to right, we need right_weight → alpha
        alpha_param = to_beta_param(rw)  # right_weight controls right side (towards u=1)
        beta_param = to_beta_param(lw)   # left_weight controls left side (towards u=0)

        # Threshold-based selection: keep u where pdf >= 0.5 * max(pdf) over u in [0.01, 0.99]
        n = max(1, int(n_samples))
        u = np.linspace(0.0, 1.0, 2001)
        # FIX: Use transformed parameters, not raw weights
        pdf_vals = beta.pdf(u, alpha_param, beta_param)
        inner_mask = (u >= 0.01) & (u <= 0.99)
        max_pdf = float(np.max(pdf_vals[inner_mask])) if np.any(inner_mask) else float(np.max(pdf_vals))
        threshold = 0.5 * max_pdf
        keep_mask = pdf_vals >= threshold

        if not np.any(keep_mask):
            # Fallback: uniform samples over [0,1]
            u_samples = np.linspace(0.0, 1.0, n)
        else:
            u_kept = u[keep_mask]
            if len(u_kept) >= n:
                idxs = np.linspace(0, len(u_kept) - 1, n)
                u_samples = u_kept[np.round(idxs).astype(int)]
            else:
                u_samples = np.linspace(float(u_kept[0]), float(u_kept[-1]), n)

        # Map from [0, 1] to [theta_ph - theta_range, theta_ph + theta_range]
        headings = theta_ph - theta_range + u_samples * (2 * theta_range)
        return headings

    def _extract_heading(self, global_path: List[np.ndarray]) -> float:
        """Extract path heading by finding closest point and looking ahead."""
        if len(global_path) < 2:
            return 0.0
        
        # 1. Find the closest point on the path to robot position
        min_dist = float('inf')
        closest_idx = 0
        
        for i, path_point in enumerate(global_path):
            dist = np.linalg.norm(path_point - self.position)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # 2. Look ahead by look_ahead_idx steps from the closest point
        look_ahead_idx = min(closest_idx + self.look_ahead_idx, len(global_path) - 1)
        look_pt = global_path[look_ahead_idx]
        
        # 3. Calculate heading from robot to look-ahead point
        rel = look_pt - self.position  # vector from robot to look-ahead point in world coordinates
        absulte_angle = math.atan2(rel[1], rel[0])  # angle from robot to look-ahead point in world coordinates
        heading = absulte_angle - getattr(self, "orientation", 0.0)  # angle from robot to look-ahead point in robot coordinates
        
        return self._normalize_angle(heading)

    def _extract_curvature(self, global_path: List[np.ndarray]) -> float:
        """Extract path curvature using look-ahead mechanism (consistent with heading extraction)."""
        if len(global_path) < 3:  # need at least 3 points for curvature calculation
            return 0.0
        
        # 1. Find the closest point on the path to robot position (same as heading extraction)
        min_dist = float('inf')
        closest_idx = 0
        
        for i, path_point in enumerate(global_path):
            dist = np.linalg.norm(path_point - self.position)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # 2. Use look-ahead mechanism to get three points for curvature calculation
        # Point 1: look-ahead point (same as heading extraction)
        look_ahead_idx = min(closest_idx + self.look_ahead_idx, len(global_path) - 1)
        
        # Point 2: further ahead for curvature calculation
        point2_idx = min(look_ahead_idx + self.n_skip, len(global_path) - 1)
        
        # Point 3: even further ahead
        point3_idx = min(point2_idx + self.n_skip, len(global_path) - 1)
        
        # Ensure we have at least 3 distinct points
        if point3_idx <= look_ahead_idx:
            return 0.0
        
        # 3. Calculate Menger curvature using the three look-ahead points
        p0 = global_path[look_ahead_idx]  # First look-ahead point
        p1 = global_path[point2_idx]      # Second point
        p2 = global_path[point3_idx]      # Third point
        
        # Shift p0 to origin
        p1_shifted = p1.copy() - p0
        p2_shifted = p2.copy() - p0
        
        # Calculate area of triangle formed by the three points
        area = abs(p1_shifted[0]*p2_shifted[1] - p1_shifted[1]*p2_shifted[0]) * 0.5
        
        # Calculate Menger curvature: κ = 4A / (a*b*c) where A is area, a,b,c are side lengths
        denom = np.linalg.norm(p1_shifted) * np.linalg.norm(p2_shifted) * np.linalg.norm(p1_shifted - p2_shifted) + 1e-6
        return 4 * area / denom

    def _generate_ts_samples(
        self, dw: Tuple[float, float, float, float], theta_ph: float, kappa: float
    ) -> List[Tuple[float, float]]:
        """Polar velocity generator with path‑aware bias."""
        v_min, v_max, w_min, w_max = dw

        # Translational sampling in polar space ---------------------------
        # Use weighted sampling strategy for heading angles
        headings = self._generate_weighted_headings(
            theta_ph, self.theta_range, self.n_heading
        )
        
        # Debug: Print heading distribution
        if self.verbose:
            print(f"  Headings (relative to θ_ph): min={min(headings - theta_ph):6.3f}, "
                  f"max={max(headings - theta_ph):6.3f}, mean={np.mean(headings - theta_ph):6.3f}")
        
        speeds = np.linspace(self.max_speed * 0.1, self.max_speed, self.n_speed)

        samples: list[tuple[float, float]] = []
        for h in headings:
            for s in speeds:
                vx = s * math.cos(h)
                vy = s * math.sin(h)
                v_trans = math.hypot(vx, vy)
                # clamp to dw linear limits
                v_trans = np.clip(v_trans, v_min, v_max)
                # Angular velocity coupling (Eq. 7 in paper)
                omega = v_trans * (kappa + self.alpha_ph * h)
                omega = np.clip(omega, w_min, w_max)
                samples.append((v_trans, omega))

                # Add zero‑omega version (straight motion) for exit corridors
                #samples.append((v_trans, 0.0))

        # Escape manoeuvres (left/right/back) -----------------------------
        for omega_bias in (-self.max_rotation * 0.5, self.max_rotation * 0.5):
            #samples.append((self.max_speed * 0.3, omega_bias))
            pass
        #samples.append((self.max_speed * 0.2, 0.0))  # reverse‑like slow move
        return samples

    # -------  Legacy helpers copied verbatim from original planner  -----

    def _dynamic_window(self):
        vs = [0, self.max_speed, -self.max_rotation, self.max_rotation]
        vd = [
            self.v - self.max_accel * self.dt,
            self.v + self.max_accel * self.dt,
            self.w - self.max_angular_accel * self.dt,
            self.w + self.max_angular_accel * self.dt,
        ]
        return [
            max(vs[0], vd[0]),
            min(vs[1], vd[1]),
            max(vs[2], vd[2]),
            min(vs[3], vd[3]),
        ]

    def _predict_trajectory(self, v: float, w: float):
        time = 0.0
        traj = [self.position.copy()]
        pos = self.position.copy()
        theta = getattr(self, "orientation", 0.0)
        while time < self.predict_time:
            theta += w * self.dt
            pos[0] += v * math.cos(theta) * self.dt
            pos[1] += v * math.sin(theta) * self.dt
            traj.append(pos.copy())
            time += self.dt
        return np.array(traj)

    def _heading_score(self, v: float, w: float) -> float:
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
        next_theta = self._normalize_angle(next_theta)
        
        # Angular difference between next heading and goal direction
        angle_diff = abs(self._normalize_angle(desired_theta - next_theta))
        
        # Base score: 1.0 when aligned, 0.0 when perpendicular, negative when opposite
        base_score = 1.0 - (angle_diff / math.pi)
        
        # Penalize high angular velocities to prevent spinning (diminishing returns)
        # Trajectories with |w| > pi/2 get progressively penalized
        #rotation_penalty = max(0.0, (abs(w) - math.pi/2) / (math.pi/2))  # 0 to 1 for w from pi/2 to pi
        #rotation_penalty = max(0.0, (abs(w) - math.pi) / (math.pi))  # 0 to 1 for w from pi to 2pi
        rotation_penalty = 0.0
        return base_score - 0.3 * rotation_penalty

    def _goal_score(self, traj):
        final = traj[-1]
        dist_goal = np.linalg.norm(self.goal - final)
        dist_max = np.linalg.norm(self.goal - self.position) + 1e-6
        direction_goal = (self.goal - self.position) / max(dist_max, 1e-6)
        direction_traj = (final - self.position) / max(np.linalg.norm(final - self.position), 1e-6)
        alignment = np.dot(direction_goal, direction_traj)
        return 0.7 * (1 - dist_goal / dist_max) + 0.3 * alignment

    def _clearance_score(self, traj, people):

        # Update wall check points based on trajectory length and corridor width
        corridor_width = self.corridor_bounds['y_max'] - self.corridor_bounds['y_min']
        self.wall_check_points = max(1, int(len(traj) / (0.5 * corridor_width)))

        min_dist = float("inf")
        # Only consider people within a limited radius of the robot
        max_person_distance = 4.0  # meters; tune as needed
        for person in people:
            if not person.active:
                continue
            # Skip people that are far away from the robot
            if np.linalg.norm(person.position - self.position) > max_person_distance:
                continue
            for p in traj:
                d = np.linalg.norm(p - person.position) - self.radius - person.radius
                min_dist = min(min_dist, d)
                if min_dist <= 0:
                    return -float("inf")

        # corridor collisions - check the closest points to the corridor boundaries
        bounds = self.corridor_bounds
        for p in traj[:self.wall_check_points]:  # Based on the trajectory length and corridor width (self.wall_check_points is a dynamic parameter)
            dists = [
                p[0] - bounds["x_min"] - self.radius,
                bounds["x_max"] - p[0] - self.radius,
                p[1] - bounds["y_min"] - self.radius,
                bounds["y_max"] - p[1] - self.radius,
            ]
            min_dist = min(min_dist, *dists)
            if min_dist <= 0:
                return -float("inf")
        return min(min_dist / 1.0, 1.0)

    @staticmethod
    def _normalize_angle(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
