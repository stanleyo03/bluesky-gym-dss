import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn


import gymnasium as gym
from gymnasium import spaces


# =========================
# Constants
# =========================

# Normalization constants
ALT_MEAN = 1500
ALT_STD = 3000

# Aircraft parameters
AC_SPD = 150
ACTION_FREQUENCY = 20

# Action scaling
ACTION_2_MS = 12.5  # Vertical speed scaling (m/s)
D_HEADING = 45      # Heading change scaling (degrees)

# Reward parameters
ALT_DIF_REWARD_SCALE = -5 / 3000
REACH_REWARD = 100
CRASH_PENALTY = -100
RWY_ALT_DIF_REWARD_SCALE = -5/300

# Altitude initialization
ALT_MIN = 2000
ALT_MAX = 4000
TARGET_ALT_DIF = 500

VZ_MEAN = 0
VZ_STD = 5

# Rendering
MAX_ALT_RENDER = 5000

# Waypoint parameters
NUM_WAYPOINTS = 1
DISTANCE_MARGIN = 5         # km - landing zone radius
WAYPOINT_DISTANCE_MIN = 75  # km - minimum distance to waypoint
WAYPOINT_DISTANCE_MAX = 300  # km - maximum distance to waypoint

# =========================
# Environment
# =========================

class DescentEnvXYZ(gym.Env):
    """
    3D (x, y, z) landing environment

    x : horizontal movement (distance along x-axis)
    y : vertical movement across the screen (distance along y-axis)
    z : altitude

    z is visualized via transparency (lower = darker, higher = lighter)
    """

    metadata = {"render_modes": ["human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height)

        self.observation_space = spaces.Dict(
            {
                "waypoint_distance": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
                "cos_difference": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
                "sin_difference": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
                "waypoint_reached": spaces.Box(0, 1, shape=(NUM_WAYPOINTS,), dtype=np.float64),
                "altitude": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "target_altitude": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "vz": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64)
            }
        )
        # holding horizontal velocity constant at 150
        # vertical velocity, heading 
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize BlueSky
        if bs.sim is None:
            bs.init(mode="sim", detached=True)

        bs.scr = ScreenDummy()
        bs.stack.stack("DT 1;FF")

        # Logging variables
        self.total_reward = 0
        self.final_altitude = 0

        # Episode state
        self.reached = False
        self.landed = False
        
        # Initialize observation variables (will be set in reset)
        self.altitude = 0.0
        self.vz = 0.0
        self.ac_hdg = 0.0
        self.wpt_dis = 0.0
        self.wpt_qdr = []
        self.wpt_reach = [0]
        self.wpt_cos = 0.0
        self.wpt_sin = 0.0
        self.drift = 0.0
        self.target_alt = 0.0
        self.wpt_lat = 0.0
        self.wpt_lon = 0.0

        # previous heading for reward fn
        self.prev_hdg = None

        # For rendering
        self.window = None
        self.clock = None
        self.font = None
        self.plane_img = None

    # =========================
    # Observation
    # =========================

    def _get_obs(self):
        # Horizontal observations
        NM2KM = 1.852
        ac_idx = bs.traf.id2idx('KL001')

        # Current heading of aircraft
        self.ac_hdg = bs.traf.hdg[ac_idx]
        
        # Get absolute angles between aircraft and waypoint location
        wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(
            bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], 
            self.wpt_lat, self.wpt_lon
        )

        self.wpt_dis = wpt_dis * NM2KM
        self.wpt_qdr = [wpt_qdr]  # Store as list for consistency

        drift = self.ac_hdg - wpt_qdr
        drift = fn.bound_angle_positive_negative_180(drift)

        self.wpt_cos = np.cos(np.deg2rad(drift))
        self.wpt_sin = np.sin(np.deg2rad(drift))
        self.drift = drift

        # Vertical Observations - Get altitude from BlueSky
        self.vz = bs.traf.vs[0]
        self.altitude = bs.traf.alt[0]

        # Normalize observations
        obs_altitude = np.array([(self.altitude - ALT_MEAN) / ALT_STD])
        obs_target_alt = np.array([(self.target_alt - ALT_MEAN) / ALT_STD])

        # Convert to arrays with shape (NUM_WAYPOINTS,) = (1,)
        wpt_reach_arr = np.array(self.wpt_reach, dtype=np.float64)
        
        # Mask observations if waypoint is reached (multiply by (1 - reached))
        mask = (wpt_reach_arr - 1) * -1  # 1 if not reached, 0 if reached

        obs = {
            "waypoint_distance": mask * np.array([self.wpt_dis]) / WAYPOINT_DISTANCE_MAX,
            "cos_difference": mask * np.array([self.wpt_cos]),
            "sin_difference": mask * np.array([self.wpt_sin]),
            "waypoint_reached": wpt_reach_arr,
            "altitude": obs_altitude,
            "target_altitude": obs_target_alt,
            "vz": np.array([(self.vz - VZ_MEAN) / VZ_STD]),
        }

        return obs

    def _get_info(self):
        # Here you implement any additional info that you want to return after a step,
        # but that should not be used by the agent for decision making, so used for logging and debugging purposes
        # for now just have 10, because it crashed if I gave none for some reason.
        return {
            "total_reward": self.total_reward,
            "final_altitude": self.final_altitude
        }

    # =========================
    # Reward
    # =========================

    def _get_reward(self):
        """
        Reward function for 3D landing task.

        Objectives:
        1. Track target altitude when far from landing zone
        2. Gradually transition to descending as landing zone approaches
        3. Reach landing zone at low altitude (~0)
        4. Avoid crashes and oscillatory control
        """

        # Distance to landing zone (km)
        d = self.wpt_dis

        # Crash condition
        if self.altitude <= 0:
            reward = CRASH_PENALTY
            self.final_altitude = self.altitude
            self.total_reward += reward
            return reward, True

        # Distance-based altitude target
        # alpha = 1 far away, 0 near landing zone
        alpha = np.clip(d / WAYPOINT_DISTANCE_MAX, 0.0, 1.0)

        # Desired altitude transitions smoothly:
        # far -> target_alt
        # near -> 0
        desired_alt = alpha * self.target_alt

        # Altitude tracking penalty
        alt_error = alpha * abs(self.altitude - self.target_alt)
        altitude_penalty = ALT_DIF_REWARD_SCALE * alt_error


        # Encourage getting closer to the landing zone
        distance_penalty = -0.02 * d


        # Penalize aggressive vertical motion
        vz_penalty = -0.01 * abs(self.vz)

        hdg_penalty = 0
        # Penalize too many turns, 
        if self.prev_hdg:
            hdg_penalty = (self.ac_hdg - self.prev_hdg) * -0.001

        # Waypoint reached logic
        if d <= DISTANCE_MARGIN and not self.reached:
            self.reached = True
            self.wpt_reach = [1]

        # Successful landing condition
        if self.reached and self.altitude <= 100:
            reward = REACH_REWARD
            self.final_altitude = self.altitude
            self.total_reward += reward
            self.landed = True
            return reward, True

        # Bad terminal: reached horizontally but too high
        if self.reached and self.altitude > 100:
            reward = RWY_ALT_DIF_REWARD_SCALE * self.altitude
            self.final_altitude = self.altitude
            self.total_reward += reward
            return reward, True

        # Total reward
        reward = altitude_penalty + distance_penalty + vz_penalty + hdg_penalty
        self.total_reward += reward

        return reward, False


    # =========================
    # Action
    # =========================

    def _get_action(self, act):
        # Action[0]: Vertical speed (climb/descend)
        alt_action = act[0] * ACTION_2_MS

        # BlueSky interprets vertical velocity command through altitude commands 
        # with a vertical speed (magnitude). Check sign of action and give appropriate altitude target
        if alt_action >= 0:
            bs.traf.selalt[0] = 1000000  # High target altitude to start climb
            bs.traf.selvs[0] = alt_action
        else:
            bs.traf.selalt[0] = 0  # Low target altitude to start descent
            bs.traf.selvs[0] = alt_action
        
        self.prev_hdg = self.ac_hdg

        # Action[1]: Change horizontal heading
        hdg_action = (self.ac_hdg + act[1] * D_HEADING) % 360
        bs.stack.stack(f"HDG KL001 {hdg_action}")

    # =========================
    # Reset
    # =========================

    def _generate_waypoint(self, acid = "KL001"):
        wpt_dis_init = np.random.randint(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
        wpt_hdg_init = np.random.randint(0, 359)

        ac_idx = bs.traf.id2idx(acid)

        self.wpt_lat, self.wpt_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_dis_init, wpt_hdg_init)    
        self.wpt_reach = [0]  # List for consistency with observation space
        self.wpt_qdr = []  # Initialize waypoint bearing list

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset logging variables
        self.total_reward = 0
        self.final_altitude = 0
        self.reached = False
        self.landed = False

        # Initialize altitude and target altitude
        alt_init = np.random.randint(ALT_MIN, ALT_MAX)
        self.target_alt = alt_init + np.random.randint(-TARGET_ALT_DIF, TARGET_ALT_DIF)

        # Create aircraft in BlueSky (default position)
        bs.traf.cre("KL001", actype="A320", acalt=alt_init, acspd=AC_SPD)
        bs.traf.swvnav[0] = False

        # Generate landing zone waypoint (after aircraft is created)
        self._generate_waypoint()

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    # =========================
    # Step
    # =========================

    def step(self, action):
        self._get_action(action)

        # Run simulation for ACTION_FREQUENCY steps
        for _ in range(ACTION_FREQUENCY):
            bs.sim.step()
            
            if self.render_mode == "human":
                self._get_obs()  # Update observations for rendering
                self._render_frame()

        obs = self._get_obs()
        reward, terminated = self._get_reward()
        info = self._get_info()

        # Delete aircraft if episode terminated
        if terminated:
            for acid in bs.traf.id:
                try:
                    idx = bs.traf.id2idx(acid)
                    bs.traf.delete(idx)
                except (ValueError, AttributeError):
                    pass

        return obs, reward, terminated, False, info

    # =========================
    # Rendering
    # =========================

    def _altitude_to_red_color(self, alt):
        """Convert altitude to red color value.
        Lower altitude = dark red, higher altitude = light red
        """
        alt_norm = np.clip(alt / MAX_ALT_RENDER, 0, 1)
        # Lower altitude = dark red (low value, e.g., 50)
        # Higher altitude = light red (high value, e.g., 255)
        red_value = int(50 + 205 * alt_norm)
        return red_value

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        max_distance = 200  # Width of screen in km

        canvas = pygame.Surface(self.window_size)
        # Fill entire background with blue
        canvas.fill((135, 206, 235))

        # Get aircraft index and heading
        ac_idx = bs.traf.id2idx('KL001')
        
        # Get red color value based on altitude (dark red at 0, lighter as altitude increases)
        red_value = self._altitude_to_red_color(self.altitude)
        aircraft_color = (red_value, 0, 0)  # Red color, brightness based on altitude

        # Draw aircraft (ownship) at center of screen
        ac_length = 8
        heading_end_x = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length) / max_distance) * self.window_width
        heading_end_y = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length) / max_distance) * self.window_width

        pygame.draw.line(
            canvas,
            aircraft_color,
            (self.window_width / 2, self.window_height / 2),
            ((self.window_width / 2) + heading_end_x, (self.window_height / 2) - heading_end_y),
            width=4
        )

        # Draw heading line (longer line showing heading direction)
        heading_length = 50
        heading_end_x = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length) / max_distance) * self.window_width
        heading_end_y = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length) / max_distance) * self.window_width

        pygame.draw.line(
            canvas,
            (0, 0, 0),  # Black heading line
            (self.window_width / 2, self.window_height / 2),
            ((self.window_width / 2) + heading_end_x, (self.window_height / 2) - heading_end_y),
            width=1
        )

        # Draw aircraft (sprite)
        # # Lazy-load plane sprite
        # if self.plane_img is None:
        #     self.plane_img = pygame.image.load("static\plane.png").convert_alpha()
        #     self.plane_img = pygame.transform.smoothscale(self.plane_img, (32, 32))

        # # Rotate sprite to match heading
        # # Pygame rotates CCW; heading is CW → negate
        # hdg = bs.traf.hdg[ac_idx]
        # rotated_plane = pygame.transform.rotate(self.plane_img, -hdg)

        # # Altitude-based red tint
        # red_value = self._altitude_to_red_color(self.altitude)
        # tint = pygame.Surface(rotated_plane.get_size(), pygame.SRCALPHA)
        # tint.fill((red_value, 0, 0, 80))  # last value = transparency
        # rotated_plane.blit(tint, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

        # # Draw centered at aircraft position
        # plane_rect = rotated_plane.get_rect(
        #     center=(self.window_width // 2, self.window_height // 2)
        # )
        # canvas.blit(rotated_plane, plane_rect)


        # Draw landing zone waypoint (relative to aircraft position)
        if len(self.wpt_qdr) > 0:
            qdr = self.wpt_qdr[0]
            dis = self.wpt_dis

            circle_x = ((np.cos(np.deg2rad(qdr)) * dis) / max_distance) * self.window_width
            circle_y = ((np.sin(np.deg2rad(qdr)) * dis) / max_distance) * self.window_width

            # Color based on whether waypoint is reached
            if self.reached:
                color = (155, 155, 155)  # Gray if reached
            else:
                color = (0, 0, 0)  # Black if not reached

            # Inner filled circle
            pygame.draw.circle(
                canvas,
                color,
                ((self.window_width / 2) + circle_x, (self.window_height / 2) - circle_y),
                radius=4,
                width=0
            )

            # Outer outline circle (radius based on distance margin)
            pygame.draw.circle(
                canvas,
                color,
                ((self.window_width / 2) + circle_x, (self.window_height / 2) - circle_y),
                radius=int((DISTANCE_MARGIN / max_distance) * self.window_width),
                width=2
            )

        # Draw altitude text on screen
        if self.font is None:
            self.font = pygame.font.Font(None, 36)
        
        altitude_text = f"Target Alt: {self.target_alt:.0f} m, Altitude: {self.altitude:.0f} m"
        text_surface = self.font.render(altitude_text, True, (0, 0, 0))  # Black text
        canvas.blit(text_surface, (10, 10))  # Position at top-left corner

        # Draw distance to waypoint
        distance_text = f"Distance: {self.wpt_dis:.1f} km"
        dist_surface = self.font.render(distance_text, True, (0, 0, 0))
        canvas.blit(dist_surface, (10, 40))

        # Draw lat/longs
        lat_long_text = f"Latitude: {bs.traf.lat[ac_idx]:.2f}{chr(176)}, Longitude: {bs.traf.lon[ac_idx]:.2f}{chr(176)}"
        lat_long_surface = self.font.render(lat_long_text, True, (0, 0, 0))
        canvas.blit(lat_long_surface, (10, 70))
        
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        bs.stack.stack("quit")
