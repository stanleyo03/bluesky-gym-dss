import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy

import gymnasium as gym
from gymnasium import spaces


# =========================
# Constants
# =========================

ALT_MEAN = 1500
ALT_STD = 3000 
VZ_MEAN = 0
VZ_STD = 5

LAT_MEAN = 0
LAT_STD = 100
VY_MEAN = 0
VY_STD = 10

RWY_DIS_MEAN = 100
RWY_DIS_STD = 200

ACTION_2_MS_VERT = 12.5
ACTION_2_MS_LAT = 8.0

ALT_DIF_REWARD_SCALE = -5 / 3000
RWY_ALT_DIF_REWARD_SCALE = -50 / 3000
CRASH_PENALTY = -100

ALT_MIN = 2000
ALT_MAX = 4000
TARGET_ALT_DIF = 500

AC_SPD = 150
ACTION_FREQUENCY = 30

MAX_ALT_RENDER = 5000
MAX_LAT_OFFSET = 200
MAX_DISTANCE = 180  # km


# =========================
# Environment
# =========================

class DescentEnvXYZ(gym.Env):
    """
    3D (x, y, z) landing environment

    x : distance to runway
    y : lateral offset from runway centerline
    z : altitude

    z is visualized via brightness (lower = darker, higher = lighter)
    """

    metadata = {"render_modes": ["human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.window_width = 512
        self.window_height = 256
        self.window_size = (self.window_width, self.window_height)

        self.observation_space = spaces.Dict(
            {
                "altitude": spaces.Box(-np.inf, np.inf, dtype=np.float64),
                "vz": spaces.Box(-np.inf, np.inf, dtype=np.float64),
                "lateral_offset": spaces.Box(-np.inf, np.inf, dtype=np.float64),
                "lateral_speed": spaces.Box(-np.inf, np.inf, dtype=np.float64),
                "target_altitude": spaces.Box(-np.inf, np.inf, dtype=np.float64),
                "runway_distance": spaces.Box(-np.inf, np.inf, dtype=np.float64),
            }
        )

        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize BlueSky
        if bs.sim is None:
            bs.init(mode="sim", detached=True)

        bs.scr = ScreenDummy()
        bs.stack.stack("DT 1;FF")

        self.total_reward = 0
        self.final_altitude = 0

        self.lateral_offset = 0.0
        self.lateral_speed = 0.0

        self.window = None
        self.clock = None

    # =========================
    # Observation
    # =========================

    def _get_obs(self):
        DEFAULT_RWY_DIS = 200
        RWY_LAT = 52
        RWY_LON = 4
        NM2KM = 1.852

        self.altitude = bs.traf.alt[0]
        self.vz = bs.traf.vs[0]

        self.runway_distance = (
            DEFAULT_RWY_DIS
            - bs.tools.geo.kwikdist(
                RWY_LAT, RWY_LON, bs.traf.lat[0], bs.traf.lon[0]
            )
            * NM2KM
        )

        obs = {
            "altitude": np.array([(self.altitude - ALT_MEAN) / ALT_STD]),
            "vz": np.array([(self.vz - VZ_MEAN) / VZ_STD]),
            "lateral_offset": np.array(
                [(self.lateral_offset - LAT_MEAN) / LAT_STD]
            ),
            "lateral_speed": np.array(
                [(self.lateral_speed - VY_MEAN) / VY_STD]
            ),
            "target_altitude": np.array(
                [(self.target_alt - ALT_MEAN) / ALT_STD]
            ),
            "runway_distance": np.array(
                [(self.runway_distance - RWY_DIS_MEAN) / RWY_DIS_STD]
            ),
        }

        return obs

    # =========================
    # Reward
    # =========================

    def _get_reward(self):
        if self.runway_distance > 0 and self.altitude > 0:
            reward = abs(self.target_alt - self.altitude) * ALT_DIF_REWARD_SCALE
            self.total_reward += reward
            return reward, False

        elif self.altitude <= 0:
            reward = CRASH_PENALTY
            self.final_altitude = -100
            self.total_reward += reward
            return reward, True

        elif self.runway_distance <= 0:
            reward = self.altitude * RWY_ALT_DIF_REWARD_SCALE
            self.final_altitude = self.altitude
            self.total_reward += reward
            return reward, True

    # =========================
    # Action
    # =========================

    def _get_action(self, act):
        act = np.asarray(act).squeeze()

        vz_cmd = act[0] * ACTION_2_MS_VERT
        vy_cmd = act[1] * ACTION_2_MS_LAT

        if vz_cmd >= 0:
            bs.traf.selalt[0] = 100000
            bs.traf.selvs[0] = vz_cmd
        else:
            bs.traf.selalt[0] = 0
            bs.traf.selvs[0] = vz_cmd

        self.lateral_speed = vy_cmd

    # =========================
    # Reset
    # =========================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.total_reward = 0
        self.final_altitude = 0

        alt_init = np.random.randint(ALT_MIN, ALT_MAX)
        self.target_alt = alt_init + np.random.randint(
            -TARGET_ALT_DIF, TARGET_ALT_DIF
        )

        self.lateral_offset = np.random.uniform(-100, 100)
        self.lateral_speed = 0.0

        bs.traf.cre("KL001", actype="A320", acalt=alt_init, acspd=AC_SPD)
        bs.traf.swvnav[0] = False

        obs = self._get_obs()
        info = {"total_reward": self.total_reward}

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    # =========================
    # Step
    # =========================

    def step(self, action):
        self._get_action(action)

        for _ in range(ACTION_FREQUENCY):
            bs.sim.step()
            self.lateral_offset += self.lateral_speed

            if self.render_mode == "human":
                self._render_frame()

        obs = self._get_obs()
        reward, terminated = self._get_reward()
        info = {"total_reward": self.total_reward}

        if terminated:
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)

        return obs, reward, terminated, False, info

    # =========================
    # Rendering
    # =========================

    def _altitude_to_color(self, alt):
        alt_norm = np.clip(alt / MAX_ALT_RENDER, 0, 1)
        brightness = int(50 + 205 * alt_norm)
        return (brightness, brightness, brightness)

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135, 206, 235))

        # ground
        pygame.draw.rect(
            canvas,
            (154, 205, 50),
            pygame.Rect(0, self.window_height - 40, self.window_width, 40),
        )

        # runway
        runway_x = int(
            ((self.runway_distance + 20) / MAX_DISTANCE) * self.window_width
        )
        pygame.draw.line(
            canvas,
            (120, 120, 120),
            (runway_x, self.window_height - 40),
            (runway_x + 40, self.window_height - 40),
            width=4,
        )

        # aircraft position
        x = int((20 / MAX_DISTANCE) * self.window_width)
        y = int(
            self.window_height / 2
            - (self.lateral_offset / MAX_LAT_OFFSET)
            * (self.window_height / 2)
        )

        color = self._altitude_to_color(self.altitude)

        pygame.draw.circle(canvas, color, (x, y), radius=6)

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        bs.stack.stack("quit")
