from dataclasses import replace

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .config import TunnelEnvConfig
from .width_profile import constant_profile


class TunnelEnv(gym.Env):
    """
    Discrete-time LTI double-integrator environment: the agent navigates a
    2D tunnel from left to right.

    Reward: a dense -1 per-step time penalty, a per-step penalty on wall
    contact (non-terminal -- the episode continues through an inelastic
    bounce), a sparse terminal reward on reaching the goal, and optional
    potential-based progress shaping — all configurable via TunnelEnvConfig.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: TunnelEnvConfig = None, render_mode=None, **overrides):
        super().__init__()

        if config is None:
            config = TunnelEnvConfig()
        if overrides:
            config = replace(config, **overrides)
        self.config = config

        self.dt = config.dt
        self.L = config.tunnel_length
        self.W = config.tunnel_width
        self.v_max = config.v_max
        self.u_max = config.u_max
        self.noise_bound_p = config.noise_bound_p
        self.noise_bound_v = config.noise_bound_v
        self.max_steps = config.max_steps

        # Derived fresh from `self.W` rather than read off the config once
        # (see TunnelEnvConfig.width_profile for why).
        self.width_profile = (
            config.width_profile if config.width_profile is not None
            else constant_profile(self.W)
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # State bounds (physical): [p_x, p_y, v_x, v_y]. The y bound is the
        # widest extent the width profile ever allows, not the (possibly
        # tighter) position-dependent bound the collision check enforces.
        y_lo, y_hi = self.width_profile.envelope()
        self.x_min = np.array([-1.0, y_lo, -self.v_max, -self.v_max], dtype=np.float32)
        self.x_max = np.array([self.L + 1.0, y_hi, self.v_max, self.v_max], dtype=np.float32)

        # Observation space: [p_x, p_y, v_x, v_y]
        self.obs_min = self.x_min
        self.obs_max = self.x_max

        self.observation_space = spaces.Box(low=self.obs_min, high=self.obs_max, dtype=np.float32)

        # Action: [a_x, a_y]
        self.action_space = spaces.Box(low=-self.u_max, high=self.u_max, shape=(2,), dtype=np.float32)

        # LTI System Matrices
        # x_{k+1} = A x_k + B u_k
        self.A = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.B = np.array([
            [0.5 * self.dt**2, 0.0],
            [0.0, 0.5 * self.dt**2],
            [self.dt, 0.0],
            [0.0, self.dt]
        ], dtype=np.float32)

        # Half-widths of the disturbance box W, per state component (see
        # the config's noise_bound_p). This replaced a Gaussian's `std_dev`
        # (and a `Sigma` covariance kept for inspection) on 2026-09-28.
        self.noise_bound = np.array([self.noise_bound_p, self.noise_bound_p,
                                     self.noise_bound_v, self.noise_bound_v], dtype=np.float32)

        self.state = None
        self.steps = 0

        self._fig = None
        self._ax = None
        self._trail = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        if options is not None and "init_state" in options:
            # Forces a specific [p_x, p_y, v_x, v_y] instead of the random
            # draw below -- used to evaluate/solve from a fixed, reusable
            # grid of initial conditions (see algorithms/optimal_solver.py's
            # spawn_grid/precompute_optimal_grid) rather than a fresh random
            # one every reset.
            self.state = np.asarray(options["init_state"], dtype=np.float32).copy()
        else:
            # Randomize initial position uniformly in a small box on the left
            # p_x in [0, 2], p_y in [-W/4, W/4]
            p_x = self.np_random.uniform(low=0.0, high=2.0)
            p_y = self.np_random.uniform(low=-self.W/4.0, high=self.W/4.0)

            # Initial velocity is exactly zero
            self.state = np.array([p_x, p_y, 0.0, 0.0], dtype=np.float32)

        self._trail = [self.state[:2].copy()]

        # Per-episode wall-contact bookkeeping (see step()): a running count
        # and the *individual* (not cumulative) penalty charged by each
        # contact this episode, so downstream logging can tell "one bad
        # crash" apart from "many small grazes" instead of only seeing a
        # single collided/not-collided flag.
        self._episode_collision_count = 0
        self._episode_collision_impacts = []

        obs = self.state.copy()
        info = {
            "is_success": False,
            "collision": False,
            "collision_count": 0,
            "collision_impacts": [],
            "distance_to_goal": float(max(self.L - self.state[0], 0.0)),
        }

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        self.steps += 1

        # Captured for the progress-shaping term below, before the LTI
        # update overwrites self.state.
        p_x_prev = float(self.state[0])

        # Clip action to bounds (just in case)
        u = np.clip(action, self.action_space.low, self.action_space.high)

        # The speed limit acts on the acceleration the plant actually
        # delivers: u is cut, per axis, to what brings the velocity exactly to
        # +-v_max and no further, so a step at top speed covers v_max * dt and
        # no more. Until 2026-09-24 the limit was only the state clip below,
        # after the position had already taken u's u * dt**2 / 2 term, and
        # agents gained 10% per step by thrusting at top speed -- enough to
        # beat the min-time oracle. The full story is at the same place in
        # SlalomEnv.step (scenarios/slalom/envs/slalom_env.py).
        v = self.state[2:]
        u = np.clip(u, (-np.float32(self.v_max) - v) / np.float32(self.dt),
                    (np.float32(self.v_max) - v) / np.float32(self.dt))

        # Additive disturbance, uniform in the box W (see the config's
        # noise_bound_p). Nothing is drawn when W = {0}, so the
        # deterministic environment consumes no randomness here.
        if np.any(self.noise_bound > 0):
            w = self.np_random.uniform(low=-self.noise_bound, high=self.noise_bound)
        else:
            w = np.zeros(4, dtype=np.float32)

        # LTI step
        next_state = self.A @ self.state + self.B @ u + w

        # Clip to state bounds. For the velocity this is now only a guard
        # against noise and float32 rounding: the speed limit is applied to u
        # above.
        self.state = np.clip(next_state, self.x_min, self.x_max).astype(np.float32)

        # Extract positions
        p_x, p_y = self.state[0], self.state[1]

        terminated = False
        truncated = False
        collision = False
        is_success = False
        reward = self.config.step_penalty

        # Wall contact, looked up at the post-step, post-clip p_x. A fully
        # inelastic bounce: p_y is clamped to the violated bound and v_y is
        # zeroed (v_x is untouched -- only the wall-normal component is
        # absorbed on impact). No mirroring/reflection is needed since a
        # zero-restitution bounce can only move p_y back to the boundary,
        # never past the opposite wall. The episode does not end; the
        # contact is charged a small per-step penalty (additive to
        # step_penalty) instead of the old terminal one.
        y_lo, y_hi = self.width_profile.bounds_at(p_x)
        if p_y <= y_lo or p_y >= y_hi:
            p_y = y_lo if p_y <= y_lo else y_hi
            self.state[1] = p_y
            self.state[3] = 0.0
            collision = True

            contact_cost = self.config.contact_penalty
            reward += contact_cost
            self._episode_collision_count += 1
            self._episode_collision_impacts.append(contact_cost)

        # Check goal reached (independent of the contact check above -- a
        # step that reaches the goal is a clean terminal success regardless
        # of an incidental wall graze the same step).
        if p_x >= self.L:
            reward = self.config.goal_reward
            terminated = True
            is_success = True

        # Check max steps
        if self.steps >= self.max_steps and not terminated:
            truncated = True

        # Potential-based progress shaping (Sec 8.5), on top of whichever
        # branch fired above.
        reward += self.config.progress_reward_coef * (p_x - p_x_prev)

        self._trail.append(self.state[:2].copy())

        obs = self.state.copy()
        info = {
            "is_success": is_success,
            "collision": collision,
            "collision_count": self._episode_collision_count,
            "collision_impacts": list(self._episode_collision_impacts),
            "distance_to_goal": float(max(self.L - p_x, 0.0)),
        }

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "Calling render() without specifying render_mode returns nothing. "
                "Pass render_mode='human' or render_mode='rgb_array' to TunnelEnv()."
            )
            return
        return self._render_frame()

    def _render_frame(self):
        import matplotlib.pyplot as plt

        if self._fig is None:
            if self.render_mode == "human":
                plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(8, 4))

        ax = self._ax
        ax.clear()
        ax.set_xlim(float(self.x_min[0]), float(self.x_max[0]))
        ax.set_ylim(float(self.x_min[1]), float(self.x_max[1]))
        ax.set_aspect("equal")

        # Tunnel walls
        ax.axhline(self.W / 2.0, color="black", linewidth=2)
        ax.axhline(-self.W / 2.0, color="black", linewidth=2)
        # Goal line
        ax.axvline(self.L, color="tab:green", linestyle="--", linewidth=1.5, label="goal")

        # Trajectory trail
        if len(self._trail) > 1:
            trail = np.array(self._trail)
            ax.plot(trail[:, 0], trail[:, 1], color="tab:blue", alpha=0.5, linewidth=1)

        # Agent
        if self.state is not None:
            ax.plot(self.state[0], self.state[1], "o", color="tab:red", markersize=8, label="agent")

        ax.set_xlabel("p_x")
        ax.set_ylabel("p_y")
        ax.set_title(f"TunnelEnv  step={self.steps}")
        ax.legend(loc="upper right", fontsize=8)

        self._fig.canvas.draw()

        if self.render_mode == "human":
            plt.pause(1.0 / self.metadata["render_fps"])
            return None

        # rgb_array
        renderer = self._fig.canvas.get_renderer()
        width, height = self._fig.canvas.get_width_height()
        buf = np.frombuffer(renderer.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
        return buf[:, :, :3].copy()

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None
