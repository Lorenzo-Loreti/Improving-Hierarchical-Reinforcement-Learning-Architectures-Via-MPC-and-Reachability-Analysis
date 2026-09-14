import numpy as np
from gymnasium import spaces

from .config import SlalomEnvConfig
from .width_profile import constant_profile


class SlalomVecEnv:
    """
    Batched SlalomEnv exploiting the shared LTI dynamics: every one of the
    `num_envs` copies uses the same (A, B) matrices, so a step is one batched
    matmul instead of `num_envs` independent env.step() calls in a loop.
    Used for rollout collection; SlalomEnv remains the evaluation path.

    Auto-reset convention (classic VecEnv style, as used by CleanRL/SB3): a
    sub-environment that terminates or truncates during step() is reset in
    that same call, so the returned `obs[i]` for a done env is already the
    next episode's initial observation. The pre-reset terminal transition is
    preserved in `info["final_observation"][i]` / `info["final_info"][i]` for
    correct value bootstrapping.
    """

    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, num_envs: int, config: SlalomEnvConfig = None):
        if num_envs <= 0:
            raise ValueError(f"num_envs must be > 0, got {num_envs}")

        self.num_envs = num_envs
        self.config = config or SlalomEnvConfig()
        c = self.config

        self.dt = c.dt
        self.L = c.tunnel_length
        self.W = c.tunnel_width
        self.v_max = c.v_max
        self.u_max = c.u_max
        self.sigma_p = c.sigma_p
        self.sigma_v = c.sigma_v
        self.max_steps = c.max_steps

        # Derived fresh from `self.W` (see SlalomEnvConfig.width_profile).
        self.width_profile = (
            c.width_profile if c.width_profile is not None
            else constant_profile(self.W)
        )

        y_lo, y_hi = self.width_profile.envelope()
        self.x_min = np.array([-1.0, y_lo, -self.v_max, -self.v_max], dtype=np.float32)
        self.x_max = np.array([self.L + 1.0, y_hi, self.v_max, self.v_max], dtype=np.float32)

        self.single_observation_space = spaces.Box(low=self.x_min, high=self.x_max, dtype=np.float32)
        self.single_action_space = spaces.Box(low=-self.u_max, high=self.u_max, shape=(2,), dtype=np.float32)

        self.A = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

        self.B = np.array([
            [0.5 * self.dt**2, 0.0],
            [0.0, 0.5 * self.dt**2],
            [self.dt, 0.0],
            [0.0, self.dt],
        ], dtype=np.float32)

        self.std_dev = np.array([self.sigma_p, self.sigma_p, self.sigma_v, self.sigma_v], dtype=np.float32)

        self._np_random = np.random.default_rng()
        self.states = np.zeros((num_envs, 4), dtype=np.float32)
        self.steps = np.zeros(num_envs, dtype=np.int64)

    def seed(self, seed=None):
        self._np_random = np.random.default_rng(seed)

    def _sample_initial(self, mask: np.ndarray):
        n = int(mask.sum())
        if n == 0:
            return
        p_x = self._np_random.uniform(0.0, 2.0, size=n).astype(np.float32)
        p_y = self._np_random.uniform(-self.W / 4.0, self.W / 4.0, size=n).astype(np.float32)
        zeros = np.zeros(n, dtype=np.float32)
        self.states[mask] = np.stack([p_x, p_y, zeros, zeros], axis=1)
        self.steps[mask] = 0

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        self._sample_initial(np.ones(self.num_envs, dtype=bool))
        return self.states.copy(), {}

    def step(self, actions: np.ndarray):
        actions = np.clip(actions, -self.u_max, self.u_max).astype(np.float32)
        self.steps += 1

        if np.any(self.std_dev > 0):
            noise = self._np_random.normal(loc=0.0, scale=self.std_dev, size=(self.num_envs, 4)).astype(np.float32)
        else:
            noise = np.zeros((self.num_envs, 4), dtype=np.float32)

        # Captured for the progress-shaping term below, before the line
        # after next rebinds self.states.
        prev_p_x = self.states[:, 0].copy()

        next_states = self.states @ self.A.T + actions @ self.B.T + noise
        self.states = np.clip(next_states, self.x_min, self.x_max).astype(np.float32)

        p_x, p_y = self.states[:, 0], self.states[:, 1]

        y_lo, y_hi = self.width_profile.bounds_at(p_x)
        collision = (p_y <= y_lo) | (p_y >= y_hi)
        goal = (p_x >= self.L) & ~collision
        terminated = collision | goal
        truncated = (self.steps >= self.max_steps) & ~terminated

        rewards = np.full(self.num_envs, self.config.step_penalty, dtype=np.float32)
        impact_speed = np.abs(self.states[:, 3])
        time_penalty_refund = -self.config.step_penalty * (self.steps - 1)
        rewards[collision] = (
            self.config.collision_reward
            - self.config.impact_penalty_coef * impact_speed[collision]
            + time_penalty_refund[collision]
        )
        rewards[goal] = self.config.goal_reward

        # Potential-based progress shaping (Sec 8.5 of SLALOM_ENV.md). Must
        # run before `_sample_initial(done)` below mutates `self.states` in
        # place, since `p_x` is a view into it and would otherwise pick up
        # the post-reset row instead of the terminal one.
        rewards += self.config.progress_reward_coef * (p_x - prev_p_x)

        done = terminated | truncated

        final_observation = np.full_like(self.states, np.nan)
        final_info = {
            "is_success": np.zeros(self.num_envs, dtype=bool),
            "collision": collision.copy(),
        }

        obs_out = self.states.copy()
        if np.any(done):
            final_observation[done] = self.states[done]
            final_info["is_success"][done] = goal[done]
            self._sample_initial(done)
            obs_out[done] = self.states[done]

        info = {
            "final_observation": final_observation,
            "final_info": final_info,
            "distance_to_goal": np.maximum(self.L - p_x, 0.0).astype(np.float32),
        }

        return obs_out, rewards, terminated, truncated, info
