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

        # Per-episode wall-contact bookkeeping, per env (see step()): a
        # running count and the *individual* (not cumulative) penalty of
        # each contact this episode.
        self._collision_counts = np.zeros(num_envs, dtype=np.int64)
        self._collision_impacts = [[] for _ in range(num_envs)]

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
        self._collision_counts[mask] = 0
        for i in np.flatnonzero(mask):
            self._collision_impacts[i] = []

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        self._sample_initial(np.ones(self.num_envs, dtype=bool))
        return self.states.copy(), {}

    def step(self, actions: np.ndarray):
        actions = np.clip(actions, -self.u_max, self.u_max).astype(np.float32)
        # The speed limit acts on the delivered acceleration, exactly as in
        # SlalomEnv.step (see the comment there): cut per axis to what brings
        # the velocity to +-v_max and no further, so a step at top speed
        # covers v_max * dt and no more.
        v = self.states[:, 2:]
        actions = np.clip(actions, (-np.float32(self.v_max) - v) / np.float32(self.dt),
                          (np.float32(self.v_max) - v) / np.float32(self.dt))
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

        # Wall contact, looked up at the post-step, post-clip p_x. A fully
        # inelastic bounce: p_y is clamped to the violated bound and v_y is
        # zeroed for every collided env (v_x untouched). No mirroring is
        # needed since a zero-restitution bounce cannot overshoot the
        # opposite wall. The episode no longer ends on contact.
        y_lo, y_hi = self.width_profile.bounds_at(p_x)
        collision = (p_y <= y_lo) | (p_y >= y_hi)
        if np.any(collision):
            clamped_y = np.where(p_y <= y_lo, y_lo, y_hi)
            self.states[collision, 1] = clamped_y[collision]
            self.states[collision, 3] = 0.0
            p_y = self.states[:, 1]

        goal = p_x >= self.L
        terminated = goal
        truncated = (self.steps >= self.max_steps) & ~terminated

        rewards = np.full(self.num_envs, self.config.step_penalty, dtype=np.float32)
        rewards[collision] += self.config.contact_penalty
        rewards[goal] = self.config.goal_reward

        # Potential-based progress shaping. Must
        # run before `_sample_initial(done)` below mutates `self.states` in
        # place, since `p_x` is a view into it and would otherwise pick up
        # the post-reset row instead of the terminal one.
        rewards += self.config.progress_reward_coef * (p_x - prev_p_x)

        # Per-episode wall-contact bookkeeping, per env (see __init__): a
        # running count and the *individual* (not cumulative) penalty of
        # each contact this episode. Updated -- and snapshotted into
        # final_info below -- before `_sample_initial(done)` resets a done
        # env's counters.
        self._collision_counts[collision] += 1
        for i in np.flatnonzero(collision):
            self._collision_impacts[i].append(float(self.config.contact_penalty))

        done = terminated | truncated

        final_observation = np.full_like(self.states, np.nan)
        final_info = {
            "is_success": np.zeros(self.num_envs, dtype=bool),
            "collision": collision.copy(),
            "collision_count": self._collision_counts.copy(),
            "collision_impacts": [list(lst) for lst in self._collision_impacts],
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
