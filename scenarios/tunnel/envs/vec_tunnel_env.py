import numpy as np
from gymnasium import spaces

from .actuation import deliver, project_disk, sample_disturbance
from .config import TunnelEnvConfig
from .spawn_sampler import SobolSpawnStream, spawn_box
from .width_profile import constant_profile


class TunnelVecEnv:
    """
    Batched TunnelEnv exploiting the shared LTI dynamics: every one of the
    `num_envs` copies uses the same (A, B) matrices, so a step is one batched
    matmul instead of `num_envs` independent env.step() calls in a loop.
    Used for rollout collection; TunnelEnv remains the evaluation path.

    Auto-reset convention (classic VecEnv style, as used by CleanRL/SB3): a
    sub-environment that terminates or truncates during step() is reset in
    that same call, so the returned `obs[i]` for a done env is already the
    next episode's initial observation. The pre-reset terminal transition is
    preserved in `info["final_observation"][i]` / `info["final_info"][i]` for
    correct value bootstrapping.
    """

    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, num_envs: int, config: TunnelEnvConfig = None):
        if num_envs <= 0:
            raise ValueError(f"num_envs must be > 0, got {num_envs}")

        self.num_envs = num_envs
        self.config = config or TunnelEnvConfig()
        c = self.config

        self.dt = c.dt
        self.L = c.tunnel_length
        self.W = c.tunnel_width
        self.v_max = c.v_max
        self.u_max = c.u_max
        self.noise_bound_p = c.noise_bound_p
        self.noise_bound_v = c.noise_bound_v
        self.max_steps = c.max_steps

        # Derived fresh from `self.W` (see TunnelEnvConfig.width_profile).
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

        # Radii of the disturbance's disks, per state component, only read to
        # tell whether W = {0} (see TunnelEnv.__init__).
        self.noise_bound = np.array([self.noise_bound_p, self.noise_bound_p,
                                     self.noise_bound_v, self.noise_bound_v], dtype=np.float32)
        # effort_penalty / u_max^2, as in TunnelEnv.
        self._effort_scale = np.float32(c.effort_penalty / c.u_max ** 2)

        # Where and how episodes start: see envs/spawn_sampler.py and the
        # config's init_sampler.
        self.init_sampler = c.init_sampler
        self.spawn_low, self.spawn_high = spawn_box(self.W)

        self.seed()
        self.states = np.zeros((num_envs, 4), dtype=np.float32)
        self.steps = np.zeros(num_envs, dtype=np.int64)

        # Per-episode wall-contact bookkeeping, per env (see step()): a
        # running count and the *individual* (not cumulative) penalty of
        # each contact this episode.
        self._collision_counts = np.zeros(num_envs, dtype=np.int64)
        self._collision_impacts = [[] for _ in range(num_envs)]

    def seed(self, seed=None):
        self._np_random = np.random.default_rng(seed)
        # One Sobol' stream for all the sub-environments, restarted with the
        # seed. It never draws from _np_random, which then holds the
        # disturbance alone; under "uniform" the starts and the disturbance
        # share it, interleaved in the order episodes end.
        self._spawn_stream = SobolSpawnStream(seed) if self.init_sampler == "sobol" else None

    def _sample_initial(self, mask: np.ndarray):
        n = int(mask.sum())
        if n == 0:
            return
        low, high = self.spawn_low, self.spawn_high
        if self.init_sampler == "sobol":
            # The stream's next n points, mapped onto the box. `self.states[mask]`
            # below fills rows in ascending env index, so environments that
            # finish on the same step take the points in that order.
            u = self._spawn_stream.take(n)
            p_x = (low[0] + u[:, 0] * (high[0] - low[0])).astype(np.float32)
            p_y = (low[1] + u[:, 1] * (high[1] - low[1])).astype(np.float32)
        else:
            p_x = self._np_random.uniform(low[0], high[0], size=n).astype(np.float32)
            p_y = self._np_random.uniform(low[1], high[1], size=n).astype(np.float32)
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
        # The delivered acceleration, exactly as in TunnelEnv.step (see the
        # comment there and envs/actuation.py): saturated radially to
        # ||u|| <= u_max, then cut to what keeps ||v'|| <= v_max.
        actions = deliver(actions, self.states[:, 2:], self.u_max, self.v_max, self.dt)
        self.steps += 1

        # Uniform on the disks of W, drawn only when W != {0}, as in the
        # scalar env's step().
        if np.any(self.noise_bound > 0):
            noise = sample_disturbance(self._np_random, self.noise_bound_p, self.noise_bound_v, self.num_envs)
        else:
            noise = np.zeros((self.num_envs, 4), dtype=np.float32)

        # Captured for the progress-shaping term below, before the line
        # after next rebinds self.states.
        prev_p_x = self.states[:, 0].copy()

        next_states = self.states @ self.A.T + actions @ self.B.T + noise
        # Position in its box, velocity in the speed disk (a guard, as in
        # TunnelEnv.step).
        next_states[:, :2] = np.clip(next_states[:, :2], self.x_min[:2], self.x_max[:2])
        next_states[:, 2:] = project_disk(next_states[:, 2:], self.v_max)
        self.states = next_states.astype(np.float32)

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

        # Potential-based progress shaping, cut at L as in TunnelEnv.step.
        # Must run before `_sample_initial(done)` below mutates `self.states`
        # in place, since `p_x` is a view into it and would otherwise pick up
        # the post-reset row instead of the terminal one.
        rewards += self.config.progress_reward_coef * (np.minimum(p_x, np.float32(self.L)) - prev_p_x)

        # Control effort on the delivered u, as in TunnelEnv.step.
        effort = self._effort_scale * (actions[:, 0] * actions[:, 0] + actions[:, 1] * actions[:, 1])
        rewards += effort

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
            # This step's effort term (already in `rewards`), per env: hPPO
            # weighs it into its worker's reward on its own.
            "effort_penalty": effort,
        }

        return obs_out, rewards, terminated, truncated, info
