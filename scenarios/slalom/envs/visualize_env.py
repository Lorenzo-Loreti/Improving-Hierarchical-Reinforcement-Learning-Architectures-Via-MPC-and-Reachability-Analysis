"""Static visualization of SlalomEnv's track geometry.

Draws the corridor, gates, spawn box, and goal line implied by a
SlalomEnvConfig -- built purely from `config` + `width_profile`, no
`gym.Env` instantiation needed. Unlike `SlalomEnv.render()`, which only
ever draws flat walls at +-tunnel_width/2, this traces the actual
per-segment `WidthProfile` bounds, so the two gates are visible.

Usage (from scenarios/slalom/, so `envs` resolves as a package):
    python -m envs.visualize_env
    python -m envs.visualize_env --tunnel-length 12 --output envs/env_layout.png
"""

import argparse
from dataclasses import replace

import matplotlib.pyplot as plt

from .config import SlalomEnvConfig
from .spawn_sampler import spawn_box
from .width_profile import slalom_profile


def resolve_profile(config: SlalomEnvConfig):
    """Mirrors SlalomEnv's own fallback: an explicit `config.width_profile`
    wins, otherwise fall back to `slalom_profile` (the profile every real
    training run in this project actually uses -- see scripts/script_ppo.py)."""
    if config.width_profile is not None:
        return config.width_profile
    return slalom_profile(
        tunnel_length=config.tunnel_length,
        tunnel_width=config.tunnel_width,
    )


def plot_env(config: SlalomEnvConfig, ax=None, overlay=False):
    """Draw SlalomEnv's track layout onto `ax` (a new figure if omitted).

    `overlay=True` draws the same geometry in neutral greys and leaves out the
    title and the parameter footer: the style for figures that draw
    trajectories on top of the track (algorithms/study.py), where color has to
    belong to the trajectories alone -- orange gates and a blue spawn box
    would read as two more series. The footer is also a figure-level text, so
    it cannot be drawn once per panel of a multi-panel figure.
    """
    profile = resolve_profile(config)
    wall_ink, wall_width = ("0.3", 1.2) if overlay else ("black", 2)
    gate_color = "0.72" if overlay else "tab:orange"
    spawn_color = "0.45" if overlay else "tab:blue"
    goal_color = "0.4" if overlay else "tab:green"
    # In an overlay, dashes are left to the trajectories (the study figures
    # draw the oracle dashed), so the spawn box is dotted and the goal solid.
    spawn_ls, spawn_lw = (":", 1.1) if overlay else ("--", 1.3)
    goal_ls, goal_lw = ("-", 1.3) if overlay else ("--", 1.8)
    label_box = dict(fc="white", ec="none", pad=0.8) if overlay else None

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))

    x_lo, x_hi = -1.0, config.tunnel_length + 1.0
    y_lo_env, y_hi_env = profile.envelope()
    pad = 0.4
    full_half = config.tunnel_width / 2.0

    gate_num = 0
    for seg in profile.segments:
        x_start = max(seg.x_start, x_lo)
        x_end = min(seg.x_end, x_hi)
        if x_start >= x_end:
            continue

        y_lo, y_hi = seg.center_y - seg.half_width, seg.center_y + seg.half_width
        is_gate = seg.half_width < full_half - 1e-9
        color = gate_color if is_gate else "0.85"

        # Shaded wall regions (above y_hi, below y_lo), so narrow segments
        # visibly cut further into the corridor than full-width ones.
        ax.fill_between([x_start, x_end], y_hi, y_hi_env + pad, color=color, alpha=0.7, linewidth=0)
        ax.fill_between([x_start, x_end], y_lo_env - pad, y_lo, color=color, alpha=0.7, linewidth=0)
        ax.hlines([y_lo, y_hi], x_start, x_end, color=wall_ink, linewidth=wall_width)
        ax.axvline(seg.x_start, color="0.5", linewidth=0.5, linestyle=":")

        if is_gate:
            gate_num += 1
            ax.text((x_start + x_end) / 2.0, y_hi_env + pad * 0.6, f"gate {gate_num}",
                    ha="center", va="bottom", fontsize=8,
                    color="0.35" if overlay else "tab:orange")

    # Spawn box: p_x in [0, 2], p_y in [-W/4, W/4], the box SlalomEnv.reset()
    # draws from (envs/spawn_sampler.py).
    spawn_low, spawn_high = spawn_box(config.tunnel_width)
    ax.add_patch(plt.Rectangle(
        tuple(spawn_low), *(spawn_high - spawn_low),
        fill=False, edgecolor=spawn_color, linestyle=spawn_ls, linewidth=spawn_lw,
    ))
    ax.text((spawn_low[0] + spawn_high[0]) / 2.0, spawn_low[1] - pad * 0.4, "spawn box",
            ha="center", va="top", fontsize=8, color=spawn_color)

    # Goal line.
    ax.axvline(config.tunnel_length, color=goal_color, linestyle=goal_ls, linewidth=goal_lw)
    ax.text(config.tunnel_length, y_hi_env + pad * 0.6, "goal",
            ha="center", va="bottom", fontsize=8, color=goal_color, bbox=label_box)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo_env - pad, y_hi_env + pad * 1.6)
    ax.set_aspect("equal")
    ax.set_xlabel("p_x")
    ax.set_ylabel("p_y")
    if overlay:
        return ax
    ax.set_title(f"SlalomEnv track layout  (num_gates={gate_num})")

    param_text = (
        f"dt={config.dt}   v_max={config.v_max}   u_max={config.u_max}   "
        f"noise_bound_p={config.noise_bound_p}   noise_bound_v={config.noise_bound_v}\n"
        f"max_steps={config.max_steps}   step_penalty={config.step_penalty}   "
        f"goal_reward={config.goal_reward}   contact_penalty={config.contact_penalty}   "
        f"progress_reward_coef={config.progress_reward_coef}"
    )
    ax.figure.subplots_adjust(bottom=0.26)
    ax.figure.text(0.5, 0.02, param_text, ha="center", va="bottom", fontsize=7.5, family="monospace")

    return ax


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SlalomEnv's track geometry.")
    parser.add_argument("--tunnel-length", type=float, default=None)
    parser.add_argument("--tunnel-width", type=float, default=None)
    parser.add_argument("--output", type=str, default=None, help="save the figure to this path")
    parser.add_argument("--no-show", action="store_true", help="don't open an interactive window")
    return parser.parse_args()


def main():
    args = parse_args()

    overrides = {}
    if args.tunnel_length is not None:
        overrides["tunnel_length"] = args.tunnel_length
    if args.tunnel_width is not None:
        overrides["tunnel_width"] = args.tunnel_width

    config = SlalomEnvConfig(**overrides)
    config = replace(config, width_profile=slalom_profile(
        tunnel_length=config.tunnel_length,
        tunnel_width=config.tunnel_width,
    ))

    plot_env(config)

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
