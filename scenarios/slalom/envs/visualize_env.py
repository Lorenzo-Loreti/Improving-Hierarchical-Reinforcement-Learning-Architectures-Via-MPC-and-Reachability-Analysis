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


def plot_env(config: SlalomEnvConfig, ax=None):
    """Draw SlalomEnv's track layout onto `ax` (a new figure if omitted)."""
    profile = resolve_profile(config)

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
        color = "tab:orange" if is_gate else "0.85"

        # Shaded wall regions (above y_hi, below y_lo), so narrow segments
        # visibly cut further into the corridor than full-width ones.
        ax.fill_between([x_start, x_end], y_hi, y_hi_env + pad, color=color, alpha=0.7, linewidth=0)
        ax.fill_between([x_start, x_end], y_lo_env - pad, y_lo, color=color, alpha=0.7, linewidth=0)
        ax.hlines([y_lo, y_hi], x_start, x_end, color="black", linewidth=2)
        ax.axvline(seg.x_start, color="0.5", linewidth=0.5, linestyle=":")

        if is_gate:
            gate_num += 1
            ax.text((x_start + x_end) / 2.0, y_hi_env + pad * 0.6, f"gate {gate_num}",
                    ha="center", va="bottom", fontsize=8, color="tab:orange")

    # Spawn box: p_x in [0, 2], p_y in [-W/4, W/4] -- matches SlalomEnv.reset().
    ax.add_patch(plt.Rectangle(
        (0.0, -config.tunnel_width / 4.0), 2.0, config.tunnel_width / 2.0,
        fill=False, edgecolor="tab:blue", linestyle="--", linewidth=1.3,
    ))
    ax.text(1.0, -config.tunnel_width / 4.0 - pad * 0.4, "spawn box",
            ha="center", va="top", fontsize=8, color="tab:blue")

    # Goal line.
    ax.axvline(config.tunnel_length, color="tab:green", linestyle="--", linewidth=1.8)
    ax.text(config.tunnel_length, y_hi_env + pad * 0.6, "goal",
            ha="center", va="bottom", fontsize=8, color="tab:green")

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo_env - pad, y_hi_env + pad * 1.6)
    ax.set_aspect("equal")
    ax.set_xlabel("p_x")
    ax.set_ylabel("p_y")
    ax.set_title(f"SlalomEnv track layout  (num_gates={gate_num})")

    param_text = (
        f"dt={config.dt}   v_max={config.v_max}   u_max={config.u_max}   "
        f"sigma_p={config.sigma_p}   sigma_v={config.sigma_v}\n"
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
