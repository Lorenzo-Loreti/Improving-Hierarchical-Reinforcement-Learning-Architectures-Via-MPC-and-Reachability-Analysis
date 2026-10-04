"""The figures of a seed study (algorithms/study.py), and of a comparison of two.

Everything here draws from an `analysis` dict (study.analyze_phase), never from
a live agent, so a figure can be restyled and redrawn in seconds with the
`plot` phase.

Colors follow one rule across every figure: a color means an algorithm (flat
PPO blue, hPPO orange, the first two slots of a colorblind-validated
categorical palette), the oracle is black and dashed because it is the
reference, and a wall contact is a red cross with its own legend entry.
The track is drawn in neutral greys (plot_env's `overlay` style) so that it
carries no color. The one exception is the grid heatmap, whose diverging
scale (faster / slower than the oracle) has its own colorbar.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from study import seed_table

INK = "#0b0b0b"          # primary text, and the oracle
INK_2 = "#52514e"        # secondary text, manager goals
MUTED = "#898781"        # limits and reference hairlines
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTACT = "#d03b3b"      # "critical" status red: a wall contact is a failure

STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 9, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "text.color": INK, "legend.frameon": False, "legend.fontsize": 8,
    "lines.solid_capstyle": "round", "savefig.dpi": 200, "savefig.bbox": "tight",
}

# The solved-check's worst gap spans ~1000 early in a run and goes negative
# once an agent beats the oracle (see the note on the oracle in study.py), so
# it goes on a symmetric-log axis: linear within +-GAP_LINTHRESH, log beyond.
GAP_LINTHRESH = 10.0

# The heatmap's diverging pair, blue <-> red with a grey midpoint (the
# reference palette's): faster than the oracle <-> slower. The one place a
# color does not mean an algorithm; the colorbar names both ends.
FASTER, EQUAL, SLOWER = "#1c5cab", "#f0efec", "#e34948"

STEPS_FMT = FuncFormatter(lambda x, _: f"{x / 1e3:g}k")


def _save(fig, fig_dir, name):
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  figures/{name}.png")


def _diverging():
    return LinearSegmentedColormap.from_list("faster_slower", [FASTER, EQUAL, SLOWER])


def _stack(analysis, key, max_steps=None):
    """(steps, seeds x evals matrix) of one eval-time curve, cut to the evals
    every seed has (they all do, unless a run was cut short) and to max_steps."""
    curves = [analysis["curves"][s] for s in analysis["seeds"]]
    n = min(len(c["step"]) for c in curves)
    steps = curves[0]["step"][:n]
    values = np.vstack([c[key][:n] for c in curves])
    if max_steps is not None:
        keep = steps <= max_steps
        steps, values = steps[keep], values[:, keep]
    return steps, values


def _first_solves(analysis):
    return [v for v in analysis["first_solve"].values() if v is not None]


def _center(analysis):
    """The representative seed's and the oracle's rollouts from the center start."""
    i = analysis["center"]
    return analysis["agent_grid"][analysis["rep_seed"]][i], analysis["oracle_grid"][i]


def _outcome(ro):
    if not ro["success"]:
        return f"did not reach the goal (stopped at x = {ro['states'][-1, 0]:.1f} m)"
    text = f"{ro['length']} steps"
    n = int(ro["contacts"].sum())
    return text + (f", {n} contact{'s' if n != 1 else ''}" if n else "")


# --------------------------------------------------------------------------
# trajectory drawing
# --------------------------------------------------------------------------

def _draw_track(ax, analysis, plot_env):
    plot_env(analysis["env_config"], ax=ax, overlay=True)
    ax.grid(False)
    ax.set_xlabel("$p_x$ (m)")
    ax.set_ylabel("$p_y$ (m)")


def _draw_traj(ax, ro, color, lw=2.0, ls="-", alpha=1.0, zorder=3, contacts=True):
    s = ro["states"]
    ax.plot(s[:, 0], s[:, 1], color=color, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
    if contacts and ro["contacts"].any():
        # contacts[k] is the step from states[k] to states[k+1]; the clamp
        # puts the agent on the wall at states[k+1].
        hit = s[np.flatnonzero(ro["contacts"]) + 1]
        ax.scatter(hit[:, 0], hit[:, 1], marker="x", s=30, lw=1.5, color=CONTACT, zorder=zorder + 2)


def _draw_start(ax, ro):
    ax.plot(*ro["states"][0, :2], "o", ms=6, mfc="white", mec=INK, mew=1.2, zorder=7)


def _traj_legend(entries, with_contacts=True):
    """entries: (label, color, linestyle) per line."""
    handles = [Line2D([], [], color=c, lw=2 if ls == "-" else 1.4, ls=ls, label=l) for l, c, ls in entries]
    if with_contacts:
        handles.append(Line2D([], [], color=CONTACT, marker="x", ls="", ms=6, mew=1.5, label="wall contact"))
    handles.append(Line2D([], [], color=INK, marker="o", mfc="white", ls="", ms=6, label="start"))
    return handles


# --------------------------------------------------------------------------
# learning curves and first-solve distribution
# --------------------------------------------------------------------------

def fig_learning_curves(analyses, fig_dir, name, plot_max_steps=None):
    """Median and interquartile range over seeds of the four eval-time
    signals; with one study, each seed also as a thin line."""
    single = len(analyses) == 1
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), sharex=True)
    # Steps to the goal rather than the success rate: a wall contact does not
    # end an episode, so every seed reaches the goal from its first
    # evaluation on (100% throughout), and what improves is how fast.
    panels = [
        ("return", "(a) Evaluation return", "return"),
        ("length", "(b) Steps to the goal (evaluation)", "steps per episode"),
        ("contacts", "(c) Wall contacts per evaluation episode", "contacts / episode"),
        ("worst_gap", "(d) Solved-check: worst return gap to the oracle", "worst gap over 25 starts"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        for a in analyses:
            steps, values = _stack(a, key, plot_max_steps)
            if single:
                for row in values:
                    ax.plot(steps, row, color=a["color"], lw=0.7, alpha=0.28, zorder=2)
            q25, q50, q75 = np.percentile(values, [25, 50, 75], axis=0)
            ax.fill_between(steps, q25, q75, color=a["color"], alpha=0.18, lw=0, zorder=1)
            ax.plot(steps, q50, color=a["color"], lw=2.0, zorder=3)
            firsts = _first_solves(a)
            if firsts:
                ax.axvline(np.median(firsts), color=a["color"], lw=0.9, ls=(0, (3, 3)), zorder=0)
        ax.set_title(title)
        if key != "worst_gap":
            ax.set_ylabel(ylabel)
        if key == "return":
            oracle = np.mean([o["return"] for a in analyses for o in a["oracle_eval"].values()])
            ax.axhline(oracle, color=INK, lw=1.2, ls="--", zorder=4)
            ax.text(ax.get_xlim()[1], oracle, f"oracle {oracle:.0f} ", ha="right", va="bottom",
                    fontsize=7.5, color=INK)
        elif key == "length":
            oracle = np.mean([o["length"] for a in analyses for o in a["oracle_eval"].values()])
            ax.axhline(oracle, color=INK, lw=1.2, ls="--", zorder=4)
            ax.text(ax.get_xlim()[1], oracle, f"oracle {oracle:.1f} ", ha="right", va="bottom",
                    fontsize=7.5, color=INK)
        elif key == "worst_gap":
            ax.set_yscale("symlog", linthresh=GAP_LINTHRESH, linscale=1.5)
            ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
            tol = analyses[0]["config"]["solved_tolerance"]
            ax.axhline(tol, color=INK, lw=1.2, ls="--", zorder=4)
            ax.text(ax.get_xlim()[1], tol, f"solved tolerance {tol:g} ", ha="right", va="bottom",
                    fontsize=7.5, color=INK)
            ax.set_ylabel("worst gap over 25 starts\n(< 0: better than the oracle)")
    for ax in axes[1]:
        ax.set_xlabel("environment steps")
        ax.xaxis.set_major_formatter(STEPS_FMT)
    axes[0, 0].set_xlim(0, max(_stack(a, "return", plot_max_steps)[0][-1] for a in analyses))

    handles = []
    for a in analyses:
        n = len(a["seeds"])
        handles.append(Line2D([], [], color=a["color"], lw=2, label=f"{a['label']}: median of {n} seeds"))
        handles.append(plt.Rectangle((0, 0), 1, 1, color=a["color"], alpha=0.18, lw=0,
                                     label=f"{a['label']}: interquartile range"))
    if single:
        handles.append(Line2D([], [], color=analyses[0]["color"], lw=0.7, alpha=0.5, label="one seed"))
    handles.append(Line2D([], [], color=analyses[0]["color"] if single else MUTED, lw=0.9, ls=(0, (3, 3)),
                          label="median first solve"))
    handles.append(Line2D([], [], color=INK, lw=1.2, ls="--", label="oracle / tolerance"))
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08 if single else 0.1, 1, 1))
    _save(fig, fig_dir, name)


def _first_solve_p(analyses):
    """Two-sided Mann-Whitney U on two studies' first-solve steps, formatted.
    A seed that never solved ranks after every seed that did."""
    from scipy.stats import mannwhitneyu
    samples = [[a["first_solve"][s] if a["first_solve"][s] is not None else np.inf
                for s in a["seeds"]] for a in analyses]
    p = mannwhitneyu(*samples).pvalue
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def fig_first_solve_ecdf(analyses, fig_dir, name, plot_max_steps=None):
    """Share of seeds that have passed the strict solved-check, against steps."""
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x_max = plot_max_steps or max(a["total_timesteps"] for a in analyses)
    for a in analyses:
        n = len(a["seeds"])
        firsts = sorted(_first_solves(a))
        xs = [0] + firsts + [x_max]
        ys = [0] + [(i + 1) / n for i in range(len(firsts))] + [len(firsts) / n]
        label = f"{a['label']}: {len(firsts)}/{n} solved"
        if firsts:
            label += f", median {np.median(firsts) / 1e3:.0f}k"
        ax.step(xs, ys, where="post", color=a["color"], lw=2, label=label)
    title = "Seeds that passed the strict solved-check"
    if len(analyses) == 2:
        title += f"  (Mann-Whitney {_first_solve_p(analyses)})"
    ax.set_title(title)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.03)
    ax.xaxis.set_major_formatter(STEPS_FMT)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{100 * y:.0f}%"))
    ax.set_xlabel("environment steps")
    ax.set_ylabel("share of seeds solved")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, fig_dir, name)


# --------------------------------------------------------------------------
# trajectories
# --------------------------------------------------------------------------

def fig_trajectories_vs_oracle(analyses, fig_dir, name, plot_env):
    """The representative seed of each study against the oracle, one panel per
    showcase start."""
    ref = analyses[0]
    rows = len(ref["showcase"])
    fig, axes = plt.subplots(rows, 1, figsize=(7.2, 2.75 * rows + 0.5))
    for ax, i in zip(np.atleast_1d(axes), ref["showcase"]):
        _draw_track(ax, ref, plot_env)
        oracle = ref["oracle_grid"][i]
        _draw_traj(ax, oracle, INK, lw=1.4, ls="--", zorder=4)
        parts = []
        for a in analyses:
            ro = a["agent_grid"][a["rep_seed"]][i]
            _draw_traj(ax, ro, a["color"], zorder=3)
            parts.append(f"{a['label']} {_outcome(ro)}")
        _draw_start(ax, oracle)
        x0, y0 = oracle["states"][0, :2]
        ax.set_title(f"start ({x0:.1f}, {y0:.1f}):  " + ";  ".join(parts)
                     + f";  oracle {oracle['length']} steps")
    entries = [(f"{a['label']} (seed {a['rep_seed']}, {a['checkpoint']}.pt)", a["color"], "-") for a in analyses]
    entries.append(("oracle (one min-time path)", INK, "--"))
    fig.legend(handles=_traj_legend(entries), loc="lower center", ncol=len(entries) + 2,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, fig_dir, name)


def fig_trajectories_grid(analysis, fig_dir, name, plot_env):
    """Oracle and representative seed from all 25 grid starts, one panel each."""
    a = analysis
    agent = a["agent_grid"][a["rep_seed"]]
    reached = sum(r["success"] for r in agent)
    extra = np.mean([r["length"] - o["length"] for r, o in zip(agent, a["oracle_grid"])])
    contacts = sum(int(r["contacts"].sum()) for r in agent)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2))
    _draw_track(axes[0], a, plot_env)
    for ro in a["oracle_grid"]:
        _draw_traj(axes[0], ro, INK, lw=1.0, alpha=0.55)
    axes[0].set_title(f"Oracle from the {len(agent)} grid starts (least control effort among the min-time paths)")
    _draw_track(axes[1], a, plot_env)
    for ro in agent:
        _draw_traj(axes[1], ro, a["color"], lw=1.1, alpha=0.65)
    axes[1].set_title(f"{a['label']}, seed {a['rep_seed']} ({a['checkpoint']}.pt), same starts: "
                      f"{reached}/{len(agent)} reach the goal, {extra:+.1f} steps vs the oracle on average, "
                      f"{contacts} contacts")
    for ax in axes:
        ax.plot(a["grid"][:, 0], a["grid"][:, 1], "o", ms=3.5, mfc="white", mec=INK, mew=0.9, zorder=7)
    fig.tight_layout()
    _save(fig, fig_dir, name)


def fig_all_seeds(analysis, fig_dir, name, plot_env):
    """Every seed's policy from the center start, the representative one bold."""
    a = analysis
    i = a["center"]
    oracle = a["oracle_grid"][i]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    _draw_track(ax, a, plot_env)
    lengths = []
    for seed in a["seeds"]:
        ro = a["agent_grid"][seed][i]
        lengths.append(ro["length"] if ro["success"] else None)
        if seed != a["rep_seed"]:
            _draw_traj(ax, ro, a["color"], lw=1.0, alpha=0.4)
    _draw_traj(ax, a["agent_grid"][a["rep_seed"]][i], a["color"], lw=2.2, zorder=5)
    _draw_traj(ax, oracle, INK, lw=1.4, ls="--", zorder=6)
    _draw_start(ax, oracle)
    reached = [l for l in lengths if l is not None]
    span = f"{min(reached)}-{max(reached)} steps" if reached else "none reached the goal"
    ax.set_title(f"{a['label']}, all {len(a['seeds'])} seeds from ({oracle['states'][0, 0]:.1f}, "
                 f"{oracle['states'][0, 1]:.1f}): {len(reached)} reach the goal, {span} "
                 f"(oracle {oracle['length']})")
    handles = _traj_legend([(f"seed {a['rep_seed']} (representative)", a["color"], "-"),
                            ("oracle", INK, "--")])
    handles.insert(1, Line2D([], [], color=a["color"], lw=1.0, alpha=0.4, label="other seeds"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5)
    fig.tight_layout()
    _save(fig, fig_dir, name)


def fig_kinematics(analyses, fig_dir, name):
    """Velocity and acceleration against time, from the center start: both
    components and the magnitudes, ||v|| and ||u||, which are what the limits
    bound since 2026-10-04 (envs/actuation.py; the panels were v_x, v_y, a_x,
    a_y against per-axis limits before). The oracle reaches v_max within
    about a second and then cruises; its thrust starts at u_max and tapers off
    rather than being bang-bang: the arrival step is an integer, which leaves
    slack, and among the paths that arrive on it the oracle takes the one with
    the least control effort -- which the reward now charges too (the
    effort_penalty in envs/config.py). Before the speed-limit fix of
    2026-09-24 this was the figure that showed trained agents holding a_x near
    u_max while v_x was pinned at v_max, which the environment rewarded with
    extra distance per step (see the note on the oracle in study.py); thrust
    at the limit now buys nothing. The accelerations are the ones the plant
    delivered (the rollout's "actions")."""
    ref = analyses[0]
    cfg = ref["env_config"]
    dt = cfg.dt
    oracle = ref["oracle_grid"][ref["center"]]
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 4.8), sharex=True)
    series = [(a["label"], a["color"], "-", _center(a)[0]) for a in analyses]
    series.append(("oracle", INK, "--", oracle))
    for label, color, ls, ro in series:
        t_state = np.arange(len(ro["states"])) * dt
        t_act = np.arange(len(ro["actions"]) + 1) * dt
        lw = 1.3 if ls == "--" else 1.8
        v = ro["states"][:, 2:]
        for y, ax in ((v[:, 0], axes[0, 0]), (v[:, 1], axes[0, 1]), (np.linalg.norm(v, axis=1), axes[0, 2])):
            ax.plot(t_state, y, color=color, ls=ls, lw=lw)
        acts = ro["actions"]
        for y, ax in ((acts[:, 0], axes[1, 0]), (acts[:, 1], axes[1, 1]),
                      (np.linalg.norm(acts, axis=1), axes[1, 2])):
            ax.step(t_act, np.append(y, y[-1]), where="post", color=color, ls=ls, lw=lw)
    for ax, title in ((axes[0, 0], "(a) $v_x$ (m/s)"), (axes[0, 1], "(b) $v_y$ (m/s)"),
                      (axes[1, 0], "(d) $a_x$ (m/s²)"), (axes[1, 1], "(e) $a_y$ (m/s²)")):
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(title)
    for ax, title, lim in ((axes[0, 2], r"(c) speed $\|v\|$ (m/s)", cfg.v_max),
                           (axes[1, 2], r"(f) thrust $\|a\|$ (m/s²)", cfg.u_max)):
        ax.axhline(lim, color=MUTED, lw=0.8)
        ax.set_ylim(0, 1.08 * lim)
        ax.set_title(title)
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    start = oracle["states"][0, :2]
    handles = [Line2D([], [], color=c, ls=ls, lw=1.8 if ls == "-" else 1.3, label=l)
               for l, c, ls, _ in series]
    handles.append(Line2D([], [], color=MUTED, lw=0.8, label="limits $v_{max}$, $u_{max}$"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Kinematics from ({start[0]:.1f}, {start[1]:.1f})", x=0.01, ha="left",
                 fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, fig_dir, name)


def fig_grid_heatmap(analysis, fig_dir, name):
    """Arrival step minus the oracle's, at every grid start: for the
    representative seed, and the slowest seed at each start. Negative means
    the agent arrived first (see the note on the oracle in study.py)."""
    a = analysis
    grid = a["grid"]
    xs, ys = np.unique(grid[:, 0]), np.unique(grid[:, 1])

    def matrices(rollouts):
        extra = np.zeros((len(ys), len(xs)))
        contacts = np.zeros_like(extra, dtype=int)
        failed = np.zeros_like(extra, dtype=bool)
        for point, ro, oracle in zip(grid, rollouts, a["oracle_grid"]):
            iy, ix = np.searchsorted(ys, point[1]), np.searchsorted(xs, point[0])
            extra[iy, ix] = ro["length"] - oracle["length"]
            contacts[iy, ix] = int(ro["contacts"].sum())
            failed[iy, ix] = not ro["success"]
        return extra, contacts, failed

    rep = matrices(a["agent_grid"][a["rep_seed"]])
    per_seed = [matrices(a["agent_grid"][s]) for s in a["seeds"]]
    worst = (np.max([m[0] for m in per_seed], axis=0), np.max([m[1] for m in per_seed], axis=0),
             np.any([m[2] for m in per_seed], axis=0))
    vlim = max(3.0, np.abs(rep[0]).max(), np.abs(worst[0]).max())
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    extent = (xs[0] - dx / 2, xs[-1] + dx / 2, ys[0] - dy / 2, ys[-1] + dy / 2)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), constrained_layout=True)
    for ax, (extra, contacts, failed), title in (
            (axes[0], rep, f"(a) seed {a['rep_seed']} (representative)"),
            (axes[1], worst, f"(b) slowest of the {len(a['seeds'])} seeds at each start")):
        im = ax.imshow(extra, origin="lower", extent=extent, cmap=_diverging(), vmin=-vlim, vmax=vlim,
                       aspect="auto")
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                text = "fail" if failed[iy, ix] else f"{extra[iy, ix]:+.0f}".replace("+0", "0")
                if contacts[iy, ix]:
                    text += f"\n{contacts[iy, ix]}×"
                dark = abs(extra[iy, ix]) > 0.55 * vlim
                ax.text(x, y, text, ha="center", va="center", fontsize=7.5,
                        color="white" if dark else INK)
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.set_yticks(ys)
        ax.grid(False)
        ax.set_xlabel("start $p_x$ (m)")
    axes[0].set_ylabel("start $p_y$ (m)")
    cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label("arrival step - oracle's\n(< 0: faster than the oracle)")
    cbar.outline.set_visible(False)
    fig.suptitle(f"{a['label']} ({a['checkpoint']}.pt) from the solved-check grid; "
                 f"N× = wall contacts in the episode", x=0.01, ha="left", fontsize=9.5, fontweight="bold")
    _save(fig, fig_dir, name)


def fig_progression(analysis, fig_dir, name, plot_env):
    """The representative seed's policy at a few evaluations through training,
    from the center start."""
    a = analysis
    panels = a["progression"]
    oracle = a["oracle_grid"][a["center"]]
    cols = 2
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.4, 1.9 * rows + 0.6))
    axes = np.atleast_2d(axes)
    first = a["first_solve"][a["rep_seed"]]
    for ax, (step, ro) in zip(axes.flat, panels):
        _draw_track(ax, a, plot_env)
        _draw_traj(ax, oracle, INK, lw=1.0, ls="--", zorder=4)
        _draw_traj(ax, ro, a["color"], lw=1.8)
        _draw_start(ax, oracle)
        tag = " (first solve)" if step == first else (" (end)" if step == panels[-1][0] else "")
        ax.set_title(f"{step / 1e3:.0f}k steps{tag}: {_outcome(ro)}", fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    handles = _traj_legend([(f"{a['label']}, seed {a['rep_seed']}", a["color"], "-"), ("oracle", INK, "--")])
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"How the policy learns: {a['label']} seed {a['rep_seed']} from "
                 f"({oracle['states'][0, 0]:.1f}, {oracle['states'][0, 1]:.1f}) at evaluation checkpoints",
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, fig_dir, name)


def _goal_cap(analysis):
    """The distance the plant can cover in one manager segment,
    v_max * manager_freq * dt: goal arrows are drawn at most this long."""
    cfg = analysis["env_config"]
    return cfg.v_max * analysis["config"]["manager_freq"] * cfg.dt


def _capped(goals, cap):
    norm = np.linalg.norm(goals, axis=-1, keepdims=True)
    return goals * np.minimum(1.0, cap / np.maximum(norm, 1e-9))


def fig_subgoals(analysis, fig_dir, name, plot_env):
    """What the manager asks for: its goal at every re-plan along the
    representative seed's center-start trajectory, and the goal the worker
    tracks in between (it decays by the agent's displacement)."""
    a = analysis
    ro, oracle = _center(a)
    goals, replanned = ro["goals"], ro["replanned"]
    pos = ro["states"][:-1, :2]    # the policy acted from states[k] at step k
    cap = _goal_cap(a)
    dt = a["env_config"].dt
    fig = plt.figure(figsize=(7.4, 7.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1, 1])
    ax = fig.add_subplot(gs[0])
    _draw_track(ax, a, plot_env)
    _draw_traj(ax, oracle, INK, lw=1.0, ls="--", zorder=3)
    _draw_traj(ax, ro, a["color"], lw=1.8, zorder=4)
    k = np.flatnonzero(replanned)
    arrows = _capped(goals[k], cap)
    ax.quiver(pos[k, 0], pos[k, 1], arrows[:, 0], arrows[:, 1], angles="xy", scale_units="xy", scale=1,
              color=INK_2, width=0.004, headwidth=4, headlength=5, zorder=6)
    ax.plot(pos[k, 0], pos[k, 1], "o", ms=5, mfc="white", mec=a["color"], mew=1.4, zorder=7)
    ax.set_title(f"(a) {a['label']} seed {a['rep_seed']}: re-plans every {a['config']['manager_freq']} "
                 f"steps (circles), arrows toward the goal (capped at {cap:.1f} m)")
    t = np.arange(len(goals)) * dt
    for sub, col, label in ((gs[1], 0, "(b) goal $g_x$ (m), relative to the agent"),
                            (gs[2], 1, "(c) goal $g_y$ (m), relative to the agent")):
        axg = fig.add_subplot(sub)
        for tk in t[k]:
            axg.axvline(tk, color=GRID, lw=0.9, zorder=0)
        axg.plot(t, goals[:, col], color=a["color"], lw=1.8, drawstyle="steps-post")
        axg.plot(t[k], goals[k, col], "o", ms=4, mfc="white", mec=a["color"], mew=1.2)
        axg.axhline(0, color=MUTED, lw=0.8)
        axg.set_title(label)
        axg.set_xlim(0, t[-1] + dt)
    axg.set_xlabel("time (s); vertical lines = re-plans")
    fig.tight_layout()
    _save(fig, fig_dir, name)


# --------------------------------------------------------------------------
# animation
# --------------------------------------------------------------------------

def save_animation(entries, oracle, analysis, plot_env, fig_dir, name, fps=10, hold=15):
    """Agents and the oracle ghost moving together, in real time at fps=10
    (dt = 0.1 s). `entries` is a list of (label, color, rollout, goal_cap or
    None); a hierarchical entry also shows its manager's current goal."""
    os.makedirs(fig_dir, exist_ok=True)
    dt = analysis["env_config"].dt
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    _draw_track(ax, analysis, plot_env)
    movers = []
    for label, color, ro, cap in entries:
        trail, = ax.plot([], [], color=color, lw=2, zorder=4)
        head, = ax.plot([], [], "o", ms=8, color=color, mec="white", mew=1.2, zorder=6)
        quiv = None
        if cap is not None and ro["goals"] is not None:
            quiv = ax.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1, color=INK_2,
                             width=0.004, headwidth=4, headlength=5, zorder=5)
        movers.append((ro, trail, head, quiv, cap))
    o_trail, = ax.plot([], [], color=INK, lw=1.2, ls="--", zorder=3)
    o_head, = ax.plot([], [], "o", ms=8, mfc="none", mec=INK, mew=1.4, zorder=5)
    clock = ax.text(0.01, 0.97, "", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    handles = [Line2D([], [], color=c, lw=2, marker="o", label=l) for l, c, _, _ in entries]
    handles.append(Line2D([], [], color=INK, lw=1.2, ls="--", marker="o", mfc="none", label="oracle"))
    if any(e[3] is not None for e in entries):
        handles.append(Line2D([], [], color=INK_2, lw=1.2, marker=">", ms=4, label="manager goal"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(handles))
    fig.tight_layout()
    frames = max(len(ro["states"]) for ro in [oracle] + [e[2] for e in entries]) + hold

    def update(frame):
        for ro, trail, head, quiv, cap in movers:
            k = min(frame, len(ro["states"]) - 1)
            trail.set_data(ro["states"][:k + 1, 0], ro["states"][:k + 1, 1])
            head.set_data([ro["states"][k, 0]], [ro["states"][k, 1]])
            if quiv is not None:
                if k < len(ro["goals"]):
                    g = _capped(ro["goals"][k:k + 1], cap)[0]
                    quiv.set_offsets([ro["states"][k, :2]])
                    quiv.set_UVC([g[0]], [g[1]])
                else:
                    quiv.set_UVC([0], [0])
        k = min(frame, len(oracle["states"]) - 1)
        o_trail.set_data(oracle["states"][:k + 1, 0], oracle["states"][:k + 1, 1])
        o_head.set_data([oracle["states"][k, 0]], [oracle["states"][k, 1]])
        arrived = [f"{label} {ro['length'] * dt:.1f} s" for (label, _, ro, _) in entries
                   if ro["success"] and frame >= ro["length"]]
        if frame >= oracle["length"]:
            arrived.append(f"oracle {oracle['length'] * dt:.1f} s")
        clock.set_text(f"t = {min(frame, frames - hold) * dt:.1f} s"
                       + ("   arrived: " + ", ".join(arrived) if arrived else ""))
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    anim.save(os.path.join(fig_dir, f"{name}.gif"), writer=animation.PillowWriter(fps=fps), dpi=110)
    print(f"  figures/{name}.gif")
    if animation.writers.is_available("ffmpeg"):
        anim.save(os.path.join(fig_dir, f"{name}.mp4"), writer=animation.FFMpegWriter(fps=fps, bitrate=3000),
                  dpi=200)
        print(f"  figures/{name}.mp4")
    plt.close(fig)


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def plot_study(study, analysis, out_dir, plot_max_steps=None, animate=True):
    fig_dir = os.path.join(out_dir, "figures")
    a = analysis
    print(f"[{a['label']}] drawing figures")
    with plt.rc_context(STYLE):
        fig_learning_curves([a], fig_dir, "learning_curves", plot_max_steps)
        fig_first_solve_ecdf([a], fig_dir, "first_solve_ecdf", plot_max_steps)
        fig_trajectories_vs_oracle([a], fig_dir, "trajectories_vs_oracle", study.plot_env)
        fig_trajectories_grid(a, fig_dir, "trajectories_grid", study.plot_env)
        fig_all_seeds(a, fig_dir, "trajectories_all_seeds", study.plot_env)
        fig_kinematics([a], fig_dir, "kinematics")
        fig_grid_heatmap(a, fig_dir, "grid_extra_steps")
        if a["progression"]:
            fig_progression(a, fig_dir, "progression", study.plot_env)
        if a["hierarchical"]:
            fig_subgoals(a, fig_dir, "subgoals", study.plot_env)
        if animate:
            ro, oracle = _center(a)
            cap = _goal_cap(a) if a["hierarchical"] else None
            save_animation([(f"{a['label']} (seed {a['rep_seed']})", a["color"], ro, cap)], oracle, a,
                           study.plot_env, fig_dir, "animation")


def plot_comparison(analyses, out_dir, plot_env, plot_max_steps=None, animate=True):
    fig_dir = os.path.join(out_dir, "figures")
    labels = " vs ".join(a["label"] for a in analyses)
    print(f"[{labels}] drawing comparison figures")
    if plot_max_steps is None:
        plot_max_steps = min(a["total_timesteps"] for a in analyses)
    with plt.rc_context(STYLE):
        fig_learning_curves(analyses, fig_dir, "compare_learning_curves", plot_max_steps)
        fig_first_solve_ecdf(analyses, fig_dir, "compare_first_solve_ecdf", plot_max_steps)
        fig_trajectories_vs_oracle(analyses, fig_dir, "compare_trajectories", plot_env)
        fig_kinematics(analyses, fig_dir, "compare_kinematics")
        if animate:
            entries = [(f"{a['label']} (seed {a['rep_seed']})", a["color"], _center(a)[0],
                        _goal_cap(a) if a["hierarchical"] else None) for a in analyses]
            save_animation(entries, _center(analyses[0])[1], analyses[0], plot_env, fig_dir,
                           "compare_animation")
    write_comparison_summary(analyses, out_dir)


def write_comparison_summary(analyses, out_dir):
    lines = ["# Seed-study comparison", "",
             "| algorithm | seeds | solved within budget | first solve median / mean | final eval return "
             "| % of oracle | grid: starts reached on the oracle's step | grid: mean extra steps "
             "| grid: contacts |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for a in analyses:
        rows = seed_table(a)
        firsts = _first_solves(a)
        extra = np.array([[r["length"] - o["length"] for r, o in zip(a["agent_grid"][s], a["oracle_grid"])]
                          for s in a["seeds"]])
        lines.append(
            f"| {a['label']} | {len(rows)} | {len(firsts)}/{len(rows)} | "
            + (f"{np.median(firsts) / 1e3:.0f}k / {np.mean(firsts) / 1e3:.0f}k" if firsts else "-")
            + f" | {np.mean([r['final_eval_return'] for r in rows]):.1f} "
            f"| {np.mean([r['pct_of_oracle'] for r in rows]):.1f} % "
            f"| {100 * np.mean(extra <= 0):.0f} % of {extra.size} | {extra.mean():+.2f} "
            f"| {sum(r['grid_contacts'] for r in rows)} |")
    if len(analyses) == 2:
        lines += ["", f"First-solve steps, Mann-Whitney U (two-sided): {_first_solve_p(analyses)}. "
                      "A seed that never solved ranks after every seed that did."]
    with open(os.path.join(out_dir, "compare_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  compare_summary.md")
