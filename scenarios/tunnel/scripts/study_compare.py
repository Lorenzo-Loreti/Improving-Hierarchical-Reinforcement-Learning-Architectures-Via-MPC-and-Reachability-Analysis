"""Overlay the finished PPO and hPPO seed studies on the tunnel, and PPO+MPC's
once study_ppo_mpc.py has been analyzed: learning curves, the first-solve
distribution (with a Mann-Whitney test when there are two), trajectories and
kinematics against the oracle, and a side-by-side animation. Run the study
scripts first; this only reads their analysis.pkl.

    python scenarios/tunnel/scripts/study_compare.py
    python scenarios/tunnel/scripts/study_compare.py --plot-max-steps 120000

Figures go to scenarios/tunnel/studies/compare/, which git ignores.

The tunnel's seed studies were added on 2026-10-04, with the disk limits and
the effort penalty (scenarios/tunnel/envs/actuation.py, envs/config.py);
before that the tunnel had only the 3-seed benchmark of 2026-09-23
(docs/benchmark.md).
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
# Scenario root, for `envs` (also what unpickles each study's env config);
# then algorithms/, for the flat `study` modules.
sys.path.append(os.path.abspath(os.path.join(HERE, '..')))
sys.path.append(os.path.abspath(os.path.join(HERE, '..', '..', '..', 'algorithms')))
from envs.visualize_env import plot_env
from study import compare_main

STUDIES = os.path.abspath(os.path.join(HERE, '..', 'studies'))

if __name__ == "__main__":
    defaults = [os.path.join(STUDIES, "ppo"), os.path.join(STUDIES, "hppo")]
    # PPO+MPC's study (rebuilt 2026-09-28) joins the default comparison once it exists.
    if os.path.exists(os.path.join(STUDIES, "ppo_mpc", "analysis.pkl")):
        defaults.append(os.path.join(STUDIES, "ppo_mpc"))
    compare_main(defaults, os.path.join(STUDIES, "compare"), plot_env)
