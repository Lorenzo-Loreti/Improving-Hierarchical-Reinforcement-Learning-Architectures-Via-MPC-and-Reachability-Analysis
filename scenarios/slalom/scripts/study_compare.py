"""Overlay the finished PPO and hPPO seed studies on the slalom: learning
curves, the first-solve distribution (with a Mann-Whitney test), trajectories
and kinematics against the oracle, and a side-by-side animation. Run
study_ppo.py and study_hppo.py first; this only reads their analysis.pkl.

    python scenarios/slalom/scripts/study_compare.py
    python scenarios/slalom/scripts/study_compare.py --plot-max-steps 120000

Figures go to scenarios/slalom/studies/compare/, which git ignores.
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
    compare_main([os.path.join(STUDIES, "ppo"), os.path.join(STUDIES, "hppo")],
                 os.path.join(STUDIES, "compare"), plot_env)
