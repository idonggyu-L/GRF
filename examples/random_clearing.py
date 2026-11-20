"""Random attackers on the clearing scenario: a smoke test of the environment.

Runs a few 400-step budget episodes of rdsa_2v2c_ab with both attackers taking uniform
random actions and prints, per episode, the goals, the attacks started and the clearances.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
from grf_clear import ClearingBudget  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--opp-n', type=int, default=1, help='defenders (1 or 2)')
    ap.add_argument('--budget', type=int, default=400, help='steps per episode')
    args = ap.parse_args()
    cb = ClearingBudget(opp_n=args.opp_n, budget=args.budget, ball_start_p=0.5)
    rng = np.random.default_rng(0)
    for ep in range(args.episodes):
        cb.begin(seed=ep)
        over = False
        while not over:
            _, _, over, _ = cb.step(rng.integers(0, 19, size=2))
        print(f'episode {ep}: goals {cb.goals}, attacks {len(cb.starts)} (ego starts {cb.starts.count("ego")}), '
              f'clearances {len(cb.clears)}')
    cb.close()


if __name__ == '__main__':
    main()
