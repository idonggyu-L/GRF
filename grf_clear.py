"""Clearing-defender budget episodes on rdsa_2v2c_ab (user proposal, 2026-11-09).

A B-step episode of play, many attacks. When the right team wins the ball (defender tackle
or keeper save) it only CLEARS it: the scene is rebuilt with every player where it stood and
the ball loose CLEAR_BACK behind the spot of the steal (towards our goal), at a random
height -- losing the ball costs the time to fetch it, not the attack. GRF cannot do this
with its own kicks (passes go to the right keeper, the only team-mate; a shot goes at our
goal), hence the rebuild. A goal or the ball going out of play ends the attack and a fresh
one starts as on rdsa_2v2_ab (carrier 'ego'/'partner' by ball_start_p).

All draws (fresh-attack start, engine seed, clearance spot) come from
(SIGMA_SEED, episode seed, attack k, clearance j): every arm meets the same sequence.
"""
import numpy as np
import gfootball.env as fe
import gfootball.scenarios.rdsa_2v2c_ab as S
from grf_crn import prime_abs
SIGMA_SEED = 20260723      # = v1_partner.SIGMA_SEED (not imported: workers would pull in torch)

CLEAR_BACK = (0.1, 0.3)      # how far behind the steal the ball lands (x, our frame)
                             # (0.3, 0.6) until 19:3x: the re-attack from midfield was not learnt in 3M
CLEAR_Y = 0.3                # landing height ~ U(-CLEAR_Y, CLEAR_Y)
X_MIN = -0.5                 # never cleared deeper than this (our keeper's area is ~ -0.85)


def clearance(o, opp_n, rng):
    """o: our-frame raw observation of the step the right team won the ball.
    -> RESTART dict for the scenario (everyone where they stand, ball loose behind the steal)
    and a record of the event."""
    bx, by = (float(v) for v in o['ball'][:2])
    spot = (max(bx - rng.uniform(*CLEAR_BACK), X_MIN), float(rng.uniform(-CLEAR_Y, CLEAR_Y)))
    lt, rt = o['left_team'], o['right_team']
    restart = {'left': [tuple(map(float, lt[1])), tuple(map(float, lt[2]))],
               'right': [tuple(map(float, p)) for p in rt[1:1 + opp_n]], 'ball': spot}
    return restart, {'steal': (bx, by), 'spot': spot,
                     'by': 'keeper' if o['ball_owned_player'] == 0 else 'defender'}


def _checkpoint_wrapper(env):
    e = env
    while e is not None:
        if type(e).__name__ == 'CheckpointRewardWrapper':
            return e
        e = getattr(e, 'env', None)
    return None


def _ckpt_level(cw, d):
    """How many checkpoints a ball carrier at distance d from the goal has passed -- the
    wrapper's own collection loop (gfootball/env/wrappers.py CheckpointRewardWrapper)."""
    n, k = cw._num_checkpoints, 0
    while k < n:
        thr = 0.99 - 0.8 if n == 1 else 0.99 - 0.8 / (n - 1) * k
        if d > thr:
            break
        k += 1
    return k


def rebuild(env, restart, start, eng_seed=None, ckpt='reopen'):
    """Reset into `restart` (None = fresh attack) -> (obs, per-slot reward correction).
    What a clearance does to the checkpoint shaping (the set GRF clears on every reset):
      'keep'    keep the collected set: running the ball up again pays nothing
      'reopen'  (user 19:3x) checkpoints ahead of the clearance spot can be collected again,
                and the clearance takes their value back at once -- potential-based, so
                losing the ball and winning the same ground back nets zero (no farming)
      'free'    reopen without taking anything back"""
    S.START, S.RESTART = start, restart
    if eng_seed is not None:
        prime_abs(env, eng_seed)
    cw = _checkpoint_wrapper(env) if restart is not None else None
    kept = dict(cw._collected_checkpoints) if cw is not None else None
    obs = env.reset()
    pen = np.zeros(2)
    if cw is not None:
        if ckpt == 'keep':
            cw._collected_checkpoints = kept
        else:
            bx, by = restart['ball']
            lvl = _ckpt_level(cw, ((bx - 1) ** 2 + by ** 2) ** 0.5)
            new = {i: min(c, lvl) for i, c in kept.items()}
            if ckpt == 'reopen':
                for i, c in kept.items():
                    pen[i] = (c - new[i]) * cw._checkpoint_reward
            cw._collected_checkpoints = new
    return obs, pen


class ClearingBudget:
    def __init__(self, opp_n=1, budget=400, ball_start_p=0.5, action_set=None, rewards='scoring', def_mate_p=None):
        S.OPP_N = opp_n
        self.opp_n, self.budget, self.p = opp_n, budget, ball_start_p
        self.def_mate_p = def_mate_p          # None: leave S.DEF_AT as it is (default 'carrier')
        extra = {'action_set': action_set} if action_set else {}
        # rewards='scoring,checkpoints' reproduces the training reward (with the clearance's
        # potential-based checkpoint correction) in .last_r -- for a critic's TD error; goals are
        # always counted from the score, so the choice never changes what is reported
        self.env = fe.create_environment(env_name='rdsa_2v2c_ab', representation='simple115v2',
                                         number_of_left_players_agent_controls=2, rewards=rewards,
                                         render=False, other_config_options=extra)
        self.last_r = np.zeros(2)

    def raw(self):
        return self.env.unwrapped.observation()

    def _fresh(self):
        r = np.random.default_rng([SIGMA_SEED, self.seed, self.k, 5])
        start = 'partner' if r.random() < self.p else 'ego'
        if self.def_mate_p is not None:       # defender placement per attack, its own CRN stream
            rd = np.random.default_rng([SIGMA_SEED, self.seed, self.k, 13])
            S.DEF_AT = 'other' if rd.random() < self.def_mate_p else 'carrier'
            self.defs.append(S.DEF_AT)
        self.starts.append(start)
        self.k, self.j = self.k + 1, 0
        return rebuild(self.env, None, start, int(r.integers(0, 2 ** 31 - 1)))[0]

    def begin(self, seed):
        self.seed, self.k, self.j, self.t = int(seed), 0, 0, 0
        self.goals, self.starts, self.clears, self.defs = 0, [], [], []
        return self._fresh()

    def step(self, a):
        """-> obs, goal (0/1), over (budget spent), event in {None,'goal','out','clear'}"""
        obs, rew, done, _ = self.env.step(list(a))
        self.last_r = np.asarray(rew, dtype=np.float64).copy()
        self.t += 1
        o = self.raw()[0]
        goal, ev = int(o['score'][0] > 0), None
        if done:
            ev = 'goal' if goal else 'out'
            self.goals += goal
            if self.t < self.budget:
                obs = self._fresh()
        elif o['ball_owned_team'] == 1:
            ev = 'clear'
            r = np.random.default_rng([SIGMA_SEED, self.seed, self.k, self.j, 9])
            self.j += 1
            restart, rec = clearance(o, self.opp_n, r)
            self.clears.append(dict(rec, t=self.t))
            if self.t < self.budget:
                obs, pen = rebuild(self.env, restart, self.starts[-1], int(r.integers(0, 2 ** 31 - 1)))
                self.last_r = self.last_r - pen
        return obs, goal, self.t >= self.budget, ev

    def close(self):
        self.env.close()
