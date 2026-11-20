"""Parallel GRF engines behind one batched interface.

Measured on this machine (`grf_scaling.py`): a single engine does ~680 steps/s
and 32 workers do ~18k, i.e. 0.84 efficiency, so the workers are worth having --
but everything NOT inside a worker (pipe traffic, the policy forward, the PPO
update) does not shrink with worker count, which is why the trainer keeps the
rollout long and the minibatches large instead of just adding processes.

Each worker owns one engine and both controlled seats.  Observations come back
as one (n_env, 2, 115) array so the policy can score every agent in a single
GPU call.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.environ.get('GFOOTBALL_ROOT')   # a gfootball source checkout; only needed if gfootball is not installed
N_ACTIONS = 19
OBS_DIM = 115

# On `rdsa_2v2` our keeper is `controllable=False`, which pins each control slot
# to one attacker: verified 1984/1984 steps with slot0 on left[1] and slot1 on
# left[2], and our keeper selected 0 times. Without that, slots roam all three
# of our players and "the partner seat" is not a body at all. The guard below
# re-checks the pinning every step, because every downstream claim -- SR's
# training signal, the damage cells, the ego/partner split -- is void without it.
PLAYERS = (1, 2)
ACTIVE_OFF, ACTIVE_W = 97, 11       # simple115v2: the active-player one-hot


def active_players(obs):
    """Which player each control slot is driving, read from the observation."""
    sl = np.asarray(obs)[:, ACTIVE_OFF:ACTIVE_OFF + ACTIVE_W]
    hit = sl.max(axis=1) > 0.5
    idx = sl.argmax(axis=1)
    return [int(i) if h else -1 for i, h in zip(idx, hit)]


def _worker(remote, parent_remote, cfg):
    parent_remote.close()
    if ROOT:
        sys.path.insert(0, ROOT)
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    import gfootball.env as fe

    opts = {}
    if cfg.get('difficulty') is not None:
        opts['left_team_difficulty'] = float(cfg['difficulty'])
    if cfg.get('action_set'):
        opts['action_set'] = cfg['action_set']     # 'v2' adds 19 = builtin_ai
    env = fe.create_environment(
        env_name=cfg['scenario'], representation='simple115v2',
        number_of_left_players_agent_controls=2,
        rewards=cfg['rewards'], render=False, other_config_options=opts)
    # Ball start per episode (rdsa_2v2_ab): the scenario reads a module global on
    # every reset, so it is set here, inside the worker, right before each reset.
    smod, srng, start = None, None, 'ego'
    if cfg.get('ball_start_p') is not None:
        import importlib
        smod = importlib.import_module('gfootball.scenarios.' + cfg['scenario'])
        srng = np.random.default_rng(cfg.get('seed', 0))
    # Clearing defenders (rdsa_2v2c_ab, grf_clear): a right-team ball is rebuilt as a loose
    # ball behind the steal and the episode goes on, up to clear_len steps.
    clear_n = cfg.get('clear_opp_n')
    if clear_n:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import grf_clear
        smod.OPP_N, smod.RESTART = clear_n, None

    # Defender placement per attack (user 2026-11-10): beside the carrier or beside the other attacker,
    # drawn from its own stream so runs without it keep their random streams.
    drng = np.random.default_rng(cfg.get('seed', 0) + 7_777_777) if cfg.get('def_mate_p') is not None else None

    def _reset():
        nonlocal start
        if smod is not None:
            start = 'partner' if srng.random() < cfg['ball_start_p'] else 'ego'
            smod.START, smod.RESTART = start, None
            if drng is not None:
                smod.DEF_AT = 'other' if drng.random() < cfg['def_mate_p'] else 'carrier'
        return env.reset()
    obs = _reset()
    ep_r, ep_t, ep_viol, ep_clr = 0.0, 0, 0, 0
    ep_r_slot = np.zeros(2, dtype=np.float64)   # per-slot, for the single-agent arm

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, r, done, _ = env.step(list(data))
            r = np.asarray(r, dtype=np.float32)
            ep_r += float(r.sum()); ep_r_slot += r; ep_t += 1
            if tuple(sorted(active_players(obs))) != PLAYERS:
                ep_viol += 1
            if clear_n and not done:
                raw = env.unwrapped.observation()
                if raw[0]['ball_owned_team'] == 1:
                    restart, _ = grf_clear.clearance(raw[0], clear_n, srng)
                    obs, pen = grf_clear.rebuild(env, restart, start, ckpt=cfg['clear_ckpt'])
                    r = r - pen.astype(np.float32); ep_r -= float(pen.sum()); ep_r_slot -= pen
                    ep_clr += 1
                done = ep_t >= cfg['clear_len']
            info = {}
            if done:
                # The unshaped outcome: goals, read off the engine rather than
                # inferred from a reward that carries the checkpoint bonus.
                raw = env.unwrapped.observation()
                info = {'ret': ep_r, 'len': ep_t, 'slot_viol': ep_viol,
                        'ret_slot': ep_r_slot.copy(), 'start': start, 'clears': ep_clr,
                        'goal': float(raw[0]['score'][0]) if raw is not None else 0.0}
                obs = _reset()
                ep_r, ep_t, ep_viol, ep_clr = 0.0, 0, 0, 0
                ep_r_slot[:] = 0.0
            remote.send((np.asarray(obs, dtype=np.float32), r, done, info))
        elif cmd == 'raw':
            # The rule bot reads named fields, not the 115-vector, so a worker
            # must be able to hand back the engine's own observation.
            remote.send(env.unwrapped.observation())
        elif cmd == 'reset':
            obs = _reset(); ep_r, ep_t, ep_viol, ep_clr = 0.0, 0, 0, 0
            ep_r_slot[:] = 0.0
            remote.send(np.asarray(obs, dtype=np.float32))
        elif cmd == 'close':
            env.close(); remote.close(); break


class VecGRF:
    def __init__(self, n_env, scenario, rewards='scoring,checkpoints',
                 difficulty=None, start_method='spawn', ball_start_p=None, seed=0,
                 action_set=None, clear_opp_n=None, clear_len=400, clear_ckpt='reopen', def_mate_p=None):
        ctx = mp.get_context(start_method)
        cfg = {'scenario': scenario, 'rewards': rewards, 'difficulty': difficulty,
               'ball_start_p': ball_start_p, 'action_set': action_set,
               'clear_opp_n': clear_opp_n, 'clear_len': clear_len, 'clear_ckpt': clear_ckpt,
               'def_mate_p': def_mate_p}
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_env)])
        self.ps = []
        for k, (wr, r) in enumerate(zip(self.work_remotes, self.remotes)):
            p = ctx.Process(target=_worker, args=(wr, r, dict(cfg, seed=seed * 1000 + k)),
                            daemon=True)
            p.start(); wr.close(); self.ps.append(p)
        self.n_env = n_env
        self.pids = [p.pid for p in self.ps]      # exact PIDs, for clean shutdown

    def reset(self):
        for r in self.remotes:
            r.send(('reset', None))
        return np.stack([r.recv() for r in self.remotes])        # (n,2,115)

    def step(self, actions):
        """actions: (n_env, 2) int"""
        for r, a in zip(self.remotes, actions):
            r.send(('step', a))
        out = [r.recv() for r in self.remotes]
        obs = np.stack([o[0] for o in out])
        rew = np.stack([o[1] for o in out])
        done = np.array([o[2] for o in out], dtype=bool)
        infos = [o[3] for o in out]
        return obs, rew, done, infos

    def raw_obs(self):
        for r in self.remotes:
            r.send(('raw', None))
        return [r.recv() for r in self.remotes]

    def close(self):
        for r in self.remotes:
            try:
                r.send(('close', None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.ps:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()                     # only our own children
