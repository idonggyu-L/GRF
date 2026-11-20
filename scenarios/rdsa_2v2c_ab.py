"""CLEARING DEFENDER (user proposal, 2026-11-09): rdsa_2v2_ab / rdsa_2v3_ab whose defenders,
on winning the ball, only CLEAR it -- losing the ball costs time, not the attack.

GRF has no "clear the ball" action (the passes go to the right keeper, the only team-mate;
a shot goes at our goal), so the clearance is done by the caller (rdsa/grf_clear.py): when
the right team gains the ball the caller rebuilds the scene with every player where it stood
and the ball LOOSE at a clearance spot, via the module global RESTART read on every reset
(GRF rebuilds the scenario on each reset via config.NewScenario). Possession change therefore
must NOT end the episode here.

Module globals (set by the caller before reset):
  START    'ego' | 'partner'   who carries the ball on a fresh attack (as rdsa_2v2_ab)
  OPP_N    1 | 2               opponent field players (1 = rdsa_2v2_ab, 2 = rdsa_2v3_ab)
  RESTART  None, or {'left': [(x, y), (x, y)], 'right': [(x, y), ...], 'ball': (x, y)}
           positions in OUR (left-team) frame for left[1], left[2], the right field players
           and the loose ball; keepers always go back to their goals.
  KICK_SLOT  True only for the kick-mechanics probe: the right keeper is made uncontrollable
           so one right-team control slot lands on the defender.
Control slots stay pinned (our keeper is uncontrollable): slot0 -> left[1], slot1 -> left[2].
"""
from . import *

START = 'ego'
OPP_N = 1
RESTART = None
KICK_SLOT = False
DEF_AT = 'carrier'   # 'carrier' (default, every trained net) | 'other': with OPP_N=1 the defender starts beside the
                     # attacker WITHOUT the ball instead (user 2026-11-10 probe: carrier free, mate marked)
CARRIER, OTHER = (0.7, -0.3), (0.7, 0.0)
DEF = [(0.75, -0.3), (0.75, 0.0)]           # our frame: beside the carrier / the other attacker


def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False   # the caller decides
    if RESTART is None:
        p1, p2 = (CARRIER, OTHER) if START == 'partner' else (OTHER, CARRIER)
        ball, opp = (0.7, -0.28), (DEF[1:2] if DEF_AT == 'other' and OPP_N == 1 else DEF[:OPP_N])
    else:
        (p1, p2), ball, opp = RESTART['left'], RESTART['ball'], RESTART['right']
    builder.SetBallPosition(ball[0], ball[1])
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(p1[0], p1[1], e_PlayerRole_CB)   # left[1] = slot0 (partner seat)
    builder.AddPlayer(p2[0], p2[1], e_PlayerRole_CB)   # left[2] = slot1 (ego seat)
    builder.SetTeam(Team.e_Right)                     # right-team coordinates are mirrored
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=not KICK_SLOT)
    for x, y in opp:
        builder.AddPlayer(-x, -y, e_PlayerRole_CB)
