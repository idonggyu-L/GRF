"""rdsa_2v2 with the ball starting at EITHER attacker -- one attack per episode.

rdsa_2v2 always gives the ball to left[2] (slot1, the ego seat), so the partner
can only ever contribute by receiving a pass: the PR arm learned to pass and
never shoot, SR to never pass, and the two arms shared no skill to blend.

Here the ball carrier always starts in the SAME spot (0.7, -0.3, the defender
beside it) and the other attacker at (0.7, 0.0); only WHO is in which spot is
switched. `START` is a module global read on every reset (GRF rebuilds the
scenario on each reset via config.NewScenario), set by the caller before reset:
  'ego'      left[2] (slot1, ego seat) carries the ball  -- identical to rdsa_2v2
  'partner'  left[1] (slot0, partner seat) carries the ball
Control slots stay pinned (our keeper is uncontrollable): slot0 -> left[1],
slot1 -> left[2], whichever of them starts with the ball.
"""
import os

from . import *

START = 'ego'
# probe knob: number of opponent field players (default 1 = the trained setting)
OPP_N = int(os.environ.get('RDSA_OPP_N', '1'))
CARRIER, OTHER = (0.7, -0.3), (0.7, 0.0)


def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    builder.SetBallPosition(0.7, -0.28)
    p1, p2 = (CARRIER, OTHER) if START == 'partner' else (OTHER, CARRIER)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(p1[0], p1[1], e_PlayerRole_CB)   # left[1] = slot0 (partner seat)
    builder.AddPlayer(p2[0], p2[1], e_PlayerRole_CB)   # left[2] = slot1 (ego seat)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.75, 0.3, e_PlayerRole_CB)      # beside the ball carrier
    if OPP_N >= 2:
        builder.AddPlayer(-0.75, 0.0, e_PlayerRole_CB)  # beside the other attacker
