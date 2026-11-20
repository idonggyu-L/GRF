# coding=utf-8
"""academy_pass_and_shoot_with_keeper with OUR keeper made UNCONTROLLABLE.

Why: `number_of_left_players_agent_controls=2` does not pin a control slot to
a player -- `obs[i]['active']` is whatever body the engine has slot i driving
this step, and it changes mid-episode. With the stock left team (GK + 2 CB)
both slots roam all three, so a slot spends a fifth of the episode driving OUR
OWN keeper. "Partner" then means "whichever of our three players that slot
happens to hold", which is not a teammate at all: a random partner randomises
our keeper, and an idle partner freezes the ball carrier.

`controllable=False` is the engine's own switch for this: Team::SelectPlayer
(team.cpp:316) only hands a player to a controller when the flag is set, so
the keeper stays on the pitch and keeps playing, but no control slot can ever
be assigned to him. The controlled set is then exactly the two attackers.

(Deleting our keeper instead also leaves two attackers, but the engine hangs
on a keeper-less team -- episodes stopped terminating. Not worth pursuing.)

Right team untouched: right[0] is the keeper we must beat, right[1] the
defender starting beside our ball carrier.
"""
from . import *


def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    builder.SetBallPosition(0.7, -0.28)

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(0.7, 0.0, e_PlayerRole_CB)      # left[1]: the outlet
    builder.AddPlayer(0.7, -0.3, e_PlayerRole_CB)     # left[2]: starts with the ball

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.75, 0.3, e_PlayerRole_CB)
