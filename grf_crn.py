"""Make episode i the same match in every condition.

Two engine knobs decide the match, and GRF leaves both free:

  game_engine_random_seed   drawn randomly at Config creation if absent, and
                            then REUSED for every reset -- set it once and all
                            episodes become the same game (measured: 120
                            identical episodes), leave it and each process
                            walks its own stream.
  reverse_team_processing   which team the engine steps first. Derived from the
                            PARITY of that seed, and cached on first use
                            (`if 'reverse_team_processing' not in self._config`,
                            scenario_builder.py:86), so it is a single hidden
                            bit fixed for the life of the process -- and it is
                            worth 0.233 vs 0.867 goals on the same config.
                            This is what made the same clean pairing come out
                            bimodal across runs.

So episode i is pinned to (seed0 + i, i % 2): the seed gives the match, and the
parity bit is alternated so a condition is never scored under one processing
order only. Every condition walks the same list, which is the common-random-
numbers the grid claimed but did not have.
"""
from __future__ import annotations


def episode_seed(seed0, i):
    return int(seed0) + int(i), bool(i % 2)


def prime(env, seed0, i):
    """Set the engine's seed and processing order for the NEXT reset."""
    cfg = env.unwrapped._config
    seed, rev = episode_seed(seed0, i)
    cfg._values['game_engine_random_seed'] = seed
    cfg._values['reverse_team_processing'] = rev
    return seed, rev


def prime_abs(env, seed):
    """Same, keyed on one absolute episode seed: the seed picks the match and
    its parity picks the processing order, so a run alternates orders as the
    seed walks and never scores a condition under one order only."""
    cfg = env.unwrapped._config
    cfg._values['game_engine_random_seed'] = int(seed)
    cfg._values['reverse_team_processing'] = bool(int(seed) % 2)
    return int(seed), bool(int(seed) % 2)
