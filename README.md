# GRF_clear

Two-attacker cooperative scenarios for [Google Research Football](https://github.com/google-research/football) (GRF), including a **clearing-defender** variant. In the clearing variant, losing the ball costs the attackers time, not the attack.

In stock GRF, losing possession either ends the episode or hands the ball to a team that then plays on. Here, when a defender or the keeper wins the ball, it is only **cleared**:
- The scene is rebuilt with every player where they stood.
- The ball is left loose a short distance behind the spot where it was won.

The attackers then have to win the ground back. GRF has no "clear the ball" action, so the clearance is done by rebuilding the scenario.

## Files

| File | What it is |
|---|---|
| `scenarios/rdsa_2v2.py` | `academy_pass_and_shoot_with_keeper` with **our keeper uncontrollable**, so the two control slots stay pinned to the two attackers (slot0 → `left[1]`, slot1 → `left[2]`). Opponents: keeper + 1 defender. One attack per episode. |
| `scenarios/rdsa_2v2_ab.py` | `rdsa_2v2` where the ball can start at **either** attacker (module global `START`). |
| `scenarios/rdsa_2v3_ab.py` | `rdsa_2v2_ab` with a **second defender**. |
| `scenarios/rdsa_2v2c_ab.py` | **Clearing** scenario. Possession change does not end the episode; the scene is rebuilt from module globals (see below). 1 or 2 defenders. |
| `grf_clear.py` | The clearing rule (`clearance`, `rebuild`) and `ClearingBudget`, a fixed-length episode with many attacks. |
| `grf_crn.py` | Pins the engine seed and the team-processing order per episode, so episode *i* is the same match for every agent (common random numbers). |
| `grf_vec.py` | `VecGRF`: parallel GRF engines (one process each) behind a batched interface, with optional clearing. |
| `install_scenarios.py` | Copies `scenarios/*.py` into the installed `gfootball.scenarios` package. |
| `examples/random_clearing.py` | Smoke test: random attackers on the clearing scenario. |

## Install

1. Install GRF following the [upstream instructions](https://github.com/google-research/football). Either `pip install .` in a clone, or put a built source checkout on `PYTHONPATH`.

   Tested with upstream commit `3d9e754` (gfootball 2.10.3), Python 3.9, numpy 2.0.2 and gym 0.25.2.
2. Copy the scenarios into the GRF package. GRF loads scenarios by name from `gfootball.scenarios`, so this step is required:
   ```bash
   python install_scenarios.py          # identical files are skipped; --force overwrites files that differ
   ```
3. Run headless:
   ```bash
   export SDL_VIDEODRIVER=dummy
   python examples/random_clearing.py --episodes 3 --opp-n 1
   ```

## The clearing scenario (`rdsa_2v2c_ab`)

- **Players**
  - Left team (ours, attacking towards x = +1): an uncontrollable keeper and two controlled attackers.
  - Right team: a keeper and `OPP_N` defenders.
  - Observations are `simple115v2`.
  - Actions are the default 19; with `action_set='v2'`, action 19 = built-in AI.
- **Fresh attack**
  - The ball carrier starts at (0.7, −0.3) and the other attacker at (0.7, 0.0).
  - `START = 'ego'` gives the ball to slot1 (`left[2]`); `'partner'` gives it to slot0 (`left[1]`).
  - With `OPP_N = 1`, the defender starts beside the carrier (`DEF_AT = 'carrier'`, the default) or beside the other attacker (`DEF_AT = 'other'`).
- **Clearance** (`grf_clear.clearance`): on the step the right team owns the ball (tackle or keeper save):
  - Both attackers and the right-team field players are rebuilt where they stood; both keepers go back to their goals.
  - The ball is placed loose at x = max(x_steal − U(0.1, 0.3), −0.5), y ~ U(−0.3, 0.3).
- **End of an attack**: a goal or the ball going out of play.
- **Checkpoint shaping across a clearance** (`rebuild(..., ckpt=...)`):
  - `'reopen'` (default): checkpoints ahead of the clearance spot can be collected again, and their value is taken back at the clearance. This is potential-based: losing the ball and winning the same ground back nets zero.
  - `'keep'`: keeps the collected set, so running the ball up again pays nothing.
  - `'free'`: reopens the checkpoints without taking anything back.

Scenario settings are **module globals** (`START`, `OPP_N`, `RESTART`, `DEF_AT`), read on every reset because GRF rebuilds the scenario on each reset. They therefore hold one configuration per process. `ClearingBudget` sets them in its own process; `VecGRF` sets them inside each worker.

## Usage

**Fixed-length episodes with many attacks** (evaluation):

```python
from grf_clear import ClearingBudget

cb = ClearingBudget(opp_n=1, budget=400, ball_start_p=0.5)    # P(partner carries) on each fresh attack
obs = cb.begin(seed=0)                                        # obs: (2, 115), one row per control slot
over = False
while not over:
    obs, goal, over, event = cb.step([a_slot0, a_slot1])      # event: None | 'goal' | 'out' | 'clear'
print(cb.goals, len(cb.starts), len(cb.clears))               # goals, attacks, clearances
cb.close()
```

- Every random draw is keyed on (`SIGMA_SEED`, episode seed, attack *k*, clearance *j*). This covers the fresh-attack carrier, the engine seed and the clearance spot.
- As a result, every agent evaluated with the same episode seeds meets the same sequence of situations.
- Goals are read from the engine score, so they never include the checkpoint shaping.

**Parallel engines** (training). Each episode is one attack with clearances. It ends on a goal, on the ball going out of play, or after `clear_len` steps:

```python
from grf_vec import VecGRF

if __name__ == '__main__':                                    # workers use the 'spawn' start method
    env = VecGRF(16, 'rdsa_2v2c_ab', rewards='scoring,checkpoints',
                 ball_start_p=0.5, clear_opp_n=1, clear_len=400, clear_ckpt='reopen')
    obs = env.reset()                                         # (n_env, 2, 115)
    obs, rew, done, infos = env.step(actions)                 # actions: (n_env, 2) int
    # a finished episode's info: ret, len, goal, clears, start, slot_viol
    env.close()
```

- `clear_opp_n` requires `ball_start_p`, because the scenario module is configured only when `ball_start_p` is given.
- Without `clear_opp_n`, `VecGRF` runs any scenario. For example, `rdsa_2v2_ab` with `ball_start_p` gives one attack per episode that ends on possession change.
- `slot_viol` counts steps where the control slots were not on the two attackers; it should be 0.

## Reproducibility note (`grf_crn.py`)

Two engine settings decide the match, and GRF leaves both free:
- `game_engine_random_seed`: if unset, it is drawn once and reused for every reset.
- `reverse_team_processing`: derived from that seed's parity and cached for the life of the process. This hidden bit changed the goal rate of the same pairing from 0.23 to 0.87 in our runs.

`prime_abs(env, seed)` sets both for the next reset: the seed picks the match, and its parity picks the processing order. `ClearingBudget` calls it on every fresh attack and every clearance.
