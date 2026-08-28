# AI Tom & Jerry

A cat has to catch a mouse. The mouse has to reach her hole. Three algorithms — PPO, a
genetic algorithm, and CMA-ES — each raise their own cat and their own mouse on the same
twelve rooms, and then the best Tom meets the best Jerry.

Everything on screen is real. Real policies, trained here; real episodes, streamed live
from Python while you record; real numbers, with the error bars shown.

```
viz/          the original Claude Design artboard — reference art and the first flow
app/          the recording app: plain HTML + JS, fixed 1920x1080, no build step
trainer/      Python: environment, three algorithms, tournament, live server
assets-src/   the five source renders the eight app assets are built from
runs/         checkpoints, telemetry, journals, tournament results (git-ignored)
```

## Run it

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python torch numpy pillow websockets

# train all three schools in parallel — same wall-clock, same machine, same load
.venv/bin/python trainer/scripts/train.py --minutes 45 --tag v4

# decide the championship on arenas nobody trained on
.venv/bin/python trainer/scripts/tournament_run.py --run runs/v4

# serve the app
.venv/bin/python trainer/scripts/serve.py --run runs/v4 &
python3 tools/serve_app.py
```

Then open <http://localhost:8778>. **[SHOOT.md](SHOOT.md) is the run of show** — a
suggested recording order tied to the keys, with the data beats worth pointing at.

**Keys.** `1 2 3` schools · `x` side by side · `g` level generator · `f` grand final · `b` leaderboard ·
`l` full-screen lesson · `w` how the algorithm works · `v` verdict · `esc` menu · `space` pause ·
`s` skip an episode · `[` `]` speed ·
`h` highlight reel · `t` train live on camera · `?` key card · **`r` reveal the next school**, `shift+R` re-seal one, `shift+0` re-seal all.

## The reveal

Your intro names only the first school. Press `r` on camera and the next one opens — its
name, emblem, accent colour, building and explainer all appear at once. Until then it
shows a CLASSIFIED plate, a stable codename and REDACTED specs, so it reads as
deliberately sealed rather than broken, and nothing needs blurring in the edit.

The level survives a reload (it is in `localStorage`), so a crash mid-shoot does not
unseal the rest of the video. Every screen reads a school's identity through
`Reveal.view()`, so a new screen cannot leak one by forgetting to check.

## How many holes

```bash
./run.sh train 45 v4 --nests 2      # the default
```

**One hole is not a fair game.** The cat's best strategy is simply to stand on it: the
mouse has no decision to make and the chase collapses into a stakeout. Measured on the
scripted controller at low skill, 27% of one-hole episodes died on the step limit.

Two holes, kept at least 10 cells apart so one cat cannot cover both, fix it without
changing the balance at the top:

| holes | avg trek | Tom @0.65 | Jerry @0.65 | stakeouts @0.35 | avg steps |
|---|---:|---:|---:|---:|---:|
| 1 | 28.5 | 33.8% | 64.6% | **27.3%** | 74 |
| **2** | 18.4 | 31.2% | 64.6% | **7.9%** | 54 |
| 3 | 13.6 | 20.2% | 75.8% | 4.6% | 44 |

Two is the default: the same high-skill balance as one hole, a third of the stakeouts,
and shorter, punchier episodes. Three tips it decisively to the mouse.

`--nests 1,2,3` mixes counts across the twelve rooms, so a policy learns to handle any of
them. The observation always carries `MAX_NESTS` hole slots — bearing, distance and a
valid flag each, nearest first — so one trained network can play a one-hole room or a
three-hole room without being reshaped. Changing the count changes `OBS_DIM`, so
checkpoints from a different `MAX_NESTS` build are refused with an explanation rather
than a reshape error.

## Side by side

`x` runs all three schools on the **same room at the same time**: same map, same spawns,
the same hearing-noise stream, and the same action-sampling draws shared across all three
panes. Every difference on screen is therefore the policy and nothing else — which is what
the spec means by "the comparison is the whole point". Underneath, a grid of the same
twelve rooms: read down a column and you can see whether a room is hard or whether one
school simply solved it.

## Fairness — the part that makes this an experiment

Self-play alone proves nothing across schools. A GA mouse escaping 80% of the time may
just mean the GA cat is bad. This run shows exactly that: **CMA-ES's cat catches 52% of
its own mouse and 23% of PPO's** — the same policy, a 29-point spread that depends
entirely on who is being chased.

So the championship is decided by **cross-play**: every cat plays every mouse, on eight
arenas that appear in no training set, and a cat's score is its mean catch rate against
the two mice it did *not* grow up with. Alongside it sits a **fixed anchor** — the scripted controller at skill
0.60, a difficulty absent from every school's training ladder — which puts all six
policies on one absolute axis.

Controls that make the comparison mean something:

- **Same brain.** One architecture, 2,853 parameters, one observation vector, for all
  three. PPO additionally trains a critic; it never plays and is discarded at evaluation.
- **Same rooms.** One seeded 12-arena training set shared by all three; separate held-out
  arenas for scoring.
- **Same clock, both clocks.** All three train in parallel for the same wall-clock, and
  every log line carries environment steps too, so "equal compute" and "equal samples"
  can both be read off one run.
- **Same league.** Identical opponent mix and identical promotion rule (below).
- **Error bars.** Wilson 95% intervals on every rate, and the leaderboard says out loud
  when the top two overlap.

## Four things the data forced

These are not design preferences. Each replaced something that measurably did not work,
and each is worth saying out loud on camera because the failure is more interesting than
the fix.

**1 · Pure self-play does not climb.** Over 432 CMA-ES generations the cat reached a 74%
win rate against its own archive while scoring 8% against a fixed opponent; the mouse
reached 45% and scored 0%. Both sides only ever had to beat *each other*, so they settled
into a low-level arms race. Training now faces a league: three quarters a curriculum of
scripted opponents at five difficulties, one quarter self-play against a hall of fame.

**2 · A fixed curriculum punishes the slower learner.** Hardening the ladder on a clock
means the last third of the budget is spent losing every episode. A school is now
**promoted a year** when it sustains a win rate against its current year — measured only
against the scripted slots, so it cannot promote itself by beating its own weak opponent.
How fast each school gets promoted is itself worth putting on screen.

**3 · One promotion bar is not one bar.** The first full 45-minute run promoted both
population schools' *mice* to year three within 50M steps and left their *cats* in year
one for the entire run, plateauing just under a flat 40%. Catching is simply harder than
escaping here: measured on the year-one ladder, the scripted controller at that year's
top skill catches 58% and escapes 69%. So the bar is now **60% of whatever the scripted
controller itself manages on the same ladder** — computed in 1.6s at setup, per role, per
year. Year one comes out at 34% for the cat and 41% for the mouse. Same rule for every
school; no constant that happens to suit one role.

**4 · The mouse was better off hiding.** With the spec's shaping coefficients, a timeout
scores 0 and being caught scores −1, while running the whole way home earns only +0.25 of
shaping. The mouse duly learned to hide behind cover until the step limit and never
escaped once in 400 generations. Training now uses potential-based shaping (Ng, Harada &
Russell), which cannot invent a strategy that was not already optimal — it only makes the
good one findable. The measured effect: GA's mouse went from a flat 0% to a 34% ladder win
rate in 157 generations, with episodes shortening from 164 steps to 118 as it stopped
hiding and started running. **Scoring is untouched** — catch, escape and timeout rates are
counted exactly as the spec defines them.

## The result

Three full 45-minute runs. The first two used one hole and differed only in the
promotion rule; the third uses two, which is a different game. Cross-play scores on the
eight held-out rooms, 320 episodes a pairing:

| | Tom, 1 hole | Tom, 2 holes | Jerry, 1 hole | Jerry, 2 holes |
|---|---:|---:|---:|---:|
| **PPO** | **82.3%** ±3.0 | **74.4%** ±3.4 | **41.4%** ±3.8 | **51.9%** ±3.9 |
| GA | 54.1% ±3.8 | 21.9% ±3.2 | 23.9% ±3.3 | 41.9% ±3.8 |
| CMA-ES | 40.6% ±3.8 | 28.1% ±3.5 | 13.6% ±2.7 | 38.0% ±3.7 |

SCORE here is the mean against the **other two** schools. A school's own mouse is left
out of its cat's score and vice versa, because averaging it back in is precisely the
route by which a weak sparring partner at home flatters a score — the thing this whole
tournament exists to rule out. The diagonal is still on the leaderboard, and the gap
between it and the rest is the interesting part; it simply does not vote.

That exclusion is not cosmetic. With the diagonal counted, the two-hole board crowned
GA's mouse at 49.3% — on the strength of 64.1% against GA's *own* cat, the worst cat in
the field at 21.9%. Off the diagonal, PPO's mouse leads at 51.9% against GA's 41.9%, and
the intervals no longer touch.

**PPO wins both roles at both hole counts.** Adding the second hole did not change who
wins, but it changed everything about the margins:

- Every cat got worse and every mouse got better, which is the point of the change — with
  one exit, standing on it was most of the job.
- **GA's mouse gained the most** — 23.9% to 41.9%, from last place to within a stride of
  PPO — and on the fixed anchor it actually leads (41.6% vs 38.4%). Breeding whole brains
  is a good way to learn "run for whichever door he is not covering".
- **GA's cat collapsed** to 21.9% and never got past year two. The same method is a poor
  way to learn "cover two doors at once".

So the split is not in who wins but in how much: **PPO raises the best hunter and the
best escape artist, but the second hole turns GA from the worst Jerry into the closest
challenger.** That fell out of a rules change, not out of tuning.

Three things worth pointing at on camera:

- **The diagonal is the lie, and it is not in the score.** PPO's cat catches 70–79% of
  the other schools' mice and only 56% of its own — because its own mouse is the hardest
  opponent its cat ever meets, and the best mouse in the field besides. Read the other
  two rows the other way round: a school that raised a weak sparring partner looks
  dominant at home. That is why SCORE averages the off-diagonal only.
- **CMA-ES's cat catches 52% of its own mouse and 23% of PPO's.** The same policy, a
  29-point spread. Any single matchup would have been a meaningless ranking.
- **Beating the benchmark is not being good.** Compare any school's `vs EXAMINER` column
  against its cross-play score. On the two-hole board the Examiner is the *harder*
  opponent for every cat — PPO's catches 74% of the learned mice and 42% of the scripted
  one — which is the same lesson from the other side: one fixed opponent ranks nobody.

## What the explainers draw

Nothing is a mock-up; every panel is fed from its own optimiser while it runs.

- **PPO** — the five action probabilities on a fixed probe batch, and the importance
  ratio against the clip band. The histogram piles up inside the band: that *is* the
  clipped objective, and you can watch the update not being allowed to move.
- **GA** — 48 genomes as DNA strips, ranked, elites lit, curves joining a child to its
  two parents. The strips are a fixed random projection of the weights, so a child
  visibly resembles its parents instead of looking like fresh noise.
- **CMA-ES** — the sampled brains projected onto the two principal directions of the real
  sample cloud, with its empirical covariance ellipse and the step the mean took.

During **playback** the same panel shows something better than a mock optimiser: the
policy's actual action probabilities this step, for both animals. Not a "mode" label —
the tensor.

Note: the optimiser is **separable (diagonal) CMA-ES**. A full covariance over 2,853
weights is 8.1M numbers with an eigendecomposition per update, and giving CMA-ES a smaller
network than the other two would break the comparison. The ellipse on screen is honest
regardless: it is measured from the real sample cloud.

## Finding the good bits

```bash
.venv/bin/python trainer/scripts/highlights.py --run runs/v4 --episodes 480
```

Plays the champions and scores each episode for drama: a **nail-biter** (she reaches the
hole with him inside two cells), a **heartbreak** (caught within a cell or two of home),
**the snap** (a trap fires and turns the episode), **the long one** (several separate
near-misses), **the shutout**. It writes the winners to `highlights.json`, one per room,
and `h` in the app replays exactly those.

**"Exactly those" is a load-bearing claim, so the scan is the replay.** Every candidate
is played through the same construction `serve.py` uses, keyed on (arena, seed): the
environment stream and the action-sampling stream are both re-seeded per episode, so an
episode is fully described by those two numbers and plays the same alone or in any order.
The reel request carries the seed *and the mouse's school* — the champions are usually
two different schools, and sending only the arena replayed the cat's own pair instead.

That matters because it used to be wrong in a way nobody would notice until the edit. The
scan ran hundreds of episodes in one batch, where a lane's samples come out of a shared
stream at a position no single-episode replay can reach; the app then played a different
episode under the scored one's caption. Two of ten flipped outcome outright — a
"HEARTBREAK · caught 1 cell from home" that plays as an escape.

On the two-hole run it finds 29 distinct dramatic episodes across 4 of the 12 rooms. Only
four, because both PPO policies are saturated enough to play a given room the same way
whatever the seed — which is itself worth saying on camera. A pairing with a stochastic
side has far more: `--mouse ga` gives 99 distinct dramatic episodes across 9 of the 12
rooms, including ones where she dives into the hole on the *same step* he lands on her. Those exist because reaching
the hole is checked before the catch — the rule is in the spec, and it turns out to be
the single best shot in the game.

## Recording

The live stream is journalled to `runs/journals/<stamp>.jsonl` as it goes out, frame by
frame. If a take stalls or a run goes wrong, replay it exactly:

```bash
.venv/bin/python trainer/scripts/serve.py --replay runs/journals/2026-08-27T16-27-49.jsonl
```

The app follows the stream, so a replay drives itself — it walks into the right screen and
plays the recorded episodes at the recorded pace with no trainer running. The same
mechanism means Python can drive the app during a live take.

## Solid walls

The simulation never lets anyone stand on a wall — 518,400 sampled positions, zero
violations. The *screen*, though, does not show simulation state; it shows an
interpolation of it, and both wall bugs lived there:

- **Bodies were tweened between the last two frames.** On the first frame of a new
  episode those are the old death position and the new spawn, 28 to 43 cells apart, so
  the pair slid across the room straight through the blocks. One recorded session held
  198 such frame pairs. The painter now refuses to interpolate a move larger than one
  cell, because that cannot be motion.
- **The vision cone was cast at the destination cell and then translated** to the tweened
  body, hanging an exact shape off the wrong anchor and spilling light through cover —
  up to half a cell deep. It is now cast at the tweened position; `env.js` is in the
  browser and takes fractional coordinates, so this is both exact and smooth. Measured
  penetration fell from 0.498 cells to 0.045, which is the ray-caster's own step
  tolerance.

`tools/check_render.js` replays a journal through the real painter and fails on either.

## About the hardware

You asked for the M2 Max to be leveraged. It is — just not where you would expect, and
the measurement is worth keeping because it is counter-intuitive.

**MPS loses here, and it is not close.** A bare `forward_flat` benchmark says the GPU wins
by 8x. The call training actually makes — upload observations, forward, softmax, sample,
download actions — says the opposite:

| | CPU | MPS |
|---|---:|---:|
| actor round trip, 48 policies x 12 envs | **0.67 ms** | 1.71 ms |
| PPO end to end, 45 s budget | **126k env-steps/s** | 62k |

At 2,853 parameters the run is bound by how many GPU kernels get launched, not by their
arithmetic. `nets.pick_device("auto")` therefore benchmarks the *whole call* and picks CPU;
benchmarking the matmul alone picks wrong. MPS would win back with a network an order of
magnitude larger — but a larger network is exactly what makes evolution unusable, so the
comparison would break.

Where the machine does get used: **the environment is vectorised into lookup tables**
(533k env-steps/s at batch 2048), and **all three schools train at once in separate
processes** — which is both a 3x speedup and the fairness protocol, since they then share
one wall-clock under one load.

## Verification

The environment is the contract, so it is tested rather than trusted.

| gate | what it proves |
|---|---|
| `trainer/scripts/parity.py` | `env.py` reproduces `env.js` exactly — 150 maps at each hole count, identical maps, identical trajectories, rewards to 1e-9 |
| `trainer/scripts/vec_parity.py` | the batched trainer environment matches the reference step for step |
| `trainer/scripts/balance.py` vs `js_balance.js` | the scripted Examiner behaves like the JS original across the whole skill sweep |
| `trainer/scripts/check_arenas.py` | the training, evaluation and tournament rooms are disjoint — the claim the leaderboard rests on |
| `tools/check_render.js` | replays a recorded session through the real painter: no body and no vision cone is ever drawn inside a wall |
| `trainer/scripts/train.py --minutes 3` | end-to-end smoke: three schools, checkpoints, telemetry |

Run the first three after any change to `env.py`, `vec.py` or `scripted.py`. The map
parity gate matters beyond tidiness: the app draws arenas locally from a seed while Python
plays them, and the two only agree because the port is exact.

## Assets

`tools/build_assets.py` rebuilds all eight app assets from the five renders in
`assets-src/`: the backdrops are cover-crops, and the three academy buildings are cut out
of `night.png` with an alpha built from the render itself, bottom-anchored so all three
sit on the same baseline. Swap a render and re-run it.

`tools/vendor_fonts.sh` self-hosts the three fonts if you want the app to need no network
at all during a shoot.
