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
./run.sh serve
```

That is the only command you need. Open <http://localhost:8778>, pick a school, and
press **`n`** — its academy screen sets the budget, the shaping and the algorithm's own
knobs, trains it while you watch, and lets you drag back through the run to any set of
weights it had. **[SHOOT.md](SHOOT.md) is the run of
show** — a suggested recording order tied to the keys.

The same three things from a terminal, if you prefer one:

```bash
.venv/bin/python trainer/scripts/train.py --steps 500M --tag v4     # or --minutes 45
.venv/bin/python trainer/scripts/tournament_run.py --run runs/v4    # the championship
.venv/bin/python trainer/scripts/serve.py --run runs/v4             # the live link
```

Either route runs the same code and writes the same files — the app launches
`train.py` in three processes exactly as a terminal would, and tails the telemetry each
child already writes.

**Keys.** `1 2 3` schools · `x` side by side · `g` level generator · `f` grand final · `b` leaderboard ·
`l` full-screen lesson · `w` how the algorithm works · `v` verdict · `esc` menu · `space` pause ·
`s` skip an episode · `[` `]` speed ·
`h` highlight reel · **`n` this academy** (budget, shaping, its knobs, train it) ·
`t` train the school on screen · **`p` the best and worst of one live generation** ·
`c` sprite skins · `shift+T` the RUNS screen · `shift+S` end a run early · `?` key card ·
**`r` reveal the next school**, `shift+R` re-seal one, `shift+0` re-seal all.

## Inside an academy · `n`

Each school is trained from its own screen, because that is where you are looking at it.
Press `1`, `2` or `3` for a school, then `n`:

| | |
|---|---|
| **How much work this academy gets** | a slider in millions of steps, presets from 5M to 2B, and an estimate of how long that is on this machine |
| **How it is taught** | the three shaping numbers — how hard Jerry is pulled toward a hole, how hard Tom is pulled when he can see her, and when he cannot |
| **The rules of the game** | +1 for a catch, −1 for letting her home, −0.05 for a trap. Shown, and locked: the scoreboard counts these directly, so a school taught under different ones would not be playing the same sport as the other two |
| **Its own knobs** | whatever its config actually has — PPO's batch, rollout, learning rate, clip and entropy; the GA's population, elites, mutation; CMA-ES's samples, step size, episodes per sample |
| **TRAIN THIS ACADEMY** | trains it live, on its own budget, with a bar, iteration, steps-against-target, rate and ETA — and the arena keeps playing the brain the optimiser holds right now |

The run card and the reel belong to the *run*, not to what the arena happens to be
playing: click UNTRAINED or drag the reel back to an early checkpoint and the card stays
up, still ticking, so "here is what it was, and it is still training behind us" is one
screen rather than two.

Every academy keeps its own settings, and they survive a reload.

## The reel

Every time a school is scored, the weights it was measured on are kept. That turns a run
into something you can drag through: pick a step and the arena is replayed by the brain
the school had *then*. Early frames wander; late ones hunt.

Under the arena, the reel draws Tom's catch rate and Jerry's escape rate across the run
on **one 0–100% axis** — they are both rates, and scaling each to its own maximum (which
is what it did first) made every crossing an artefact — with a legend, the current pair of
numbers, and a dashed mark where each one peaked. Rarely the same place. The handle pins a
frame, `LIVE` goes back to the policy the optimiser holds now, and it all keeps working
while the run carries on behind it: the caption says `PINNED · the brain at 4M steps ·
Tom 12% · Jerry 5% · training continues behind it`.

A reload, a second window, or a socket that drops and comes back all rebuild the same
screen — the greeting carries the reel, which run is training and which frame is pinned. Frames are written to `runs/<tag>/<school>/timeline.npz`, so a
finished run is just as scrubbable as a live one.

## The run screen · `n` from the Academy

What is left at the run level: which run is being watched, the three academies as a
summary, and **SCORE THE RUN** — the cross-play tournament and the highlight scan,
streamed onto the screen, with the leaderboard and grand final live the moment it
finishes. Each school can enter the championship as it finished or at its best.

## How long, and how far in

`--minutes` and `--steps` are the two budget clocks. Give either, or both — with both,
a run ends at whichever arrives first, which is the safe way to start something
overnight. `--steps` takes what you would actually type: `500M`, `1.5B`, `2e8`.

```bash
.venv/bin/python trainer/scripts/train.py --steps 500M --tag v4          # ~1h35m for all three
.venv/bin/python trainer/scripts/train.py --steps 500M --minutes 120     # ... but never past 2h
.venv/bin/python trainer/scripts/train.py --steps 500M --envs 2048       # PPO's batch, ~35% faster
```

The console prints a live line per school — percentage, iteration, steps against the
target, steps per second and an ETA — and the app draws the same four numbers plus a
progress bar while `t` is running. Press `shift+T` to pick the budget before you start
and `shift+S` to end a run early; a stopped run still snapshots, still picks its best
pair and still saves.

**On an M2 Max**, all three schools together sustain about **270k environment steps per
second** (~90k each) at `--envs 2048`, which puts 500M steps each at roughly **1h35m**.
That is about 46% of the twelve CPU cores; the GPU is deliberately idle, because at
2,853 parameters the MPS round-trip costs more than the arithmetic saves — `pick_device`
benchmarks the real call and picks the CPU on purpose. More threads do not help either:
the environment is one single-threaded NumPy loop per process, so the batch width is the
only lever that turns more of the chip into steps.

## The best Tom and the best Jerry

Self-play is not monotone — a cat can be walked backwards by a mouse that got good after
it did — so the policy a run *finishes* on is not always the best one it *reached*. Every
run therefore keeps a high-water mark per role, ranked on the **lower end** of the
confidence interval so a single lucky evaluation cannot claim the title, and at the end
re-scores the peak against the finish on a seed neither was picked on. The winner is
written as a fourth checkpoint:

```
runs/<tag>/<school>/checkpoints.npz    ... plus best_cat / best_mouse
runs/<tag>/<school>/best.json          which one won, both readings, and where the peak was
```

The app gets a **BEST** chip next to UNTRAINED / HALF-TRAINED / TRAINED, and
`tournament_run.py --checkpoint best` enters those policies into the championship
instead of the ones each school happened to end on. A live take saves the same way, into
`runs/<tag>/live/<stamp>/`, so an hour of training on camera is not lost when the server
stops.

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
just mean the GA cat is bad. This run shows exactly that: **CMA-ES's cat catches 50% of
its own mouse and 17% of PPO's** — the same policy, a 33-point spread that depends
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
controller itself manages on the same ladder** — computed in 1.5s at setup, per role, per
year, on the arenas and hole count this run will actually use. On the shipped two-hole
default that is 25% for the cat and 49% for the mouse in year one; on one hole it is 36%
and 38%. The gap between those two pairs is the whole argument: a constant would have been
wrong for at least one of them. Same rule for every school; no number that happens to suit
one role.

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
| **PPO** | **82.3%** ±3.0 | **73.8%** ±3.4 | **41.4%** ±3.8 | **56.6%** ±3.8 |
| GA | 54.1% ±3.8 | 20.8% ±3.1 | 23.9% ±3.3 | 38.4% ±3.8 |
| CMA-ES | 40.6% ±3.8 | 27.3% ±3.4 | 13.6% ±2.7 | 37.2% ±3.7 |

The two hole counts are two different runs, and — worth saying because the table invites
reading across — **two different networks**. The one-hole runs predate the configurable
hole count and carry a 40-number observation, 2,533 weights; the two-hole run carries the
`MAX_NESTS` hole slots and 2,853. Within a column all three schools are identical, which
is the comparison the leaderboard makes. Across columns, read the direction, not the gap.

SCORE here is the mean against the **other two** schools. A school's own mouse is left
out of its cat's score and vice versa, because averaging it back in is precisely the
route by which a weak sparring partner at home flatters a score — the thing this whole
tournament exists to rule out. The diagonal is still on the leaderboard, and the gap
between it and the rest is the interesting part; it simply does not vote.

That exclusion is not cosmetic. With the diagonal counted, the two-hole board crowns GA's
mouse at 47.7% over PPO's 47.2% — on the strength of 66.2% against GA's *own* cat, the
worst cat in the field at 20.8%. Off the diagonal, PPO's mouse leads 56.6% to 38.4% and
the intervals are nowhere near each other. One cell in nine decided the headline.

**PPO wins both roles at both hole counts.** Adding the second hole did not change who
wins, but it changed everything about the margins:

- Every cat got worse and every mouse got better, which is the point of the change — with
  one exit, standing on it was most of the job.
- **GA's mouse gained the most** — 23.9% to 38.4%, from a distant last to level with
  CMA-ES — and on the fixed anchor it actually leads the field (41.6% against PPO's
  38.4%). Breeding whole brains is a good way to learn "run for whichever door he is not
  covering", and the two measures disagreeing is itself the point: the anchor is one
  opponent, cross-play is three.
- **GA's cat collapsed** to 20.8% and never got past year two. The same method is a poor
  way to learn "cover two doors at once".

So the split is not in who wins but in how much: **PPO raises the best hunter and the
best escape artist at both hole counts, but the second hole halves its cat's margin and
doubles GA's mouse.** That fell out of a rules change, not out of tuning.

Three things worth pointing at on camera:

- **The diagonal is the lie, and it is not in the score.** PPO's cat catches 71% and 77%
  of the other schools' mice and only 56% of its own — because its own mouse is the
  hardest opponent its cat ever meets, and the best mouse in the field besides. Read the
  other two rows the other way round: a school that raised a weak sparring partner looks
  dominant at home. That is why SCORE averages the off-diagonal only.
- **CMA-ES's cat catches 50% of its own mouse and 17% of PPO's.** The same policy, a
  33-point spread. Any single matchup would have been a meaningless ranking.
- **Beating the benchmark is not being good.** Compare any school's `vs EXAMINER` column
  against its cross-play score. On the two-hole board the Examiner is the *harder*
  opponent for every cat — PPO's catches 74% of the learned mice and 42% of the scripted
  one — which is the same lesson from the other side: one fixed opponent ranks nobody.

## What the explainers draw

Nothing is a mock-up; every panel is fed from its own optimiser while it runs.

- **PPO** — the five action probabilities on a fixed probe batch, and the importance
  ratio against the clip band. The histogram piles up inside the band: that *is* the
  clipped objective, and you can watch the update not being allowed to move.
- **GA** — the generation as tiles laid out by genome so one can be followed, coloured by
  ancestry, the condemned struck through, a dynasty band underneath, curves joining a child to its
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
  penetration fell from 0.498 cells to 0.045, and then to a flat 0 once the caster
  stopped marching in fixed steps (below).
- **The cone was ray-marched in 0.18 steps and backed off a whole one on contact**, so
  every ray stopped somewhere in the last 0.18 cells before the wall and by a different
  amount each: a flat wall came out as a 4px saw, and an unobstructed ray reached 8.64
  of a nominal 8.5, feeding the policy readings of 1.016 on a [0,1] input. It now walks
  cell boundaries and returns the exact crossing.
- **Twenty-one rays over 100 degrees sample the world every 0.74 cells at full range**,
  so the fan almost never landed on the corner a shadow pivots around and the polygon
  bridged a near hit and a far miss with one straight edge — the spikes. 13% of
  neighbouring ray pairs differed by over a cell, the worst by 7.9. The drawn shape now
  gets a ray a hair either side of every wall corner in view; against the true
  visibility region its error falls from 9.2% of the lit area to 1.2%, and the 21
  readings a policy consumes are untouched.

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
| `trainer/scripts/check_arenas.py` | the training, evaluation and tournament rooms are disjoint, at each hole count — the claim the leaderboard rests on |
| `tournament_run.py`, run twice | the board is reproducible. Pairing seeds derive from a SHA-1 of the two school names, not `hash()`, which Python randomises per process — the same checkpoints used to score differently every time they were re-scored |
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
