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
.venv/bin/python trainer/scripts/train.py --minutes 45 --tag v2

# decide the championship on arenas nobody trained on
.venv/bin/python trainer/scripts/tournament_run.py --run runs/v2

# serve the app
.venv/bin/python trainer/scripts/serve.py --run runs/v2 &
python3 -m http.server 8778 --directory app
```

Then open <http://localhost:8778>. **[SHOOT.md](SHOOT.md) is the run of show** — a
suggested recording order tied to the keys, with the data beats worth pointing at.

**Keys.** `1 2 3` schools · `x` side by side · `g` level generator · `f` grand final · `b` leaderboard ·
`l` full-screen lesson · `v` verdict · `esc` menu · `space` pause · `s` skip an episode · `[` `]` speed ·
`h` highlight reel · `t` train live on camera · `?` key card · **`r` reveal the next school**, `shift+R` re-seal one, `shift+0` re-seal all.

## The reveal

Your intro names only the first school. Press `r` on camera and the next one opens — its
name, emblem, accent colour, building and explainer all appear at once. Until then it
shows a CLASSIFIED plate, a stable codename and REDACTED specs, so it reads as
deliberately sealed rather than broken, and nothing needs blurring in the edit.

The level survives a reload (it is in `localStorage`), so a crash mid-shoot does not
unseal the rest of the video. Every screen reads a school's identity through
`Reveal.view()`, so a new screen cannot leak one by forgetting to check.

## Side by side

`x` runs all three schools on the **same room at the same time**: same map, same spawns,
the same hearing-noise stream, and the same action-sampling draws shared across all three
panes. Every difference on screen is therefore the policy and nothing else — which is what
the spec means by "the comparison is the whole point". Underneath, a grid of the same
twelve rooms: read down a column and you can see whether a room is hard or whether one
school simply solved it.

## Fairness — the part that makes this an experiment

Self-play alone proves nothing across schools. A GA mouse escaping 80% of the time may
just mean the GA cat is bad. This run shows exactly that: **CMA-ES's cat catches 71% of
its own mouse and 11% of the other two.**

So the championship is decided by **cross-play**: every cat plays every mouse, on eight
arenas that appear in no training set, and a cat's score is its mean catch rate against
the whole field. Alongside it sits a **fixed anchor** — the scripted controller at skill
0.60, a difficulty absent from every school's training ladder — which puts all six
policies on one absolute axis.

Controls that make the comparison mean something:

- **Same brain.** One architecture, 2,533 parameters, one observation vector, for all
  three. PPO additionally trains a critic; it never plays and is discarded at evaluation.
- **Same rooms.** One seeded 12-arena training set shared by all three; separate held-out
  arenas for scoring.
- **Same clock, both clocks.** All three train in parallel for the same wall-clock, and
  every log line carries environment steps too, so "equal compute" and "equal samples"
  can both be read off one run.
- **Same league.** Identical opponent mix and identical promotion rule (below).
- **Error bars.** Wilson 95% intervals on every rate, and the leaderboard says out loud
  when the top two overlap.

## Three things the data forced

These are not design preferences. Each replaced something that measurably did not work.

**1 · Pure self-play does not climb.** Over 432 CMA-ES generations the cat reached a 74%
win rate against its own archive while scoring 8% against a fixed opponent; the mouse
reached 45% and scored 0%. Both sides only ever had to beat *each other*, so they settled
into a low-level arms race. Training now faces a league: three quarters a curriculum of
scripted opponents at five difficulties, one quarter self-play against a hall of fame.

**2 · A fixed curriculum punishes the slower learner.** Hardening the ladder on a clock
means the last third of the budget is spent losing every episode. A school is now
**promoted a year** when it sustains a 40% win rate against its current year — and only
against the scripted slots, so it cannot promote itself by beating its own weak opponent.
How fast each school gets promoted is itself worth putting on screen.

**3 · The mouse was better off hiding.** With the spec's shaping coefficients, a timeout
scores 0 and being caught scores −1, while running the whole way home earns only +0.25 of
shaping. The mouse duly learned to hide behind cover until the step limit and never
escaped once in 400 generations. Training now uses potential-based shaping (Ng, Harada &
Russell), which cannot invent a strategy that was not already optimal — it only makes the
good one findable. The measured effect: GA's mouse went from a flat 0% to a 34% ladder win
rate in 157 generations, with episodes shortening from 164 steps to 118 as it stopped
hiding and started running. **Scoring is untouched** — catch, escape and timeout rates are
counted exactly as the spec defines them.

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

Note: the optimiser is **separable (diagonal) CMA-ES**. A full covariance over 2,533
weights is 6.4M numbers with an eigendecomposition per update, and giving CMA-ES a smaller
network than the other two would break the comparison. The ellipse on screen is honest
regardless: it is measured from the real sample cloud.

## Finding the good bits

```bash
.venv/bin/python trainer/scripts/highlights.py --run runs/v2 --episodes 400
```

Plays the champions across hundreds of episodes and scores each for drama: a **nail-biter**
(she reaches the hole with him inside two cells), a **heartbreak** (caught within a cell or
two of home), **the snap** (a trap fires and turns the episode), **the long one** (several
separate near-misses), **the shutout**. It writes the winners to `highlights.json`, one per
room, and `h` in the app replays exactly those — the environment is deterministic given
(arena, seed), so what you watch is the episode that was scored.

On the first run it found 102 dramatic episodes in 240, including sixteen where she dives
into the hole on the *same step* he lands on her. Those exist because reaching the hole is
checked before the catch — the rule is in the spec, and it turns out to be the single best
shot in the game.

## Recording

The live stream is journalled to `runs/journals/<stamp>.jsonl` as it goes out, frame by
frame. If a take stalls or a run goes wrong, replay it exactly:

```bash
.venv/bin/python trainer/scripts/serve.py --replay runs/journals/2026-08-27T16-27-49.jsonl
```

The app follows the stream, so a replay drives itself — it walks into the right screen and
plays the recorded episodes at the recorded pace with no trainer running. The same
mechanism means Python can drive the app during a live take.

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

At 2,533 parameters the run is bound by how many GPU kernels get launched, not by their
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
| `trainer/scripts/parity.py` | `env.py` reproduces `env.js` exactly — 560 seeds, identical maps, identical trajectories, rewards to 1e-9 |
| `trainer/scripts/vec_parity.py` | the batched trainer environment matches the reference step for step |
| `trainer/scripts/balance.py` vs `js_balance.js` | the scripted Examiner behaves like the JS original across the whole skill sweep |
| `trainer/scripts/check_arenas.py` | the training, evaluation and tournament rooms are disjoint — the claim the leaderboard rests on |
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
