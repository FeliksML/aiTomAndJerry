# AI Tom & Jerry — build plan

A YouTube segment. Three schools (PPO, GA, CMA-ES) each train a cat and a mouse on the
same arenas; the best Tom meets the best Jerry in a grand final. Everything on screen is
real: real policies, real training, real numbers.

## Decisions locked

| question | answer |
|---|---|
| Who is champion | Cross-play round-robin (3 cats × 3 mice, held-out maps) **plus** a fixed scripted anchor every policy plays |
| What drives the pixels while recording | Live Python → WebSocket stream, **two modes**: `PLAY` (a saved checkpoint playing) and `TRAIN` (training happening on camera). Every stream is journalled to disk so any take can be replayed frame-identical |
| Training budget | ~30–60 min per school (3 schools × 2 roles), ≈2–3 h total |
| App shell | Fork out of `.dc.html` into a plain HTML + JS app under `app/` |
| Reveal | Keyboard, one step per press; hides school **names, emblems, accent colours, building art and explainer visuals** until pressed; survives a reload |

## Repository

```
viz/               the imported Claude Design project — reference art + the original Academy flow
app/               the recording app (plain HTML/JS, no build step, no CDN at record time)
trainer/           Python: environment, three algorithms, tournament, WebSocket server
tools/             asset pipelines (backdrops and buildings; the walking cat)
source-animation/  the five generated Tom clips, immutable source material
runs/              checkpoints, journals, tournament results (git-ignored)
build/             frame extraction, alpha caches, QA renders (git-ignored)
```

## Fairness protocol

The thing that makes this a real experiment rather than three demos.

Self-play alone proves nothing across schools: a GA mouse escaping 80% of the time may
simply mean the GA cat is bad. So:

1. **Identical inputs.** All six policies see the same observation vector and use the
   same network architecture and parameter count. No algorithm gets a bigger brain.
2. **Identical arenas.** One seeded 12-arena training set shared by all three schools,
   and a separate held-out evaluation set nobody trained on.
3. **Two budgets, both reported.** Equal *environment steps* (the sample-efficiency
   currency, which flatters PPO) and equal *wall-clock* (the compute currency, which
   flatters the population methods). Showing both is more honest than picking one — and
   it is a better story.
4. **Cross-play.** Every cat plays every mouse on the same held-out seeds.
   `cat score = mean catch rate against all three mice`; mouse score symmetric.
5. **A fixed anchor.** The scripted greedy controller from `agents.js`, frozen at one
   competence level, plays all six policies. It never learns, so it is the yardstick
   that makes numbers comparable even if every school is weak.
6. **Confidence.** Enough evaluation seeds to report a 95% interval, and the interval
   goes on screen. A 2-point gap on 20 episodes is noise, and saying so is the honest move.

## Phases

- [x] **0 · Import.** Design project pulled into `viz/`; runs standalone off a local server.
- [x] **1 · Environment port.** `trainer/catmouse/env.py`, 1:1 from `env.js`.
      Gate: `scripts/parity.py` — 560 seeds, bit-identical maps and step-for-step
      identical trajectories, rewards to 1e-9.
- [x] **2 · Vectorised environment.** Each arena compiles once into lookup tables
      (all-pairs BFS, per-cell/facing ray casts, line-of-sight, greedy-step-home) so a
      step is pure array indexing. **533k env-steps/s at batch 2048**, both observations
      included — 20M steps of training costs ~40 s of environment time.
      Gate: `scripts/vec_parity.py` — identical states, rewards and outcomes vs `env.py`.
- [x] **3a · The Examiner.** `agents.js`'s greedy controller, vectorised and frozen at
      skill 0.60, as the fixed yardstick. Gate: `scripts/balance.py` vs
      `scripts/js_balance.js` — the outcome mix tracks the JS original across the whole
      skill sweep.
- [x] **3b · Policy network.** One architecture, shared by all three algorithms.
- [x] **4 · Three algorithms.** PPO (clipped, GAE), GA (population 48, elitism,
      mutation), sep-CMA-ES (λ=16, σ-adaptation). Each emits training telemetry shaped
      for its own on-screen visualisation.
- [x] **5 · Tournament.** Cross-play grid + anchor, confidence intervals, champion picks.
- [x] **6 · WebSocket server.** `PLAY` / `TRAIN` modes over the frame protocol in
      `env-spec.json`, plus an algorithm-internals channel. Journals every frame.
- [x] **7 · App.** Fork the Academy flow to `app/`, wire the socket, add the three
      explainer visualisations and the reveal layer.
- [x] **8 · Shoot support.** Highlight detection (near-misses, trap snaps), slow-motion
      replay, deterministic re-shoot from a journal.

## Algorithm visualisations — what each one shows

Real data, not decoration. Each panel is fed from the training telemetry of its own run.

- **PPO** — the five action probabilities as bars, morphing before → after each update;
  the importance ratio plotted against the clip band at 1±ε, with clipped samples
  greying out; entropy and value loss as sparklines. The point the viewer should get:
  *small, safe steps.*
- **GA** — 48 genomes as a grid of DNA strips coloured by weight sign; fitness bars;
  elites glowing; crossover drawn as lines between two parents and a child; mutations
  as sparks. The point: *no gradients, just breeding.*
- **CMA-ES** — the λ=16 samples each generation projected onto the two principal
  components of the actual sample cloud, with the empirical covariance ellipse and the
  mean's step drawn on top. The ellipse really does stretch along the successful
  direction and contract as σ falls. The point: *the search learns its own shape.*

Note: full-covariance CMA-ES is O(n²) in the parameter count and the shared network has
far more weights than that allows, so the optimiser is **separable (diagonal) CMA-ES**.
The ellipse is drawn from the real sample cloud's empirical covariance, which is a true
picture of the search distribution — but the caption should say "diagonal" rather than
imply a full matrix.

## Where the plan changed

Three parts of the original design did not survive contact with the data. Each is
documented with its measurement in `README.md`:

1. **Pure self-play was replaced by a league.** Coevolution from a random start produced
   a cat winning 74% at home and 8% away. Training now faces a curriculum of scripted
   opponents (three quarters) plus self-play against a hall of fame (one quarter).
2. **The fixed curriculum was replaced by earned promotion.** A clock-driven ladder
   punishes whichever algorithm is slower; a school now moves up a year on a sustained
   40% win rate against its current year, measured only against the ladder.
3. **The shaping coefficients were strengthened.** The spec's numbers make hiding until
   the step limit strictly better than running home, and the mouse learned exactly that.
   Potential-based shaping fixes it without changing which strategy is optimal. Scoring
   is unchanged.

Two additions the brief implied but the original plan did not list: a **side-by-side**
screen (all three schools, one room, nothing else different) and **highlight detection**
for the fun moments.

## The sprites

Tom and Jerry are each drawn twice. The vector pair in `paint.js` is still there and
still authoritative for identity; the second skin is a set of sprite sheets. Each sheet
has the same shape — four phases across, eight directions down — and all of them are
built ahead of time, so the app only ever loads transparent PNGs.

Four sheets: a walk and a trapped animation per character.

| sheet | frame | source |
|---|---|---|
| `tom-walk` | 256 | 5 generated clips |
| `jerry-walk` | 256 | 5 generated strips |
| `tom-trapped` | 448 | 5 generated strips |
| `jerry-trapped` | 352 | 5 generated strips |

The frames are different sizes on purpose. A trapped pose throws both arms up and
reaches 185px either side of the trap, which does not fit in the walk sheet's frame; the
metadata carries `charHeight` so the renderer sizes by the *character* rather than by the
frame, and all four sheets therefore draw the same 204px character.

There are two scripts because there are two kinds of source material:

- **From video.** `tools/build_cat_sprites.py` has to *find* the four walk phases inside
  continuous footage, which is most of its length. Only Tom's walk came this way.
- **From strips.** `tools/build_strip_sprites.py` takes five images per animation, each
  already holding the four phases in a row. No gait to detect; its one hard problem is
  cutting the strip in the right places, which it does on the empty columns between the
  poses rather than on even quarters — and splitting the widest run at its thinnest
  column when a thrown-out arm closes the gap between two frames.

Drawing all four phases in one image is the point, not a convenience: within a single
generation the model holds proportions, colour and style together. That is the same
property continuous video gave Tom, and the reason neither character was assembled from
separately generated stills.

Everything downstream is shared — `tools/catlib/` does the matting, the alignment, the
mirroring, the packing and the QA for all of them.

**What a frame is anchored on depends on what is standing still.** A walk anchors on the
torso and lets the feet move. Being caught in a trap is the other way round: the trap is
the one rigid thing in the shot and the character strains around it, so those sheets
anchor on the floor — and not on the centroid of everything touching it, which the cat's
free paw drags 15px back and forth, but on the offset that best lines each frame's floor
profile up with the first frame's. Measured on the trap's own wooden base, that holds it
to within 1.4px across all 64 trapped frames.

Three things the video pipeline had to be taught, each of which quietly ruins the result
if missed:

- **The clips are 8 fps inside a 24 fps container.** Every pose is held for three frames.
  Measured on the held sequence, self-similarity reports a stride of three.
- **The background is not "white".** The character's own cream sits ~40 units off white
  and the contact shadow is *darker* than it, so the matte cuts on the ink line and on
  the drawn gradient step, not on a whiteness threshold.
- **Two of the five clips stand still before setting off, and one dollies in.** Neither
  the stride nor the framing can be read from the clip as a whole.

For both characters only five directions were generated. RIGHT, DOWN_RIGHT and UP_RIGHT
are exact horizontal mirrors of their left-facing partners — deliberately, so the two
sides cannot drift.

    npm run analyze:cat-walk   # measure the clips, write cat-walk.config.json
    npm run build:cat-walk     # config -> 32 frames + sheet + metadata + QA
    npm run build:mouse-walk   # strips -> the same, for Jerry
    npm run build:trapped      # both characters, caught in a trap
    npm run build:sprites      # all four

One config per character and animation — `cat-walk`, `mouse-walk`, `cat-trapped`,
`mouse-trapped` — so a pose that reads badly is one integer away from being fixed and the
build stays reproducible. `app/sprite-lab.html` is the development view: either character,
either animation, all eight directions at once, over the backgrounds a matte fails on.

In the arena the sprites are on by default and `c` toggles back to the vector pair. The
accent that the vector cat wears as a collar moves to a ring on the floor, because the
rule that a school is legible from the character has to survive the swap.

The trapped animation is driven by the environment, not by a timer. `CFG.freezeSteps` is
5, and the countdown maps straight onto the four frames — snap, recoil, then the struggle
repeating — so the snap is drawn on the very step the jaw closed and the two can never
drift apart. While a character is held, the vector trap in that cell is not drawn: the
sprite is holding its own.

## Open items

- None blocking.


## Measured, not assumed

The Examiner earns its place: scored against it, competence is **monotone**, which the
head-to-head rate is not.

| opponent skill | cat, catch rate vs Examiner-mouse | mouse, escape rate vs Examiner-cat |
|---:|---:|---:|
| 0.05 | 5.0% | 0.5% |
| 0.35 | 13.1% | 14.4% |
| 0.65 | 41.6% | 67.6% |
| 0.95 | 60.5% | 74.2% |

Self-play head-to-head over the same sweep wanders between 29% and 39% catch with no
trend, because both sides improve together — exactly the trap this project had to avoid.

Trap hits fall from ~6.9 per episode at skill 0.05 to ~0.06 at 0.65. That curve is the
learning story, and it is real rather than scripted.
