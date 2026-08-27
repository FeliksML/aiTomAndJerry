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
viz/       the imported Claude Design project — reference art + the original Academy flow
app/       the recording app (plain HTML/JS, no build step, no CDN at record time)
trainer/   Python: environment, three algorithms, tournament, WebSocket server
runs/      checkpoints, journals, tournament results (git-ignored)
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
- [ ] **3b · Policy network.** One architecture, shared by all three algorithms.
- [ ] **4 · Three algorithms.** PPO (clipped, GAE), GA (population 48, elitism,
      mutation), sep-CMA-ES (λ=16, σ-adaptation). Each emits training telemetry shaped
      for its own on-screen visualisation.
- [ ] **5 · Tournament.** Cross-play grid + anchor, confidence intervals, champion picks.
- [ ] **6 · WebSocket server.** `PLAY` / `TRAIN` modes over the frame protocol in
      `env-spec.json`, plus an algorithm-internals channel. Journals every frame.
- [ ] **7 · App.** Fork the Academy flow to `app/`, wire the socket, add the three
      explainer visualisations and the reveal layer.
- [ ] **8 · Shoot support.** Highlight detection (near-misses, trap snaps), slow-motion
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

## Open items

- None blocking. Assets resolved: `tools/build_assets.py` rebuilds all eight from the
  five source renders in `assets-src/`.


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
