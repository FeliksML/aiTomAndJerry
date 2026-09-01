# Run of show

A suggested order for recording, tied to the actual keys. Nothing here is a mock-up —
every screen is reading live from the trainer, so you can stop and poke at any of it.

Before you start:

```bash
./run.sh serve
```

Then open <http://localhost:8778>, press `?` once to check the key card, and confirm the
reveal chip in the top right says **REVEAL 1 / 3**. If it says anything else, press
`shift+0` — you have revealed schools in a previous session and the state persisted.

Two more things worth knowing before you roll:

- **The school screen is taller than the frame.** Once a run has a reel, the left column
  runs past the bottom of the 1920x1080 canvas — the reel's readout line and the whole
  run-control row sit below it. The canvas scrolls vertically; scroll down once when you
  want them in shot. Nothing else on any screen overflows.
- **Two beats need something to exist first.** `v` (verdict) needs a *finished* twelve-room
  playthrough and refuses with a notice until there is one. `h` (highlights) needs a scan
  on disk — `./run.sh score` writes it — and on a run without one the key does nothing at
  all, silently.

---

## 1 · The room, before anyone learns anything

**Screen:** menu → `g` (level generator)

Twelve arenas are born one at a time, on camera — the run's *own* twelve rooms, rebuilt
from their own seeds, not a new set. Talk over the six rules on the right while they
appear; they are the reason the game is not trivially won by either side:

- the room is **one connected space**, seven cover blocks and ten pillars, so there is
  always a way round and never a corridor that decides the episode for you;
- there are **two holes**, at least 10 cells apart so the cat cannot cover both — with
  one hole his best move is just to stand on it, and 27% of episodes then die on the
  clock;
- each hole has **two independent approaches**, so even one of them cannot be sealed;
- three of six traps sit **on her shortest route home**, because a hazard off the walked
  path is never learned and never seen;
- she spawns in the farthest slice of the room from either hole, he spawns at least 10
  cells from her — both get a real chance;
- **all three schools train on these same twelve rooms.** Nobody memorises one room.

## 2 · School one: PPO

**Screen:** `1` → `t` → `l` (the lesson) → `l` again (back to the arena)

Open on the arena so the audience sees the game before the algorithm.

**The lesson panel needs a live optimiser.** It draws its numbers from the running
trainer, so with nothing training it says *no telemetry yet* rather than inventing a
shape. Press `t` first, wait for the caption to turn from "starting the optimiser" to
"training live", then `l`: five action-probability bars, and the importance-ratio
histogram against the clip band. The one line worth saying out loud: *the update tried
to move, and it was not allowed to* — you can see the whole histogram piled up inside
the band.

Back on the arena, the right-hand panel depends on whether *this* school is training.

While it is, that panel is the **optimiser's** — action bars, the clip histogram, entropy.
The live-decision panel — **what each brain is about to do**, the network's actual five
probabilities this step for both animals, with what each can sense — is what the same slot
shows when the school on screen is *not* training. Both are the tensor, not a label; they
answer different questions, so decide which one you want in the shot before you press `t`.

**Checkpoints.** Click UNTRAINED, then HALF-TRAINED, then TRAINED. Same twelve rooms,
same rules, three different brains. This is the clearest "before and after" in the video.

After a reset the pills are a single UNTRAINED — there is nothing else to show — and BEST
only appears when the run wrote one.

**Coming back.** A checkpoint click takes the arena out of the shadow episode, so during a
live run the reel under it greys to READ ONLY. Drag it, or press its LIVE button, and the
arena goes back to the training episode by itself; the reel says so as it happens.

**Verdict:** `v`. Two things to point at:

- the **trap curve**. Nothing tells either of them a trap is dangerous; stepping on one
  costs five frozen steps and that is the only lesson available. Watch the count fall.
- the **checkpoint bars**, which are scored against a fixed opponent. Say why that
  matters: the head-to-head above cannot measure a student, because both sides moved.

## 3 · The reveal — school two

**Press `r`.** Live, on camera. GA opens: its name, its emblem, its green, its building,
its explainer. Until that keypress it was a CLASSIFIED plate and REDACTED specs.

`2` → `l`. The whole generation on screen at once — the panel says how many, and so does
the lesson beside it; both read the population off the run, so whatever you set in the
academy is what they say.

The grid is laid out **by genome, not by rank**: a brain keeps its square from one
generation to the next, so you can follow one. Colour is ancestry — a child is the circular
mean of its two parents' hue — and the ones about to be replaced are struck through. Under
it, a band of who everyone is descended from, one column per generation.

The line: *nothing here knows which direction is better. It plays them all, keeps the
winners, and breeds.* Then let it run for a minute and point at the band: the crowd
collapses to one or two bloodlines while you watch.

## 4 · The reveal — school three

**Press `r` again.** CMA-ES opens.

`3` → `l`. The sample cloud and its ellipse. Say what the ellipse is: the real spread of
this generation's samples — lambda of them, whatever the academy is set to — projected onto
its own two principal directions.

**Do not say "watch it stretch".** It will not. Thirty-two samples in 2,853 dimensions have
near-equal leading eigenvalues, so any two-dimensional projection of this cloud is round —
measured across a whole 45-minute run the ellipse ratio never leaves 1.01–1.02. The
stretching is real but it happens across all 2,853 axes, which is what the CONDITION row
reports; the two drawn axes cannot show it, and the row says so.

What *does* move is the size. The big frame rescales to the cloud every generation to keep
the dots legible, so read the **± ring value** for the scale, and watch the small
**AT GENERATION 1'S SCALE** inset beside the stats — that one never rescales, so the cloud
visibly widens while it is still exploring and then collapses to a knot. That is the beat:
*it stops looking around*.

Worth being straight about: this is *separable* CMA-ES — a full covariance over 2,853
weights is 8.1M numbers, and giving CMA-ES a smaller network than the others would have
broken the comparison.

## 5 · Training, live

**Press `t`** on any school. The arena keeps playing — with the policy *as it currently
stands*, re-read at the start of every episode — while the panel shows the optimiser
working. When a school earns a promotion a banner fires, and it names the side that earned it:
**CAT PROMOTED TO YEAR 2**, or **MOUSE PROMOTED TO YEAR 2**. The two animals are promoted
separately and rarely together, which is worth the sentence.

This is the moment to explain the league honestly, because it is the most interesting
thing in the build: pure self-play did not work. Measured over 432 generations, the cat
reached a 74% win rate against its own archive and 8% against a fixed opponent. Both
sides only ever had to beat *each other*. So they train against a curriculum too, and a
school moves up a year only when it earns it.

## 5b · Who bred and who did not — `p`

**Press `p` while a school is training.** Six rooms become four, or three: the best of the
generation and the ones that were about to be replaced, side by side in identical rooms —
same map, same spawns, same hearing noise, one shared draw of the dice, and the same
opponent in every lane. Every difference on screen is the genome.

It only works while that school's run is live. The brains that lost exist for one
generation and are then written over, so they are kept as they are scored; a finished run
has nothing to put in the losing lanes, and the key says so instead of opening an empty
screen. PPO has no population, so it is a GA and CMA-ES beat.

**On CMA-ES it is a different screen and it is worth the detour.** That school keeps
nobody — the whole sample is discarded every generation — so the title reads THE BEST AND
WORST DRAWS, and one lane is gold: **THE CENTRE**, the mean of the distribution. That is
the brain the arena has been playing all along, and it was never drawn and never scored.

One honest caveat for the narration: a single room is noisy. In testing a rank-119 genome
caught the mouse in one room while rank 1 let it escape. The twelve-room grid under the
panes is the comparison that settles it, not any one pane.

## 6 · Side by side

**Press `x`.** All three schools, the same room, at the same time — same map, same
spawns, the same hearing noise, the same sampling draws. Every difference on screen is
the brain and nothing else.

Underneath, the grid of the same twelve rooms. Read down a column: was that room hard, or
did one school simply solve it?

## 7 · The fun bit

**Press `h`.** The highlight reel. The scanner played the champions over 400 seeded
episodes and scored every one for drama; this plays the best, one per room, and names
each moment on screen as it runs. Every pick was scored on the *same* replay the app is
running — same arena, same seed, same pair of schools — so the caption and the episode
cannot come apart.

On this run that is four rooms out of twelve, and the reason is worth a sentence:
**both PPO policies are decided enough that a room plays out the same way whatever the
seed.** Drama needs someone still guessing. If you want a longer reel, re-run the scan
with `--mouse ga` — the runner-up mouse is stochastic and gives 99 dramatic episodes
across 9 rooms.

**Save the nail-biters for last** — and now you can, because the reel plays them there. The
scan sorts by drama descending, which is the right order for *choosing* the episodes and the
wrong one for *playing* them, so the app reverses it: the reel opens on the quietest of the
picks and builds to the best. Those are the ones where she dives into the hole on
the *same step* he lands on her — legal because reaching the hole is checked before the
catch, which is a rule in the spec and turns out to be the best shot in the game.

## 8 · The leaderboard

**Press `b`.** This is where the fairness argument pays off, and it has three beats:

1. **The diagonal is the lie, which is why it is not in the score.** A school playing
   itself is the number that flatters whoever raised a weak sparring partner. Point at a
   school whose home number is far above its away numbers — CMA-ES catches 50% of its own
   mouse and 17% of PPO's. SCORE is the mean of the other two cells only.
2. **The champion's own mouse is its hardest opponent.** PPO's cat eats the other two
   schools' mice (71% and 77%) and manages only 56% against its own — because its own is
   the best mouse in the field. That cell is on the diagonal, so it is drawn but does not
   count towards SCORE; say that out loud while pointing at it.
3. **Beating the benchmark is not the same as being good.** Compare a school's
   `vs EXAMINER` column against its cross-play SCORE. On the two-hole board the scripted
   Examiner is the *harder* opponent for every cat — PPO takes 74% off the learned mice
   and 42% off the Examiner — which is the same lesson from the other side: one fixed
   opponent ranks nobody.

And the budgets card: the same forty-five minutes buys very different numbers of updates,
and very different numbers of environment steps. Whether PPO "wins" depends on which
clock you believe — both are on screen.

If the top two overlap, the board says **TOO CLOSE TO CALL** rather than crowning
someone. Read that out if it appears; it is more interesting than a fake winner.

## 9 · The grand final

**Press `f`.** Champion cat against champion mouse, five rounds, on an arena neither has
ever seen.

---

## If something goes wrong mid-take

Every frame sent is journalled as it goes out. To replay a take exactly:

```bash
.venv/bin/python trainer/scripts/serve.py --replay runs/journals/<stamp>.jsonl
```

The app follows the stream, so it walks itself back into the right screen and replays at
the recorded pace with no trainer running at all.

## Keys, in one place

| | |
|---|---|
| `1` `2` `3` | enter a school |
| `l` | full-screen lesson (toggle) |
| `w` | how this algorithm works — six manual steps (school screens only; the arena pauses, the optimiser does not) |
| `x` | side by side — all three, same room |
| `h` | the highlight reel |
| `g` | the level generator |
| `v` | this school's verdict |
| `b` | the leaderboard |
| `f` | the grand final |
| `t` | train live, on camera |
| `p` | the best and the worst of one live generation, same room |
| `n` | this school's academy — budget, shaping, knobs (and the RUNS screen from the menu) |
| `shift+T` | the RUNS screen from anywhere |
| `shift+S` | end the live run early, keeping its checkpoints |
| `c` | sprite skins on / off |
| `esc` | back to the academy |
| `space` | pause / resume |
| `s` | skip this episode |
| `[` `]` | slower / faster |
| `?` | the key card |
| **`r`** | **reveal the next school** |
| `shift+R` | re-seal one (for a re-shoot) |
| `shift+0` | re-seal everything, back to PPO only |
