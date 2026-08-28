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

---

## 1 · The room, before anyone learns anything

**Screen:** menu → `g` (level generator)

Twelve arenas are born one at a time, on camera. Talk over the four rules on the right
while they appear — they are the reason the game is not trivially won by either side:

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

**Screen:** `1` → `l` (the lesson) → `l` again (back to the arena)

Open on the arena so the audience sees the game before the algorithm. Then `l` for the
full-screen lesson: five action-probability bars, and the importance-ratio histogram
against the clip band. The one line worth saying out loud: *the update tried to move,
and it was not allowed to* — you can see the whole histogram piled up inside the band.

Back on the arena, the right-hand panel shows **what each brain is about to do** — the
network's actual five probabilities this step, for both animals, with what each of them
can currently sense. That is the tensor, not a label.

**Checkpoints.** Click UNTRAINED, then HALF-TRAINED, then TRAINED. Same twelve rooms,
same rules, three different brains. This is the clearest "before and after" in the video.

**Verdict:** `v`. Two things to point at:

- the **trap curve**. Nothing tells either of them a trap is dangerous; stepping on one
  costs five frozen steps and that is the only lesson available. Watch the count fall.
- the **checkpoint bars**, which are scored against a fixed opponent. Say why that
  matters: the head-to-head above cannot measure a student, because both sides moved.

## 3 · The reveal — school two

**Press `r`.** Live, on camera. GA opens: its name, its emblem, its green, its building,
its explainer. Until that keypress it was a CLASSIFIED plate and REDACTED specs.

`2` → `l`. Forty-eight brains on screen at once, ranked, elites lit, curves joining two
parents to the child they just made. The line: *nothing here knows which direction is
better. It plays all forty-eight, keeps the winners, and breeds.*

## 4 · The reveal — school three

**Press `r` again.** CMA-ES opens.

`3` → `l`. The sample cloud and its ellipse. Say what the ellipse is: the real spread of
this generation's thirty-two brains, projected onto its own two principal directions.
Watch it stretch along whatever keeps working. Worth being straight about: this is
*separable* CMA-ES — a full covariance over 2,853 weights is 8.1M numbers, and giving
CMA-ES a smaller network than the others would have broken the comparison.

## 5 · Training, live

**Press `t`** on any school. The arena keeps playing — with the policy *as it currently
stands*, re-read at the start of every episode — while the panel shows the optimiser
working. When a school earns a promotion a banner fires: **PROMOTED TO YEAR 2**.

This is the moment to explain the league honestly, because it is the most interesting
thing in the build: pure self-play did not work. Measured over 432 generations, the cat
reached a 74% win rate against its own archive and 8% against a fixed opponent. Both
sides only ever had to beat *each other*. So they train against a curriculum too, and a
school moves up a year only when it earns it.

## 6 · Side by side

**Press `x`.** All three schools, the same room, at the same time — same map, same
spawns, the same hearing noise, the same sampling draws. Every difference on screen is
the brain and nothing else.

Underneath, the grid of the same twelve rooms. Read down a column: was that room hard, or
did one school simply solve it?

## 7 · The fun bit

**Press `h`.** The highlight reel. The scanner played the champions across 480 episodes
and scored every one for drama; this plays the best, one per room, and names each moment
on screen as it runs.

Save the **nail-biters** for last. Those are the ones where she dives into the hole on
the *same step* he lands on her — legal because reaching the hole is checked before the
catch, which is a rule in the spec and turns out to be the best shot in the game.

## 8 · The leaderboard

**Press `b`.** This is where the fairness argument pays off, and it has three beats:

1. **The diagonal is the lie.** A school playing itself is the number that flatters
   whoever raised a weak sparring partner. Point at a school whose home number is far
   above its away numbers.
2. **The champion's own mouse is its hardest opponent.** PPO's cat eats the other two
   schools' mice and struggles against its own — because its own is the best in the field.
3. **Beating the benchmark is not the same as being good.** Compare a school's
   `vs EXAMINER` column against its cross-play SCORE. The scripted Examiner is much
   weaker than the learned cats.

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
| `x` | side by side — all three, same room |
| `h` | the highlight reel |
| `g` | the level generator |
| `v` | this school's verdict |
| `b` | the leaderboard |
| `f` | the grand final |
| `t` | train live, on camera |
| `space` | pause / resume |
| `s` | skip this episode |
| `[` `]` | slower / faster |
| `?` | the key card |
| **`r`** | **reveal the next school** |
| `shift+R` | re-seal one (for a re-shoot) |
| `shift+0` | re-seal everything, back to PPO only |
