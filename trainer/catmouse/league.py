"""Who a learner plays against, and why it is not only itself.

Pure self-play from a random start does not climb here. Measured over 432 CMA-ES
generations: the cat reached a 74% win rate against its own archive while scoring 8%
against a fixed opponent, and the mouse reached 45% against the archive while scoring
0%. Both sides were only ever required to beat *each other*, so they settled into a
low-level arms race — relative skill oscillates, absolute skill goes nowhere. That is
the classic coevolution failure, and thirty minutes of budget is nowhere near enough
to grind out of it the way a self-play system with a huge archive eventually does.

So every school trains against a league, not a mirror:

    half  the opponent slots  a past self, sampled across the whole archive
    half  the opponent slots  the scripted controller at a ladder skill

The scripted half supplies what self-play cannot: a *stationary* opponent whose
difficulty does not sink to meet you. This is the same thing league training does
with built-in bots, and — crucially — all three schools get exactly the same deal.

**The Examiner's difficulty, 0.60, is deliberately absent from the ladder.** Training
sees 0.15, 0.30, 0.45, 0.75 and 0.90; scoring happens at 0.60. That keeps the progress
curve honest about interpolating to an unseen difficulty rather than replaying a
memorised one. It is still the same opponent *family*, though, so the headline ranking
is the cross-play tournament between schools, where every matchup is genuinely unseen.
"""

from __future__ import annotations

import numpy as np

EXAMINER_SKILL = 0.60

# A widening curriculum. Each row is the five difficulties on offer during that third
# of the run; a candidate meets all five inside a single generation.
#
# The window matters. Facing a skill-0.90 opponent from a random start is not a lesson,
# it is a wall: every candidate loses, the ranking is noise, and nothing is learned
# from those episodes. Starting narrow and easy gives the ranking something to grip.
PHASES = (
    (0.05, 0.15, 0.25, 0.35, 0.45),
    (0.15, 0.30, 0.45, 0.55, 0.70),
    (0.25, 0.45, 0.70, 0.80, 0.90),
)
assert all(EXAMINER_SKILL not in row for row in PHASES), \
    "the scored difficulty must never be a trained one"

_PHASE_ARR = np.asarray(PHASES, np.float64)


N_PHASES = _PHASE_ARR.shape[0]
PROMOTE_WINDOW = 15      # generations of evidence before a promotion
PROMOTE_FRACTION = 0.60  # of what the scripted controller itself manages, see below
_CALIBRATION: dict = {}


def reference_rate(maps, role: str, phase: int, reps: int = 40, seed: int = 3) -> float:
    """What the scripted controller ITSELF scores against this year's ladder.

    A flat "promote at 40%" is not one bar, it is two. Catching is harder than escaping
    in this environment: measured against the year-1 ladder, the scripted controller at
    that year's top skill catches 58% but escapes 69%. A single number therefore asks
    more of the cat than of the mouse, and the first 45-minute run showed exactly that —
    both population schools promoted their mouse to year three and left their cat in
    year one for the entire run.

    So the bar is expressed as a fraction of what the scripted controller manages on the
    same ladder. Same rule for every school and both roles, and no magic constant that
    happens to suit one of them.
    """
    key = (id(maps), role, phase)
    if key in _CALIBRATION:
        return _CALIBRATION[key]
    from . import env as S
    from .scripted import ScriptedPair
    from .vec import VecEnv

    n = len(maps) * reps
    e = VecEnv(maps, n, seed=seed)
    e.noise_tile = reps
    slot = np.arange(n) % reps
    e.reset(map_idx=slot % len(maps))
    learner = ScriptedPair(e, max(PHASES[phase]), seed=seed + 1)
    opponent = ScriptedPair(e, ladder_for(slot, phase), seed=seed + 2)
    for _ in range(S.MAX_STEPS + 1):
        if e.done.all():
            break
        if role == "cat":
            e.step(learner.cat_act(), opponent.mouse_act())
        else:
            e.step(opponent.cat_act(), learner.mouse_act())
    want = 1 if role == "cat" else 2
    rate = float((e.result == want).mean())
    _CALIBRATION[key] = rate
    return rate


def ladder_for(slot: np.ndarray, phase: int = N_PHASES - 1) -> np.ndarray:
    """Spread this year's five difficulties across the episode slots.

    Every candidate is scored against easy, medium and hard opponents *within* one
    generation, so its fitness averages over a fixed ladder rather than over whatever
    single difficulty the generation happened to draw. Cycling one skill per generation
    instead moves the objective under the optimiser, and the measured consequence was a
    learning curve that did not rise for 250 generations.
    """
    p = int(np.clip(phase, 0, N_PHASES - 1))
    return _PHASE_ARR[p][slot % _PHASE_ARR.shape[1]]


class Promotion:
    """Move up a year when the school has *earned* it, not when the clock says so.

    A fixed schedule punishes whichever algorithm happens to be slower: the ladder
    hardens on time whether or not anyone can climb it, and the last third of the
    budget is then spent losing every episode, which teaches nothing. Promoting on a
    sustained win rate against the current year lets each school move at its own pace —
    the same rule for all three, so it stays fair — and how fast a school gets promoted
    is itself worth putting on screen.
    """

    def __init__(self, bars: list[float] | None = None, window: int = PROMOTE_WINDOW):
        self.phase = 0
        #: One bar per year, each a fraction of what the scripted controller itself
        #: scores on that year's ladder. Falls back to a flat 40% only if uncalibrated.
        self.bars = bars or [0.40] * N_PHASES
        self.window = window
        self.hist: list[float] = []
        self.promoted_at: list[int] = []

    @property
    def threshold(self) -> float:
        return self.bars[min(self.phase, len(self.bars) - 1)]

    def update(self, win_rate: float, step: int) -> bool:
        self.hist.append(float(win_rate))
        if len(self.hist) < self.window or self.phase >= N_PHASES - 1:
            self.hist = self.hist[-self.window:]
            return False
        if float(np.mean(self.hist[-self.window:])) >= self.threshold:
            self.phase += 1
            self.promoted_at.append(step)
            self.hist.clear()
            return True
        self.hist = self.hist[-self.window:]
        return False


def scripted_share(frac: float) -> float:
    """How much of the opponent mix is the stationary scripted ladder. Held high.

    The first version of this ramped *down* — mostly scripted early, mostly self-play
    late — on the theory that self-play takes over once the basics are in. Measured, it
    did the opposite of what was wanted: by the last third the cat was scoring 54%
    against a panel that was three-quarters its own weak mouse, while still scoring 6%
    against a fixed opponent. Self-play only sharpens a policy when BOTH sides are
    already strong; before that it is a mirror telling you what you want to hear.

    So the ladder stays three-quarters of the panel and the *curriculum* supplies the
    difficulty ramp instead. The remaining quarter is self-play, which is where the
    behaviours a scripted bot cannot teach come from — ambushes, feints, waiting out a
    camper. Same schedule for all three schools, so it cannot advantage one of them.
    """
    return 0.75


def strided_archive(items: list[np.ndarray], k: int) -> list[np.ndarray]:
    """k opponents spanning the archive from oldest to newest.

    Uniform random sampling of a short archive keeps handing back neighbours from the
    same era. Striding guarantees the panel contains something ancient, something
    recent, and something in between, which is what stops the population from
    specialising against one opponent generation.
    """
    if not items:
        return []
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in idx]


def calibrated_bars(maps, role: str) -> list[float]:
    """The promotion bar for each year, for one role, on these arenas."""
    return [round(PROMOTE_FRACTION * reference_rate(maps, role, p), 4)
            for p in range(N_PHASES)]
