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
PROMOTE_AT = 0.40        # win rate against the current year's ladder
PROMOTE_WINDOW = 15      # generations of evidence before a promotion


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

    def __init__(self, threshold: float = PROMOTE_AT, window: int = PROMOTE_WINDOW):
        self.phase = 0
        self.threshold = threshold
        self.window = window
        self.hist: list[float] = []
        self.promoted_at: list[int] = []

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
