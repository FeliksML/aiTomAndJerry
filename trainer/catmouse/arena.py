"""Running matches and scoring them.

Everything that needs to answer "how good is this policy" goes through here, so the
three schools, the tournament and the live server all measure the same way.

Two scores matter and they answer different questions:

  head-to-head   this cat against that mouse. Measures the MATCHUP. Useless for
                 tracking one student, because both sides move at once.
  examiner       this policy against the frozen scripted opponent, on fixed arenas.
                 Measures the STUDENT. This is the one that goes on the leaderboard.

Every rate comes back with a Wilson 95% interval, because a 3-point gap over 200
episodes is noise and the scoreboard has to be able to say so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import env as S
from .nets import FlatActor
from .scripted import SKILL_DEFAULT, ScriptedPair
from .vec import MapSet, VecEnv

# Arena sets. The schools train on TRAIN_SEEDS; nothing is scored on them.
TRAIN_SEEDS = [20260826 + i * 911 for i in range(12)]
EVAL_SEEDS = [770001 + i * 6733 for i in range(24)]
FINAL_SEEDS = [313370 + i * 40961 for i in range(8)]

#: How many holes a room has, unless a run says otherwise.
#:
#: Two, not one. With a single hole the cat's best strategy is simply to stand on it —
#: the mouse has no choice to make and the chase collapses into a stakeout. Two holes,
#: kept far enough apart that one cat cannot cover both, turn it back into a game: she
#: picks, he guesses. `--nests` on the trainer changes it, and a level set may mix
#: counts (e.g. `--nests 1,2,3`) so a policy learns to handle any room.
DEFAULT_NESTS = 2


def parse_nests(spec) -> int | list[int]:
    """`2` -> 2 (every room), `1,2,3` -> a repeating mix across the level set."""
    if spec is None:
        return DEFAULT_NESTS
    if isinstance(spec, int):
        return spec
    parts = [int(x) for x in str(spec).split(",") if x.strip()]
    return parts[0] if len(parts) == 1 else parts


def spread(nests, n_seeds: int):
    """Turn a count, or a mix, into one count per seed."""
    if isinstance(nests, int):
        return [nests] * n_seeds
    return [nests[i % len(nests)] for i in range(n_seeds)]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval — behaves at 0 and 1, where the normal one does not."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


@dataclass
class Outcome:
    n: int
    catch: int
    escape: int
    draw: int
    mean_steps: float
    trap_hits: float

    @property
    def catch_rate(self) -> float:
        return self.catch / self.n

    @property
    def escape_rate(self) -> float:
        return self.escape / self.n

    def rate_ci(self, which: str) -> tuple[float, float, float]:
        return wilson(self.catch if which == "catch" else self.escape, self.n)

    def as_dict(self) -> dict:
        p, lo, hi = self.rate_ci("catch")
        q, mlo, mhi = self.rate_ci("escape")
        return {
            "n": self.n, "catch": p, "catchLo": lo, "catchHi": hi,
            "escape": q, "escapeLo": mlo, "escapeHi": mhi,
            "draw": self.draw / self.n, "meanSteps": self.mean_steps,
            "trapHits": self.trap_hits,
        }


def net_agent(actor: FlatActor, rng: np.random.Generator, greedy: bool = False):
    return lambda e, obs: actor.act(obs, rng, greedy=greedy)


def examiner_agent(env: VecEnv, role: str, skill: float = SKILL_DEFAULT, seed: int = 0):
    bot = ScriptedPair(env, skill, seed=seed)
    fn = (lambda e, obs: bot.cat_act()) if role == "cat" else (lambda e, obs: bot.mouse_act())
    fn.bot = bot  # type: ignore[attr-defined]
    return fn


def play(maps: MapSet, cat_fn, mouse_fn, n: int, seed: int = 0,
         map_idx=None, env: VecEnv | None = None) -> Outcome:
    """One block of n episodes, all stepped together, until every one has ended."""
    e = env if env is not None else VecEnv(maps, n, seed=seed)
    e.reset(map_idx=np.arange(n) % len(maps) if map_idx is None else map_idx)
    for _ in range(S.MAX_STEPS + 1):
        if e.done.all():
            break
        oc = e.observe("cat")
        om = e.observe("mouse")
        e.step(cat_fn(e, oc), mouse_fn(e, om))
    r = e.result
    return Outcome(
        n=n, catch=int((r == 1).sum()), escape=int((r == 2).sum()), draw=int((r == 3).sum()),
        mean_steps=float(e.step_n.mean()), trap_hits=float(e.trap_hits.mean()),
    )


def examiner_score(maps: MapSet, flat: np.ndarray, role: str, device, seed: int = 0,
                   reps: int = 8, skill: float = SKILL_DEFAULT) -> Outcome:
    """The leaderboard number: this policy against the frozen Examiner.

    Every arena is played `reps` times, so the sample is balanced across arenas rather
    than at the mercy of which ones a random draw happened to pick.
    """
    n = len(maps) * reps
    e = VecEnv(maps, n, seed=seed)
    rng = np.random.default_rng(seed + 991)
    actor = FlatActor(flat, device)
    if role == "cat":
        return play(maps, net_agent(actor, rng), examiner_agent(e, "mouse", skill, seed + 3),
                    n, seed, map_idx=np.arange(n) % len(maps), env=e)
    return play(maps, examiner_agent(e, "cat", skill, seed + 4), net_agent(actor, rng),
                n, seed, map_idx=np.arange(n) % len(maps), env=e)


def head_to_head(maps: MapSet, cat_flat: np.ndarray, mouse_flat: np.ndarray, device,
                 seed: int = 0, reps: int = 8) -> Outcome:
    n = len(maps) * reps
    rng_c = np.random.default_rng(seed + 1)
    rng_m = np.random.default_rng(seed + 2)
    ac = FlatActor(cat_flat, device)
    am = FlatActor(mouse_flat, device)
    return play(maps, net_agent(ac, rng_c), net_agent(am, rng_m), n, seed,
                map_idx=np.arange(n) % len(maps))


class HallOfFame:
    """Past selves, kept so self-play cannot forget how to beat an old strategy.

    Naive self-play chases the current opponent and cycles: the cat learns to counter
    this week's mouse, the mouse counters that, and neither ends up good in general.
    Mixing frozen snapshots into the opponent pool is the cheap, standard fix.
    """

    def __init__(self, cap: int = 8):
        self.cap = cap
        self.items: list[np.ndarray] = []

    def add(self, flat: np.ndarray) -> None:
        self.items.append(np.array(flat, np.float32, copy=True))
        if len(self.items) > self.cap:
            # Drop the second-oldest: keep the very first (a genuinely different,
            # near-random opponent) and the recent tail.
            del self.items[1]

    def sample(self, k: int, rng: np.random.Generator) -> list[np.ndarray]:
        if not self.items:
            return []
        idx = rng.integers(0, len(self.items), k)
        return [self.items[i] for i in idx]

    def __len__(self) -> int:
        return len(self.items)
