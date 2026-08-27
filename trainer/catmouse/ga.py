"""Genetic algorithm school — no gradients, just breeding.

Forty-eight whole brains per generation. The ones that did best get to be parents;
their children are a coin-flip mix of two parents with a few weights nudged. Nothing
in here knows what a derivative is, which is exactly the point on screen.

  selection   tournament of 3, plus straight elitism so the best genome can never be
              lost to an unlucky draw
  crossover   uniform, per weight
  mutation    a fraction of weights get a Gaussian nudge

The mutation is deliberately *sparse*. Perturbing all 7,109 weights at once with
sigma 0.08 moves the genome by more than half its own length — that is not a mutation,
it is a new random brain, and the population never converges. Nudging a tenth of the
weights keeps a child recognisably its parents' child.

Sigma adapts on Rechenberg's one-fifth rule: if children are beating their parents
more than a fifth of the time the search is too timid and the step grows; if they
rarely beat them it shrinks. That is also the single most watchable number in the
panel, because it visibly hunts.

Telemetry for the explainer: per-genome fitness, who the elites are, which two
parents made each child, how hard each child was mutated, and a stable fingerprint
per genome so a child visibly resembles its parents.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evo import EvoSchool, fingerprint
from .nets import FLAT_DIM, init_flat


@dataclass
class GAConfig:
    pop_size: int = 48
    elite: int = 6
    tournament: int = 3
    mutation_rate: float = 0.10      # fraction of weights nudged in a child
    sigma: float = 0.05
    sigma_min: float = 0.004
    sigma_max: float = 0.25
    fingerprint_dim: int = 40


class GASchool(EvoSchool):
    key = "ga"
    label = "Genetic Algorithm"

    def __init__(self, *a, cfg: GAConfig | None = None, **kw):
        super().__init__(*a, **kw)
        self.cfg = cfg or GAConfig()
        self.pop_size = self.cfg.pop_size

    def setup_optimiser(self) -> None:
        c = self.cfg
        self.pop = {r: init_flat(c.pop_size, self.rng) for r in ("cat", "mouse")}
        for r in ("cat", "mouse"):
            self.pop[r][0] = self.best[r]        # keep the shared starting brain in the pool
        self.sigma = {r: c.sigma for r in ("cat", "mouse")}
        self.parent_fit = {r: None for r in ("cat", "mouse")}
        self.basis = self.rng.normal(0, 1 / np.sqrt(FLAT_DIM),
                                     (FLAT_DIM, c.fingerprint_dim)).astype(np.float32)
        self.lineage = {r: None for r in ("cat", "mouse")}

    def ask(self, role: str) -> np.ndarray:
        return self.pop[role]

    def tell(self, role: str, pop: np.ndarray, fit: np.ndarray) -> dict:
        c, rng = self.cfg, self.rng
        order = np.argsort(-fit)
        elites = order[:c.elite]
        self.best[role] = pop[order[0]].copy()

        # One-fifth rule: how often did last generation's children beat the median
        # of the parents that made them?
        prev = self.parent_fit[role]
        if prev is not None and self.lineage[role] is not None:
            kids = np.arange(c.elite, c.pop_size)
            par = self.lineage[role]
            beat = np.mean([fit[k] > max(prev[par[k - c.elite][0]], prev[par[k - c.elite][1]])
                            for k in kids]) if len(kids) else 0.2
            factor = 1.06 if beat > 0.2 else 0.97
            self.sigma[role] = float(np.clip(self.sigma[role] * factor, c.sigma_min, c.sigma_max))
        else:
            beat = 0.0

        def pick() -> int:
            cand = rng.integers(0, c.pop_size, c.tournament)
            return int(cand[np.argmax(fit[cand])])

        nxt = np.empty_like(pop)
        nxt[:c.elite] = pop[elites]
        pairs, mut_mag = [], []
        for i in range(c.elite, c.pop_size):
            p1, p2 = pick(), pick()
            take = rng.random(FLAT_DIM) < 0.5
            child = np.where(take, pop[p1], pop[p2]).astype(np.float32)
            hit = rng.random(FLAT_DIM) < c.mutation_rate
            noise = rng.normal(0, self.sigma[role], FLAT_DIM).astype(np.float32) * hit
            child += noise
            nxt[i] = child
            pairs.append((p1, p2))
            mut_mag.append(float(np.linalg.norm(noise)))

        self.lineage[role] = pairs
        self.parent_fit[role] = fit.copy()
        self.pop[role] = nxt

        fp = fingerprint(pop, self.basis)
        return {
            "algo": "ga",
            "fitness": [round(float(x), 4) for x in fit],
            "elites": [int(i) for i in elites],
            "order": [int(i) for i in order],
            "sigma": self.sigma[role],
            "childBeatParents": float(beat),
            "pairs": [[int(a), int(b)] for a, b in pairs],
            "mutation": [round(m, 4) for m in mut_mag],
            "fingerprints": np.round(fp, 3).tolist(),
            "diversity": float(np.mean(np.std(pop, axis=0))),
        }
