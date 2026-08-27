"""CMA-ES school — the search learns its own shape.

Each generation draws lambda brains from a Gaussian, keeps the better half, and moves
the Gaussian: its centre shifts toward what worked, its width grows along directions
that keep paying off and shrinks along directions that do not. That adapting shape is
the whole idea, and it is what the on-screen ellipse shows.

**Separable (diagonal) CMA-ES.** The textbook algorithm carries a full n x n
covariance matrix. With 7,109 weights that is 50 million numbers and an
eigendecomposition per update — impossible here, and giving CMA-ES a smaller network
than the other two schools would break the comparison. So the covariance is kept
diagonal (Ros & Hansen, 2008), with the standard correction that multiplies the
learning rates by (n + 2) / 3. The caption should say "diagonal", not imply a full
matrix.

The ellipse on screen is honest anyway: it is the *empirical* covariance of the
lambda sampled brains, projected onto the two principal directions of that same
cloud. Those samples really were drawn from the current search distribution, so the
ellipse really is its shape — just measured from the cloud rather than read off a
matrix that does not exist.

Telemetry: the projected sample cloud with its ranks, the covariance ellipse, the
step the mean took, sigma, and the condition number of the diagonal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evo import EvoSchool
from .nets import FLAT_DIM, init_flat


@dataclass
class CMAConfig:
    lam: int = 32                # population; the classic default here is 4 + 3*ln(n) ~ 30
    sigma0: float = 0.035
    eps_per_genome: int = 12


class CMAESSchool(EvoSchool):
    key = "cmaes"
    label = "Covariance Matrix Adaptation"

    def __init__(self, *a, cfg: CMAConfig | None = None, **kw):
        super().__init__(*a, **kw)
        self.cfg = cfg or CMAConfig()
        self.pop_size = self.cfg.lam
        self.eps_per_genome = self.cfg.eps_per_genome

    def setup_optimiser(self) -> None:
        c = self.cfg
        n = FLAT_DIM
        lam = c.lam
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        self.n, self.lam, self.mu, self.w, self.mueff = n, lam, mu, w, mueff
        self.cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        self.cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        # Separable correction: a diagonal has n free parameters instead of n^2, so it
        # can be learned (n+2)/3 times faster.
        k = (n + 2) / 3
        self.c1, self.cmu = min(1.0, c1 * k), min(1 - min(1.0, c1 * k), cmu * k)
        self.damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

        init = init_flat(2, self.rng)
        self.mean = {"cat": init[0].astype(np.float64), "mouse": init[1].astype(np.float64)}
        self.best = {r: self.mean[r].astype(np.float32).copy() for r in ("cat", "mouse")}
        self.sigma = {r: c.sigma0 for r in ("cat", "mouse")}
        self.C = {r: np.ones(n) for r in ("cat", "mouse")}
        self.ps = {r: np.zeros(n) for r in ("cat", "mouse")}
        self.pc = {r: np.zeros(n) for r in ("cat", "mouse")}
        self.count = {r: 0 for r in ("cat", "mouse")}
        self._z = {}
        self._proj = {}

    def ask(self, role: str) -> np.ndarray:
        """Mirrored sampling: draw lambda/2 directions and use each one twice, plus and
        minus. Fitness here is measured over a handful of noisy episodes, and an
        antithetic pair cancels most of that noise out of the estimated direction — the
        single cheapest thing that makes evolution work on a stochastic game."""
        d = np.sqrt(self.C[role])
        half = self.lam // 2
        zh = self.rng.standard_normal((half, self.n))
        z = np.empty((self.lam, self.n))
        z[0::2] = zh
        z[1::2] = -zh
        self._z[role] = z
        x = self.mean[role][None, :] + self.sigma[role] * z * d[None, :]
        return x.astype(np.float32)

    def tell(self, role: str, pop: np.ndarray, fit: np.ndarray) -> dict:
        n, mu, w = self.n, self.mu, self.w
        d = np.sqrt(self.C[role])
        self.count[role] += 1
        g = self.count[role]

        order = np.argsort(-fit)                       # maximising, so best first
        x = pop[order].astype(np.float64)
        z = self._z[role][order]

        old_mean = self.mean[role].copy()
        zw = w @ z[:mu]                                # weighted step in z-space
        self.mean[role] = old_mean + self.sigma[role] * (zw * d)
        # The distribution mean is the recommendation, not the luckiest sample. On a
        # noisy objective the top sample is mostly the one that drew easy episodes.
        self.best[role] = self.mean[role].astype(np.float32).copy()

        self.ps[role] = (1 - self.cs) * self.ps[role] + \
            np.sqrt(self.cs * (2 - self.cs) * self.mueff) * zw
        ps_norm = float(np.linalg.norm(self.ps[role]))
        hsig = ps_norm / np.sqrt(1 - (1 - self.cs) ** (2 * g)) / self.chiN < 1.4 + 2 / (n + 1)
        self.pc[role] = (1 - self.cc) * self.pc[role] + \
            (hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff)) * (zw * d)

        rank_mu = w @ ((z[:mu] * d[None, :]) ** 2)
        self.C[role] = ((1 - self.c1 - self.cmu) * self.C[role]
                        + self.c1 * (self.pc[role] ** 2
                                     + (not hsig) * self.cc * (2 - self.cc) * self.C[role])
                        + self.cmu * rank_mu)
        self.C[role] = np.maximum(self.C[role], 1e-12)
        self.sigma[role] = float(np.clip(
            self.sigma[role] * np.exp((self.cs / self.damps) * (ps_norm / self.chiN - 1)),
            1e-5, 1.0))

        return {"algo": "cmaes", **self._panel(role, x, fit[order], old_mean)}

    # ---------- what the explainer panel draws ----------

    def _panel(self, role: str, x_sorted: np.ndarray, fit_sorted: np.ndarray,
               old_mean: np.ndarray) -> dict:
        """Project this generation's real sample cloud onto its own two principal
        directions, and report the empirical covariance there.

        The basis is re-derived every generation but sign- and order-aligned to the
        previous one, so the ellipse rotates and breathes smoothly instead of flipping
        between frames.
        """
        Y = x_sorted - old_mean[None, :]
        # Principal directions of a (lam x n) cloud via the small lam x lam Gram matrix.
        G = Y @ Y.T
        vals, vecs = np.linalg.eigh(G)
        top = vecs[:, -2:][:, ::-1]                    # two leading directions
        B = (Y.T @ top)
        norms = np.linalg.norm(B, axis=0)
        B = B / np.maximum(norms, 1e-12)

        prev = self._proj.get(role)
        if prev is not None:
            for k in range(2):
                if float(prev[:, k] @ B[:, k]) < 0:
                    B[:, k] *= -1
        self._proj[role] = B

        P = Y @ B                                      # (lam, 2) the visible cloud
        step = (self.mean[role] - old_mean) @ B
        cov = np.cov(P.T) if len(P) > 2 else np.eye(2)
        ev, evec = np.linalg.eigh(cov)
        ev = np.maximum(ev, 1e-12)
        angle = float(np.arctan2(evec[1, -1], evec[0, -1]))

        return {
            "samples": np.round(P, 4).tolist(),
            "fitness": [round(float(v), 4) for v in fit_sorted],
            "selected": int(self.mu),
            "meanStep": [round(float(step[0]), 4), round(float(step[1]), 4)],
            "ellipse": {"rx": float(np.sqrt(ev[-1])), "ry": float(np.sqrt(ev[-2])),
                        "angle": angle},
            "sigma": self.sigma[role],
            "condition": float(self.C[role].max() / self.C[role].min()),
            "axisSpread": float(np.sqrt(self.C[role]).std()),
            "psNorm": float(np.linalg.norm(self.ps[role])) / self.chiN,
        }
