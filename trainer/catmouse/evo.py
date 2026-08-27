"""Shared machinery for the two population schools.

The GA and CMA-ES differ only in how they turn last generation's scores into next
generation's candidates. Everything else — how a generation is scored, who it plays,
how the budget is counted — is here, so the two are compared on identical terms.

Scoring a generation, and why it is built this way:

  * **One batch.** P candidates x E episodes are all stepped together as
    `env = candidate * E + slot`, and the whole generation's forward pass is a single
    batched matmul. A generation costs one env loop, not P of them.
  * **Common random numbers.** Every candidate plays the same arenas in the same
    slots, meets the same hearing noise (`noise_tile`), and samples its actions from
    the same uniforms (`u`). Two candidates therefore differ by *policy*, not by luck.
    Without this, a 48-way ranking on 24 noisy episodes is mostly a lottery, and the
    selection pressure goes into the noise.
  * **A fixed opponent panel.** Slot j always faces the same panel member, so every
    candidate meets the same ladder. Half the panel is past selves spanning the whole
    archive; half is the scripted controller at this generation's ladder skill. See
    `league.py` for why the scripted half is not optional.

Fitness is the episode return the environment already defines (terminal +/-1 plus the
shaping terms) — the spec's own definition, not a second scoring system invented here.
"""

from __future__ import annotations

import numpy as np

from . import env as S
from .arena import HallOfFame
from .league import Promotion, ladder_for, scripted_share, strided_archive
from .nets import FlatActor, init_flat
from .school import School
from .scripted import ScriptedPair

OTHER = {"cat": "mouse", "mouse": "cat"}


class EvoSchool(School):
    """Base for population methods. Subclasses implement ask() / tell()."""

    pop_size = 48
    eps_per_genome = 12
    n_opponents = 4
    scripted_slots = 2          # of n_opponents; the rest are past selves
    hof_every = 4

    def setup(self) -> None:
        init = init_flat(2, self.rng)
        self.best = {"cat": init[0].copy(), "mouse": init[1].copy()}
        self.hof = {r: HallOfFame(cap=24) for r in ("cat", "mouse")}
        for r in ("cat", "mouse"):
            self.hof[r].add(self.best[r])
        self.gen = 0
        self.promo = {r: Promotion() for r in ("cat", "mouse")}
        self._eval_env = None
        self.setup_optimiser()

    def setup_optimiser(self) -> None:
        raise NotImplementedError

    def ask(self, role: str) -> np.ndarray:
        """Candidates for this generation, shaped (P, FLAT_DIM)."""
        raise NotImplementedError

    def tell(self, role: str, pop: np.ndarray, fitness: np.ndarray) -> dict:
        """Consume the scores, move the search, return telemetry for the explainer."""
        raise NotImplementedError

    def params(self, role: str) -> np.ndarray:
        return self.best[role]

    # ---------- scoring a generation ----------

    def _env_for(self, n: int, seed: int):
        from .vec import VecEnv
        if self._eval_env is None or self._eval_env.n != n:
            # Learning-strength shaping: this env produces FITNESS, not scores.
            self._eval_env = VecEnv(self.maps, n, seed=seed).training_shaping()
            self._eval_env.noise_tile = self.eps_per_genome
            self._bot = ScriptedPair(self._eval_env, 0.5, seed=seed + 77)
        return self._eval_env

    def evaluate(self, role: str, pop: np.ndarray, panel: list[np.ndarray],
                 seed: int, phase: int = 0):
        """Score every candidate against the league. Returns (fitness, counts, steps)."""
        P, E = len(pop), self.eps_per_genome
        K = self.n_opponents
        if E % K:
            raise ValueError("episodes per genome must be a multiple of the opponent panel")
        n = P * E
        e = self._env_for(n, seed)
        e.rng = np.random.default_rng(seed)
        before = e.env_steps
        slot = np.arange(n) % E
        e.reset(map_idx=slot % len(self.maps))

        # Slots [0, scripted_slots) face the scripted ladder; the rest face past selves.
        # Every candidate meets the identical mix at the identical difficulties, so
        # fitness is comparable between candidates AND between generations.
        use_bot = (slot % K) < self.scripted_slots
        bot = self._bot
        bot.skill = ladder_for(slot, phase)
        bot.rng = np.random.default_rng(seed + 9)
        bot.reset()
        learner = FlatActor(pop, self.device, assign=np.repeat(np.arange(P), E))
        opponent = FlatActor(np.stack(panel), self.device, assign=slot % K)
        rng_l = np.random.default_rng(seed + 5)
        rng_o = np.random.default_rng(seed + 6)

        fit = np.zeros(n, np.float32)
        for _ in range(S.MAX_STEPS + 1):
            if e.done.all():
                break
            oc = e.observe("cat")
            om = e.observe("mouse")
            ul = np.tile(rng_l.random(E), P)
            uo = np.tile(rng_o.random(E), P)
            if role == "cat":
                a_c = learner.act(oc, rng_l, u=ul)
                a_m = np.where(use_bot, bot.mouse_act(), opponent.act(om, rng_o, u=uo))
            else:
                a_c = np.where(use_bot, bot.cat_act(), opponent.act(oc, rng_o, u=uo))
                a_m = learner.act(om, rng_l, u=ul)
            rc, rm, _, _ = e.step(a_c, a_m)
            fit += rc if role == "cat" else rm
        self.run.steps += e.env_steps - before

        res = e.result.reshape(P, E)
        want = 1 if role == "cat" else 2
        # The promotion signal is the win rate against the LADDER only. Counting the
        # self-play slots would let a school promote itself by beating its own weak
        # opponent, which is exactly the feedback loop this design is avoiding.
        sc = use_bot.reshape(P, E)
        return (
            fit.reshape(P, E).mean(1),
            {"win": float((res == want).mean()),
             "ladderWin": float((res == want)[sc].mean()) if sc.any() else 0.0,
             "loss": float((res == (2 if role == "cat" else 1)).mean()),
             "draw": float((res == 3).mean())},
            float(e.step_n.mean()),
        )

    # ---------- one generation ----------

    def iteration(self) -> dict:
        frac = self.budget.fraction(self.run.elapsed, self.run.steps)
        self.scripted_slots = int(round(self.n_opponents * scripted_share(frac)))
        tel: dict = {"gen": self.gen, "scriptedSlots": self.scripted_slots}
        for role in ("cat", "mouse"):
            n_net = self.n_opponents - self.scripted_slots
            past = strided_archive(self.hof[OTHER[role]].items, max(0, n_net - 1))
            panel = [self.best[OTHER[role]]] * self.scripted_slots \
                + [self.best[OTHER[role]]] + past       # scripted slots ignore their entry
            panel = panel[:self.n_opponents]
            while len(panel) < self.n_opponents:
                panel.append(self.best[OTHER[role]])
            pop = self.ask(role)
            base = self.run.steps
            fit, counts, steps = self.evaluate(
                role, pop, panel, seed=self.seed + self.gen * 131 + (role == "mouse"),
                phase=self.promo[role].phase)
            t = self.tell(role, pop, fit)
            promoted = self.promo[role].update(counts["ladderWin"], self.run.steps)
            if promoted:
                self.emit("promotion", role=role, year=self.promo[role].phase + 1)
            tel[role] = {
                **t,
                "fitBest": float(fit.max()), "fitMean": float(fit.mean()),
                "fitWorst": float(fit.min()), "winRate": counts["win"],
                "ladderWin": counts["ladderWin"], "year": self.promo[role].phase + 1,
                "promoted": promoted,
                "drawRate": counts["draw"], "meanSteps": steps,
                "stepsThisGen": self.run.steps - base,
            }
        self.gen += 1
        if self.gen % self.hof_every == 0:
            for r in ("cat", "mouse"):
                self.hof[r].add(self.best[r])
        return tel


def fingerprint(pop: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """A short, stable signature per genome — what the GA panel draws as DNA strips.

    A fixed random projection, so two genomes that are close in weight space produce
    similar-looking strips and a child visibly resembles its parents. Slicing raw
    weights instead would show 48 bars of noise.
    """
    v = pop @ basis
    s = np.abs(v).max(axis=0, keepdims=True)
    return np.clip(v / np.maximum(s, 1e-6), -1, 1)
