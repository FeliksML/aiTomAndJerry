"""The Examiner — a frozen, hand-written opponent that never learns.

This is `agents.js`'s scored greedy controller, vectorised and pinned at one skill
level. It exists to be a *yardstick*. Self-play scores only compare a policy to its
own sparring partner, so a school whose cat and mouse are both weak can post the
same numbers as a school where both are strong. Every learned policy also plays the
Examiner, on the same arenas, and that score is comparable across all three schools
and across time.

It is deliberately never trained and never tuned per school. It is also *privileged*
— it reads the arena's distance fields and line-of-sight directly, which the learned
policies cannot. That is fine for a benchmark (it is the same handicap for everyone)
but it does mean "beat the Examiner" is a harder bar than "beat a peer", and the
scoreboard should say so.

Behaviour, unchanged from the JS original:

  cat   sees her -> cut across her line home; else follow the scent; else check her
        last known cell; else camp the hole for 26 steps in every 54; else patrol.
  mouse run home, weighted against a threat field built from sight, then hearing,
        then memory; break line of sight; freeze behind cover rather than bolt.

Trap caution scales with skill and collapses when the chase is close — which is
where the snap moments come from.
"""

from __future__ import annotations

import numpy as np

from . import env as S
from .vec import NCELL, VecEnv, W, H

SKILL_DEFAULT = 0.60


def _trap_weight(skill, tunnel: np.ndarray) -> np.ndarray:
    """Learned caution: near zero while clumsy, full weight once competent — and
    almost gone when the target is right there."""
    w = 26.0 * np.clip((np.asarray(skill) - 0.2) / 0.62, 0.0, 1.0)
    return np.where(tunnel, w * 0.06, w)


class ScriptedPair:
    """Both scripted roles for one batch. Either half can be used on its own.

    `skill` may be a scalar or one value per environment. The per-environment form is
    what lets a single bot present a whole difficulty ladder inside one generation:
    a candidate is then scored against easy, medium and hard opponents at once, and
    its fitness stops depending on which rung the generation happened to land on.
    Cycling one skill per generation instead makes the objective move under the
    optimiser, and the measured result of that was a flat learning curve.
    """

    def __init__(self, env: VecEnv, skill=SKILL_DEFAULT, seed: int = 0):
        self.env = env
        self.skill = skill
        self.rng = np.random.default_rng(seed)
        n = env.n
        self._ar = np.arange(n)
        # cat memory
        self.last_seen = np.zeros(n, np.int32)
        self.has_last_seen = np.zeros(n, bool)
        self.last_seen_age = np.zeros(n, np.int32)
        self.camp_clock = np.zeros(n, np.int32)
        self.wander = np.zeros(n, np.int32)
        self.has_wander = np.zeros(n, bool)
        self.wander_age = np.zeros(n, np.int32)
        # mouse memory
        self.last_cat = np.zeros(n, np.int32)
        self.has_last_cat = np.zeros(n, bool)
        self.last_cat_age = np.zeros(n, np.int32)
        # what the last act() decided, for the on-screen thought line
        self.cat_mode = np.zeros(n, np.int8)     # 0 patrol 1 chase 2 intercept 3 scent 4 memory 5 camp
        self.mouse_mode = np.zeros(n, np.int8)   # 0 dash 1 listen 2 evade

    def reset(self, mask=None) -> None:
        if mask is None:
            mask = np.ones(self.env.n, bool)
        for a in (self.has_last_seen, self.has_wander, self.has_last_cat):
            a[mask] = False
        for a in (self.last_seen_age, self.camp_clock, self.wander_age, self.last_cat_age):
            a[mask] = 0

    # ---------- shared scoring ----------

    def _pick(self, me, scores_by_action) -> np.ndarray:
        """Blocked directions are not candidates at all — they must not stand in as a
        second, penalty-free way of standing still."""
        e = self.env
        sc = np.stack(scores_by_action, 1)                      # (n, 5)
        dest = e.M.move_to[e.map_idx[:, None], me[:, None], np.arange(5)[None, :]]
        blocked = (np.arange(5)[None, :] > 0) & (dest == me[:, None])
        sc = np.where(blocked, -1e9, sc)
        return sc.argmax(1).astype(np.int32)

    def _noise(self, n: int) -> np.ndarray:
        """Exploration noise, drawn FRESH PER CANDIDATE MOVE — returns (n, 5).

        This is what the skill scalar actually controls. Drawing one value per step
        instead would add the same constant to all five scores and leave the argmax
        untouched, making every skill level behave as a fully greedy controller.
        """
        amp = 17.0 * (1.0 - np.asarray(self.skill, np.float64)) ** 1.5
        return (self.rng.random((n, 5)) - 0.5) * np.reshape(amp, (-1, 1))

    # ---------- cat ----------

    def cat_act(self) -> np.ndarray:
        e, M, mi, sk = self.env, self.env.M, self.env.map_idx, self.skill
        n = e.n
        cat, mouse = e.cat, e.mouse
        vis = M.sees[mi, cat, e.cat_face, mouse]

        scent_cell, _, scent_ok = e.scent_cue()

        # Where he aims. The branches are exclusive and ordered, as in the original.
        sk = np.broadcast_to(np.asarray(sk, np.float64), (n,))
        m_vis = vis
        m_scent = ~m_vis & scent_ok & (sk > 0.25)
        m_mem = ~m_vis & ~m_scent & self.has_last_seen & (self.last_seen_age < 14) & (sk > 0.15)
        reach_camp = ~m_vis & ~m_scent & ~m_mem & (sk > 0.48)
        self.camp_clock[reach_camp] += 1
        m_camp = reach_camp & ((self.camp_clock % 54) < 26)
        m_wander = ~m_vis & ~m_scent & ~m_mem & ~m_camp

        # Lead her: aim at where her route home will have taken her. How far ahead is
        # itself a function of skill, so chain the greedy step to the deepest lead any
        # environment needs and pick the right depth per environment.
        lead = np.floor(3 * sk + 0.5).astype(np.int32)
        chain = [mouse.copy()]
        for _ in range(int(lead.max()) if len(lead) else 0):
            chain.append(M.next_home[mi, chain[-1]].astype(np.int32))
        ahead = np.stack(chain, 0)[lead, np.arange(n)] if len(chain) > 1 else mouse.copy()

        need = m_wander & (~self.has_wander | (self.wander_age > 18) | (self.rng.random(n) < 0.06))
        if need.any():
            k = int(need.sum())
            wx = 1 + (self.rng.random(k) * (W - 2)).astype(np.int32)
            wy = 1 + (self.rng.random(k) * (H - 2)).astype(np.int32)
            self.wander[need] = wy * W + wx
            self.wander_age[need] = 0
            self.has_wander[need] = True

        target = np.where(m_vis, ahead,
                  np.where(m_scent, scent_cell,
                   np.where(m_mem, self.last_seen,
                    np.where(m_camp, M.nest_cell[mi], self.wander)))).astype(np.int32)

        self.last_seen = np.where(m_vis, mouse, self.last_seen).astype(np.int32)
        self.has_last_seen |= m_vis
        self.last_seen_age = np.where(m_vis, 0, self.last_seen_age)
        self.last_seen_age += 1
        self.wander_age += 1
        self.cat_mode = np.where(m_vis, np.where(lead > 1, 2, 1),
                         np.where(m_scent, 3, np.where(m_mem, 4, np.where(m_camp, 5, 0)))).astype(np.int8)

        field = M.dist[mi, target]                              # (n, NCELL)
        gap = np.hypot(cat % W - mouse % W, cat // W - mouse // W)
        trap_w = _trap_weight(sk, vis & (gap <= 2.2))
        noise = self._noise(n)

        scores = []
        for a in range(5):
            cell = M.move_to[mi, cat, a].astype(np.int32)
            d = field[self._ar, cell]
            sc = np.where(d >= 0, -d.astype(np.float32) * (0.35 + 0.65 * sk), -40.0)
            sc -= np.where(M.grid[mi, cell] == S.TRAP, trap_w, 0.0)
            dn = M.nest_field[mi, cell]
            sc += np.where(m_camp & (dn >= 0) & (dn <= 3), 1.5 * sk, 0.0)
            if a == 0:
                sc -= np.where(m_camp, 0.2, 1.4)
            scores.append(sc + noise[:, a])
        return self._pick(cat, scores)

    # ---------- mouse ----------

    def mouse_act(self) -> np.ndarray:
        e, M, mi, sk = self.env, self.env.M, self.env.map_idx, self.skill
        n = e.n
        cat, mouse = e.cat, e.mouse
        sk = np.broadcast_to(np.asarray(sk, np.float64), (n,))
        vis = M.sees[mi, mouse, e.mouse_face, cat]

        heard_cell = (np.rint(e.heard[:, 1]).astype(np.int32) * W
                      + np.rint(e.heard[:, 0]).astype(np.int32))
        heard_cell = np.clip(heard_cell, 0, NCELL - 1)
        m_heard = ~vis & e.heard_on
        m_mem = ~vis & ~m_heard & self.has_last_cat & (self.last_cat_age < 10)

        belief = np.where(vis, cat, np.where(m_heard, heard_cell, self.last_cat)).astype(np.int32)
        conf = np.where(vis, 1.0, np.where(m_heard, e.heard[:, 2], np.where(m_mem, 0.4, 0.0)))
        has_belief = vis | m_heard | m_mem

        self.last_cat = np.where(vis, cat, self.last_cat).astype(np.int32)
        self.has_last_cat |= vis
        self.last_cat_age = np.where(vis, 0, self.last_cat_age)
        self.last_cat_age += 1
        self.mouse_mode = np.where(vis, 2, np.where(m_heard, 1, 0)).astype(np.int8)

        threat = M.dist[mi, belief]                             # (n, NCELL)
        here = M.nest_field[mi, mouse]
        commit = np.where((here >= 0) & (here <= 7), 0.2, 1.0)  # this close to home, take the risk

        gap = np.hypot(cat % W - mouse % W, cat // W - mouse // W)
        trap_w = _trap_weight(sk, vis & (gap <= 3))
        noise = self._noise(n)
        # Staying still only pays if he cannot see where she is standing.
        hidden = has_belief & ~M.los[mi, belief, mouse]

        scores = []
        for a in range(5):
            cell = M.move_to[mi, mouse, a].astype(np.int32)
            dn = M.nest_field[mi, cell]
            sc = np.where(dn >= 0, -dn.astype(np.float32) * (0.5 + 0.5 * sk), -40.0)

            dc = threat[self._ar, cell]
            near = has_belief & (dc >= 0)
            danger = np.maximum(0, 9 - dc).astype(np.float32)
            sc -= np.where(near, danger * danger * 0.34 * sk * conf * commit, 0.0)
            sc -= np.where(near & (dc <= 2), 20.0 * sk * commit, 0.0)

            sc -= np.where(M.grid[mi, cell] == S.TRAP, trap_w, 0.0)
            sc += np.where(vis & ~M.los[mi, cat, cell] & (sk > 0.3), 6.0 * sk, 0.0)
            if a == 0:
                sc += np.where(hidden & (conf > 0.5) & (sk > 0.6), 1.5 * sk, -1.4)
            scores.append(sc + noise[:, a])
        return self._pick(mouse, scores)


CAT_MODES = ("patrol", "chase", "intercept", "scent", "memory", "camp")
MOUSE_MODES = ("dash", "listen", "evade")
