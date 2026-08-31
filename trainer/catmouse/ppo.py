"""PPO school — gradients, with a leash on how far each update may move.

Both roles learn at once from the same episodes (independent PPO), which is the
self-play the spec asks for. Naive self-play cycles, though — the cat learns to beat
this week's mouse, the mouse beats that cat, and neither gets good in general — so a
quarter of the environments pit the learner against a frozen past opponent from the
hall of fame. Environments are split once per rollout and never mix:

    group A  half the envs    current cat  vs current mouse   both learn here
    group B  a quarter        current cat  vs frozen mouse    only the cat learns
    group C  a quarter        frozen cat   vs current mouse   only the mouse learns

so each learner sees 75% of the batch and never trains on an action it did not take.

What the on-screen explainer draws, all measured during the real update:
  * the five action probabilities on a fixed probe state, before and after
  * the importance ratio against the clip band at 1 +/- eps, and how much got clipped
  * entropy, approximate KL, value loss, explained variance
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import arena
from .nets import Critic, FlatActor, PolicyNet, init_flat
from .league import Promotion, calibrated_bars, ladder_for, scripted_share
from .school import School, apply_hyper
from .scripted import ScriptedPair
from .vec import VecEnv

OTHER_ROLE = {"cat": "mouse", "mouse": "cat"}


@dataclass
class PPOConfig:
    n_envs: int = 512
    horizon: int = 128
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_start: float = 0.02
    ent_end: float = 0.004
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hof_every: int = 20          # rollouts between snapshots into the hall of fame
    probe_states: int = 96


class PPOSchool(School):
    key = "ppo"
    label = "Proximal Policy Optimization"

    TUNABLES = (
        {"key": "n_envs", "label": "PARALLEL ENVIRONMENTS", "min": 128, "max": 8192,
         "step": 128, "hint": "the batch. 2048 is ~35% more steps a second than 512 here"},
        {"key": "horizon", "label": "ROLLOUT LENGTH", "min": 32, "max": 512, "step": 32,
         "hint": "steps per environment before each update"},
        {"key": "lr", "label": "LEARNING RATE", "min": 1e-5, "max": 3e-3, "step": 1e-5,
         "hint": "how far Adam moves"},
        {"key": "clip", "label": "CLIP eps", "min": 0.05, "max": 0.4, "step": 0.01,
         "hint": "the leash: how far one update may move the policy"},
        {"key": "ent_start", "label": "ENTROPY AT THE START", "min": 0.0, "max": 0.08,
         "step": 0.002, "hint": "how much it is paid to keep trying things"},
    )

    def __init__(self, *a, cfg: PPOConfig | None = None, **kw):
        super().__init__(*a, **kw)
        self.cfg = apply_hyper(cfg or PPOConfig(), self.hyper)

    # ---------- setup ----------

    def setup(self) -> None:
        c, dev = self.cfg, self.device
        init = init_flat(2, self.rng)
        self.net = {r: PolicyNet().load_flat(init[i]).to(dev) for i, r in enumerate(("cat", "mouse"))}
        self.critic = {r: Critic().to(dev) for r in ("cat", "mouse")}
        self.opt = {
            r: torch.optim.Adam(
                list(self.net[r].parameters()) + list(self.critic[r].parameters()), lr=c.lr)
            for r in ("cat", "mouse")
        }
        self.hof = {r: arena.HallOfFame() for r in ("cat", "mouse")}
        for r in ("cat", "mouse"):
            self.hof[r].add(self.net[r].flat())

        n = c.n_envs
        # Learning-strength shaping. Scoring envs keep the spec's coefficients.
        self.env = VecEnv(self.maps, n, seed=self.seed).training_shaping(self.shaping)
        self.env.reset(map_idx=np.arange(n) % len(self.maps))
        half, quarter = n // 2, n // 4
        self.grp = np.zeros(n, np.int8)      # 0 = A, 1 = B, 2 = C
        self.grp[half:half + quarter] = 1
        self.grp[half + quarter:] = 2
        self.own = {"cat": self.grp != 2, "mouse": self.grp != 1}
        self.frozen_env = {"cat": np.flatnonzero(self.grp == 2),
                           "mouse": np.flatnonzero(self.grp == 1)}
        self.probe: dict[str, torch.Tensor] = {}
        self.bot = ScriptedPair(self.env, 0.5, seed=self.seed + 77)
        self.promo = {r: Promotion(calibrated_bars(self.maps, r)) for r in ("cat", "mouse")}

    def params(self, role: str) -> np.ndarray:
        return self.net[role].flat()

    # ---------- one rollout + update ----------

    def iteration(self) -> dict:
        c, dev = self.cfg, self.device
        n, T = c.n_envs, c.horizon
        frac = self.budget.fraction(self.run.elapsed, self.run.steps)
        ent_coef = c.ent_start + (c.ent_end - c.ent_start) * frac

        # The opponent quarter of the batch: partly a frozen past self, partly the
        # scripted ladder. Same league and same schedule as the population schools.
        # The two roles can sit in different years, so the ladder is built per role.
        share = scripted_share(frac)
        ladder = {r: ladder_for(np.arange(n), self.promo[r].phase) for r in ("cat", "mouse")}
        frozen, bot_env = {}, {}
        for r in ("cat", "mouse"):
            picks = self.hof[r].sample(1, self.rng)
            frozen[r] = FlatActor(picks[0], dev) if picks else None
            ov = self.frozen_env[r]
            cut = int(len(ov) * share)
            bot_env[r] = ov[:cut]
            frozen[r] = (frozen[r], ov[cut:])

        buf = {r: {k: [] for k in ("obs", "act", "logp", "val", "rew", "done")}
               for r in ("cat", "mouse")}
        # Learner outcomes in the envs where the OPPONENT is the scripted ladder — the
        # promotion signal. Self-play envs are excluded so a school cannot promote
        # itself by beating its own weak opponent.
        ladder_wins = {"cat": [0, 0], "mouse": [0, 0]}

        for _ in range(T):
            obs = {"cat": self.env.observe("cat"), "mouse": self.env.observe("mouse")}
            acts, step_rec = {}, {}
            for r in ("cat", "mouse"):
                o = torch.as_tensor(obs[r], device=dev)
                with torch.no_grad():
                    logits = self.net[r](o)
                    value = self.critic[r](o)
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                    lp = dist.log_prob(a)
                a_np = a.cpu().numpy().astype(np.int32)
                step_rec[r] = (obs[r], a_np, lp.cpu().numpy(), value.cpu().numpy())
                # A frozen past self plays this role in its own quarter of the batch.
                # Those envs are excluded from this role's update, so the mismatch
                # between the stored log-prob and the played action never reaches a loss.
                net_opp, net_env = frozen[r]
                a_np = a_np.copy()
                if len(bot_env[r]):
                    # `r` is the role the scripted bot is standing in for, so it faces
                    # the OTHER role's ladder year.
                    self.bot.skill = ladder[OTHER_ROLE[r]]
                    scripted = self.bot.cat_act() if r == "cat" else self.bot.mouse_act()
                    a_np[bot_env[r]] = scripted[bot_env[r]]
                if net_opp is not None and len(net_env):
                    a_np[net_env] = net_opp.act(obs[r][net_env], self.rng)
                acts[r] = a_np

            rc, rm, done, res = self.env.step(acts["cat"], acts["mouse"])
            for r in ("cat", "mouse"):
                idx = bot_env[r]
                if len(idx):
                    fin = done[idx]
                    if fin.any():
                        want = 2 if r == "cat" else 1     # the LEARNER's win, not the bot's
                        ladder_wins[r][0] += int((res[idx][fin] == want).sum())
                        ladder_wins[r][1] += int(fin.sum())
            for r, rew in (("cat", rc), ("mouse", rm)):
                o, a, lp, v = step_rec[r]
                buf[r]["obs"].append(o)
                buf[r]["act"].append(a)
                buf[r]["logp"].append(lp)
                buf[r]["val"].append(v)
                buf[r]["rew"].append(rew)
                buf[r]["done"].append(done.astype(np.float32))
            self.run.steps = self.env.env_steps
            if done.any():
                self.env.reset_where(done, map_idx=self.rng.integers(0, len(self.maps), int(done.sum())))

        tel = {"entCoef": ent_coef, "scriptedShare": share}
        for r in ("cat", "mouse"):
            # The cat's ladder record lives in the envs where the MOUSE is scripted.
            w, n_done = ladder_wins[OTHER_ROLE[r]]
            rate = w / n_done if n_done else 0.0
            promoted = self.promo[r].update(rate, self.run.steps)
            if promoted:
                self.emit("promotion", role=r, year=self.promo[r].phase + 1)
            tel[r] = {**self._update(r, buf[r], ent_coef),
                      "ladderWin": rate, "year": self.promo[r].phase + 1,
                      "promoted": promoted}
        return tel

    def _update(self, role: str, b: dict, ent_coef: float) -> dict:
        c, dev = self.cfg, self.device
        own = self.own[role]
        cols = np.flatnonzero(own)
        T = c.horizon

        obs = np.stack(b["obs"])[:, cols]                      # (T, E, OBS)
        act = np.stack(b["act"])[:, cols]
        logp = np.stack(b["logp"])[:, cols]
        val = np.stack(b["val"])[:, cols]
        rew = np.stack(b["rew"])[:, cols]
        done = np.stack(b["done"])[:, cols]

        with torch.no_grad():
            last_v = self.critic[role](
                torch.as_tensor(self.env.observe(role)[cols], device=dev)).cpu().numpy()

        # GAE. An episode that ended is bootstrapped at zero; the env autoresets, so
        # the next row already belongs to a fresh episode.
        adv = np.zeros_like(rew)
        gae = np.zeros(len(cols), np.float32)
        for t in range(T - 1, -1, -1):
            nxt = last_v if t == T - 1 else val[t + 1]
            nonterm = 1.0 - done[t]
            delta = rew[t] + c.gamma * nxt * nonterm - val[t]
            gae = delta + c.gamma * c.lam * nonterm * gae
            adv[t] = gae
        ret = adv + val

        flat = lambda x, dt=None: torch.as_tensor(  # noqa: E731
            x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1), device=dev, dtype=dt)
        O = flat(obs, torch.float32)
        A = flat(act, torch.int64)
        LP = flat(logp, torch.float32)
        AD = flat(adv, torch.float32)
        RT = flat(ret, torch.float32)
        VOLD = flat(val, torch.float32)

        N = O.shape[0]
        mb = N // c.minibatches
        stats = {"ratio": [], "clipped": [], "kl": [], "ent": [], "pl": [], "vl": []}
        for _ in range(c.epochs):
            perm = torch.randperm(N, device=dev)
            for i in range(c.minibatches):
                sl = perm[i * mb:(i + 1) * mb]
                logits = self.net[role](O[sl])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(A[sl])
                ratio = torch.exp(lp - LP[sl])
                a = AD[sl]
                a = (a - a.mean()) / (a.std() + 1e-8)
                unclipped = ratio * a
                clipped = torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * a
                pol_loss = -torch.min(unclipped, clipped).mean()
                v = self.critic[role](O[sl])
                v_loss = F.mse_loss(v, RT[sl])
                ent = dist.entropy().mean()
                loss = pol_loss + c.vf_coef * v_loss - ent_coef * ent

                self.opt[role].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.net[role].parameters()) + list(self.critic[role].parameters()),
                    c.max_grad_norm)
                self.opt[role].step()

                with torch.no_grad():
                    stats["ratio"].append(ratio.detach())
                    stats["clipped"].append(((ratio - 1).abs() > c.clip).float().mean().item())
                    stats["kl"].append((LP[sl] - lp).mean().item())
                    stats["ent"].append(ent.item())
                    stats["pl"].append(pol_loss.item())
                    stats["vl"].append(v_loss.item())

        if self.run.iters % c.hof_every == 0:
            self.hof[role].add(self.net[role].flat())

        return self._telemetry(role, stats, O, VOLD, RT)

    # ---------- what the explainer panel draws ----------

    def _telemetry(self, role: str, stats: dict, O: torch.Tensor,
                   v_old: torch.Tensor, ret: torch.Tensor) -> dict:
        c = self.cfg
        with torch.no_grad():
            # histc has no MPS kernel — the histogram is 32 bins, so CPU is free.
            ratios = torch.cat(stats["ratio"]).float().cpu()
            lo, hi = 1 - c.clip, 1 + c.clip
            e0, e1 = lo - 0.35, hi + 0.35
            hist = torch.histc(ratios.clamp(e0, e1), 32, e0, e1)
            # A fixed probe batch, so the five bars are comparable across updates
            # instead of jumping because the states changed underneath them.
            if role not in self.probe:
                self.probe[role] = O[torch.randperm(O.shape[0], device=O.device)[:c.probe_states]].clone()
            probe_p = torch.softmax(self.net[role](self.probe[role]), -1).mean(0)
            var = ret.var()
            ev = float(1 - (ret - v_old).var() / var) if float(var) > 1e-8 else 0.0

        return {
            "actionProbs": [round(float(x), 5) for x in probe_p],
            "ratioHist": [int(x) for x in hist.cpu()],
            "ratioRange": [e0, e1],
            "clipBand": [lo, hi],
            "clippedFrac": float(np.mean(stats["clipped"])),
            "approxKl": float(np.mean(stats["kl"])),
            "entropy": float(np.mean(stats["ent"])),
            "policyLoss": float(np.mean(stats["pl"])),
            "valueLoss": float(np.mean(stats["vl"])),
            "explainedVar": ev,
            "hof": len(self.hof[role]),
        }
