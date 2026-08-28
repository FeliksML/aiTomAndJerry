"""The one brain all three schools train.

Fairness starts with nobody getting a bigger head. Every policy — PPO's, the GA's,
CMA-ES's — is the *same* architecture with the *same* parameter count, reading the
*same* 50-number observation:

    50 -> 32 -> tanh -> 32 -> tanh -> 5 logits          2,853 parameters

and every policy serialises to one flat float32 vector of that length. That single
format is what lets a PPO cat and a CMA-ES mouse meet in the tournament without any
per-algorithm special casing, and what makes a checkpoint a plain array.

The network is deliberately small. Gradients do not care much about dimensionality,
but evolution does: on 7,000 weights the population methods spend the whole budget
crawling, and the comparison would end up measuring how each algorithm scales with
dimension rather than how well it plays the game. Two hidden layers of 32 are ample
for a 40-number observation and give all three a fair run at it.

PPO additionally trains a critic. That is not extra policy capacity — it is thrown
away at evaluation time and never sees the arena — but the scoreboard should say so
rather than pretend the three runs are identical in every respect.

Two forward paths, one set of weights:

    PolicyNet        a normal torch module, for PPO's gradients
    forward_flat     P whole populations at once, for the GA and CMA-ES

Actions are always *sampled* from the softmax, for every algorithm. A deterministic
argmax policy in a pursuit game walks into limit cycles — the cat paces, the mouse
oscillates in a corridor, and the episode dies on the step limit. Sampling is also
the only fair setting, since PPO is stochastic by construction.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vec import OBS_DIM

HIDDEN = (32, 32)
N_ACTIONS = 5
LAYERS = [(OBS_DIM, HIDDEN[0]), (HIDDEN[0], HIDDEN[1]), (HIDDEN[1], N_ACTIONS)]
FLAT_DIM = sum(i * o + o for i, o in LAYERS)


def pick_device(prefer: str = "auto", P: int = 48, B: int = 12) -> torch.device:
    """Time the real thing, not the matmul.

    A bare `forward_flat` benchmark says MPS wins by 8x. The call that training
    actually makes — upload the observations, forward, softmax, inverse-CDF sample,
    download the actions — says the opposite, because at this size the run is bound by
    how many GPU kernels get launched rather than by their arithmetic. Measuring the
    wrong half of the work picks the wrong device, so this benchmarks the whole call.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if not torch.backends.mps.is_available():
        return torch.device("cpu")
    import time
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((P * B, OBS_DIM)).astype(np.float32)
    u = np.tile(rng.random(B), P)
    best, best_dt = torch.device("cpu"), float("inf")
    for name in ("cpu", "mps"):
        d = torch.device(name)
        actor = FlatActor(init_flat(P, rng), d, assign=np.repeat(np.arange(P), B))
        for _ in range(10):
            actor.act(obs, rng, u=u)
        t0 = time.perf_counter()
        for _ in range(60):
            actor.act(obs, rng, u=u)
        dt = time.perf_counter() - t0
        if dt < best_dt:
            best, best_dt = d, dt
    return best


# ---------- flat parameter vector <-> per-layer views ----------

def unpack(flat: torch.Tensor):
    """(P, FLAT_DIM) -> [(W, b), ...] with W shaped (P, in, out) and b shaped (P, 1, out)."""
    out, k = [], 0
    P = flat.shape[0]
    for i, o in LAYERS:
        W = flat[:, k:k + i * o].reshape(P, i, o)
        k += i * o
        b = flat[:, k:k + o].reshape(P, 1, o)
        k += o
        out.append((W, b))
    return out


def forward_flat(flat: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """P populations x B environments in one pass.

    flat (P, FLAT_DIM), obs (P, B, OBS_DIM) -> logits (P, B, 5).
    This is the shape the GA and CMA-ES live in: a whole generation evaluated as one
    batched matmul instead of P separate networks.
    """
    x = obs
    layers = unpack(flat)
    for k, (W, b) in enumerate(layers):
        x = torch.baddbmm(b, x, W)
        if k < len(layers) - 1:
            x = torch.tanh(x)
    return x


def init_flat(n: int, rng: np.random.Generator, gain: float = 1.0) -> np.ndarray:
    """n independent networks, each initialised the way torch would initialise one.

    Every algorithm starts from this same distribution, so no school gets a luckier
    starting point than another.
    """
    out = np.zeros((n, FLAT_DIM), np.float32)
    k = 0
    for li, (i, o) in enumerate(LAYERS):
        # Xavier for the tanh trunk, deliberately small for the logit layer so the
        # first policy is close to uniform rather than already committed.
        scale = gain * np.sqrt(2.0 / (i + o)) if li < len(LAYERS) - 1 else 0.01
        out[:, k:k + i * o] = rng.normal(0, scale, (n, i * o)).astype(np.float32)
        k += i * o
        k += o  # biases start at zero
    return out


# ---------- the torch module PPO differentiates ----------

class PolicyNet(nn.Module):
    """Same weights as the flat vector, laid out for autograd."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(OBS_DIM, HIDDEN[0])
        self.l2 = nn.Linear(HIDDEN[0], HIDDEN[1])
        self.l3 = nn.Linear(HIDDEN[1], N_ACTIONS)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.l1(obs))
        x = torch.tanh(self.l2(x))
        return self.l3(x)

    def flat(self) -> np.ndarray:
        parts = []
        for lin in (self.l1, self.l2, self.l3):
            parts.append(lin.weight.detach().t().reshape(-1).cpu().numpy())
            parts.append(lin.bias.detach().reshape(-1).cpu().numpy())
        return np.concatenate(parts).astype(np.float32)

    @torch.no_grad()
    def load_flat(self, v: np.ndarray) -> "PolicyNet":
        k = 0
        for lin, (i, o) in zip((self.l1, self.l2, self.l3), LAYERS):
            lin.weight.copy_(torch.as_tensor(v[k:k + i * o].reshape(i, o).T.copy()))
            k += i * o
            lin.bias.copy_(torch.as_tensor(v[k:k + o].copy()))
            k += o
        return self


class Critic(nn.Module):
    """PPO only. Never plays; never enters the tournament."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(OBS_DIM, HIDDEN[0])
        self.l2 = nn.Linear(HIDDEN[0], HIDDEN[1])
        self.l3 = nn.Linear(HIDDEN[1], 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.l1(obs))
        x = torch.tanh(self.l2(x))
        return self.l3(x).squeeze(-1)


# ---------- acting ----------

class FlatActor:
    """Plays one or many flat policies against a VecEnv.

    `assign` maps each environment to a policy row, so a whole generation shares one
    batch: environment e is played by policy `assign[e]`.

    The permutation that groups environments by policy is fixed for the whole
    evaluation, so it is built once. Rebuilding it per step costs more than the
    forward pass — at this size the run is bound by how many GPU kernels get launched,
    not by how much arithmetic they do.
    """

    def __init__(self, flat: np.ndarray, device: torch.device, assign: np.ndarray | None = None):
        self.device = device
        self.set_params(flat)
        self.assign = assign
        self._perm = None
        self._inv = None
        self._contig = False
        if assign is not None and self.P > 1:
            counts = np.bincount(assign, minlength=self.P)
            if counts.min() != counts.max():
                raise ValueError("FlatActor.assign must give every policy the same number of envs")
            self.per = int(counts[0])
            # Already policy-major (0,0,..,1,1,..)? Then a plain reshape is the whole job.
            self._contig = bool(np.array_equal(assign, np.repeat(np.arange(self.P), self.per)))
            if not self._contig:
                order = np.argsort(assign, kind="stable")
                self._perm = torch.as_tensor(order, device=device)
                self._inv = torch.as_tensor(np.argsort(order, kind="stable"), device=device)

    def set_params(self, flat: np.ndarray) -> None:
        f = np.atleast_2d(np.asarray(flat, np.float32))
        if f.shape[1] != FLAT_DIM:
            # Almost always a checkpoint from before the observation changed shape.
            raise ValueError(
                f"policy has {f.shape[1]} weights, this build expects {FLAT_DIM}. "
                f"That run was trained against a different observation "
                f"(OBS_DIM is now {OBS_DIM}) — retrain, or check out the commit it came from.")
        self.P = f.shape[0]
        self.flat = torch.as_tensor(f, device=self.device)

    def _logits(self, o: torch.Tensor) -> torch.Tensor:
        if self.P == 1:
            return forward_flat(self.flat, o.unsqueeze(0)).squeeze(0)
        if self._contig or self.assign is None:
            per = o.shape[0] // self.P
            return forward_flat(self.flat, o.reshape(self.P, per, -1)).reshape(o.shape[0], -1)
        g = forward_flat(self.flat, o[self._perm].reshape(self.P, self.per, -1))
        return g.reshape(o.shape[0], -1)[self._inv]

    @torch.no_grad()
    def act(self, obs: np.ndarray, rng: np.random.Generator, greedy: bool = False,
            u: np.ndarray | None = None) -> np.ndarray:
        """`u` supplies the sampling uniforms, so a population can share one draw and
        differ only by its policies — the other half of common random numbers."""
        logits = self._logits(torch.as_tensor(obs, device=self.device))
        if greedy:
            return logits.argmax(-1).cpu().numpy().astype(np.int32)
        # Inverse-CDF sampling on the device. Bringing the (B, 5) probability matrix
        # back to the host every step costs more than the forward pass itself.
        uu = rng.random(logits.shape[0]) if u is None else u
        ut = torch.as_tensor(np.asarray(uu, np.float32), device=logits.device).unsqueeze(1)
        cdf = torch.softmax(logits, -1).cumsum(-1)
        a = (cdf < ut).sum(-1).clamp_(0, N_ACTIONS - 1)
        return a.to(torch.int32).cpu().numpy()

    @torch.no_grad()
    def probs(self, obs: np.ndarray) -> np.ndarray:
        """Action probabilities — what the PPO panel draws as five morphing bars."""
        o = torch.as_tensor(np.asarray(obs, np.float32), device=self.device)
        return F.softmax(self._logits(o), -1).cpu().numpy()
