"""Train the schools.

    python trainer/scripts/train.py --minutes 30                 # all three, in parallel
    python trainer/scripts/train.py --school ppo --minutes 5     # just one
    python trainer/scripts/train.py --steps 500M                 # count steps, not minutes
    python trainer/scripts/train.py --steps 500M --minutes 120   # 500M steps, hard-capped at 2h

`--minutes` and `--steps` are the two budget clocks. Give both and the run ends at
whichever arrives first, which is the safe way to start something overnight.

All three run at once in separate processes. That is not only faster — it is the
fairness protocol: identical wall-clock, on the same machine, under the same load.
Each process is pinned to a slice of the cores so one school cannot starve another.

Every run writes, under `runs/<tag>/<school>/`:
    checkpoints.npz   the three Academy checkpoints plus BEST, both roles, as flat
                      weight vectors
    best.json         which policy each role actually ended up best at, the peak-versus-
                      finish run-off that decided it, and where in the run the peak was
    history.json      the Examiner curve against both budget clocks
    events.jsonl      every telemetry event, including the algorithm internals the
                      on-screen explainers draw. This is also what a TRAIN-mode replay
                      is served from.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "trainer"))

SCHOOLS = ("ppo", "ga", "cmaes")


def with_suppressed(fn) -> None:
    """A child that has already exited raises on terminate(); that is not an error."""
    try:
        fn()
    except Exception:
        pass


def build(school: str, maps, emaps, device, budget, seed, on_event, out, envs=None):
    from catmouse.cmaes import CMAESSchool
    from catmouse.ga import GASchool
    from catmouse.ppo import PPOSchool, PPOConfig
    cls = {"ppo": PPOSchool, "ga": GASchool, "cmaes": CMAESSchool}[school]
    kw = {}
    # PPO's batch is the one hyperparameter worth exposing on this machine: the
    # environment is single-threaded NumPy, so a wider batch is the only thing that
    # turns more of the chip into steps per second. Measured on an M2 Max, 2048 envs
    # runs ~35% more steps per second than 512. The population schools size their batch
    # from the population, so the flag does not apply to them.
    if school == "ppo" and envs:
        kw["cfg"] = PPOConfig(n_envs=int(envs))
    return cls(maps, emaps, device, budget, seed=seed, on_event=on_event, out_dir=out, **kw)


def run_one(school: str, minutes: float | None, steps: int | None, seed: int, out: Path,
            threads: int, device: str, nests=None, envs: int | None = None) -> None:
    import torch
    torch.set_num_threads(threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))

    from catmouse import arena, nets, vec
    from catmouse.school import Budget, human_steps

    nests = arena.parse_nests(nests)
    dev = nets.pick_device(device)
    maps = vec.MapSet(arena.TRAIN_SEEDS, arena.spread(nests, len(arena.TRAIN_SEEDS)))
    emaps = vec.MapSet(arena.EVAL_SEEDS, arena.spread(nests, len(arena.EVAL_SEEDS)))
    out.mkdir(parents=True, exist_ok=True)
    (out / school).mkdir(parents=True, exist_ok=True)
    # Line-buffered on purpose. This file is not only a record: the app tails it to draw
    # the live progress, and with the default 8KB block buffer a school's telemetry only
    # appeared once the buffer happened to fill — PPO's events are small, so its progress
    # bar sat at zero for the whole run and then jumped to 100%.
    log = (out / school / "events.jsonl").open("w", buffering=1)

    t0 = time.time()
    budget = Budget(seconds=None if minutes is None else minutes * 60, steps=steps)

    def clock(ev: dict) -> str:
        """Where the run is, on the clock that is actually going to end it."""
        target = (f"/{human_steps(ev['targetSteps'])}" if ev.get("targetSteps") else "")
        eta = ev.get("eta")
        eta_s = f"  eta {eta / 60:5.1f}m" if eta is not None else ""
        return (f"{ev['frac']:5.0%} it={ev['iter']:6d} "
                f"steps={human_steps(ev['steps']):>6}{target:>8}  "
                f"{ev.get('sps', 0) / 1e3:5.1f}k/s{eta_s}")

    def line(text: str) -> None:
        """Overwrite the heartbeat rather than printing over half of it. Three schools
        share one terminal, and a `\r` line that is shorter than the one under it leaves
        the tail of the old one behind — which on camera reads as a garbled number."""
        print("\r" + text.ljust(108), flush=True)

    def on_event(ev: dict) -> None:
        log.write(json.dumps(ev) + "\n")
        if ev["kind"] == "progress":
            print("\r" + f"[{school:5}] {clock(ev)}".ljust(108), end="", flush=True)
        elif ev["kind"] == "eval":
            line(f"[{school:5}] {clock(ev)}  cat={ev['catExam']:6.1%} "
                 f"mouse={ev['mouseExam']:6.1%}  traps={ev['catTraps']:.2f}/{ev['mouseTraps']:.2f}")
        elif ev["kind"] == "bestFinal":
            line(f"[{school:5}] best {ev['role']:5} = {ev['pick'].upper():5} "
                 f"{ev['rate']:6.1%} (lo {ev['lo']:.1%}) @ {human_steps(ev['fromSteps'])} steps")

    s = build(school, maps, emaps, dev, budget, seed, on_event, out, envs)
    # SIGTERM is "wrap it up", not "die". A 500M-step run stopped from the app — or from
    # the terminal with ^C — must still snapshot, still hold its peak-versus-finish
    # run-off and still write its checkpoints; killing it outright would throw away
    # however many hours it had already done.
    def wrap_up(signum, frame):
        print(f"\r[{school:5}] stopping at the next iteration…".ljust(108), flush=True)
        s.request_stop()
    signal.signal(signal.SIGTERM, wrap_up)
    signal.signal(signal.SIGINT, wrap_up)
    print(f"[{school:5}] budget {budget.describe()}", flush=True)
    final = s.train(eval_every=0.04)
    log.close()
    dt = time.time() - t0
    print(f"[{school:5}] DONE {dt / 60:.1f} min  iters={s.run.iters}  "
          f"envSteps={human_steps(s.run.steps)}  "
          f"({round(s.run.steps / max(1e-9, s._train_wall)) / 1e3:.1f}k steps/s)  "
          f"cat={final['catExam']:.1%} mouse={final['mouseExam']:.1%}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--school", default="all", choices=("all",) + SCHOOLS)
    ap.add_argument("--minutes", type=float, default=None,
                    help="wall-clock budget per school; defaults to 30 unless --steps is given")
    ap.add_argument("--steps", default=None,
                    help="environment-step budget per school: 500M, 1.5B, 2e8, 500000000")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--envs", type=int, default=None,
                    help="PPO only: parallel environments per rollout (default 512). "
                         "2048 is ~35%% more steps/second on an M2 Max")
    ap.add_argument("--nests", default=str(2),
                    help="holes per room: '2', or a mix like '1,2,3' across the level set")
    ap.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()

    from catmouse.school import Budget, parse_steps
    steps = parse_steps(a.steps)
    # Neither clock set means the old default. Asking for steps and NOT for minutes means
    # exactly that: run until the steps are spent, however long it takes.
    minutes = a.minutes if a.minutes is not None else (None if steps else 30.0)
    budget = Budget(seconds=None if minutes is None else minutes * 60, steps=steps)

    out = ROOT / "runs" / a.tag
    if a._child:
        run_one(a._child, minutes, steps, a.seed, out, a.threads, a.device, a.nests, a.envs)
        return
    if a.school != "all":
        run_one(a.school, minutes, steps, a.seed, out, a.threads, a.device, a.nests, a.envs)
        return

    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "minutes": minutes, "steps": steps, "budget": budget.describe(),
        "seed": a.seed, "schools": list(SCHOOLS), "nests": a.nests, "envs": a.envs,
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    print(f"training {', '.join(SCHOOLS)} in parallel, {budget.describe()} each, "
          f"{a.nests} hole(s) per room -> runs/{a.tag}", flush=True)
    child = ["--seed", str(a.seed), "--tag", a.tag, "--device", a.device,
             "--threads", str(a.threads), "--nests", str(a.nests)]
    if a.envs:
        child += ["--envs", str(a.envs)]
    if minutes is not None:
        child += ["--minutes", str(minutes)]
    if steps:
        child += ["--steps", str(steps)]
    procs = [
        subprocess.Popen([sys.executable, __file__, "--_child", s] + child)
        for s in SCHOOLS
    ]

    # The parent is a supervisor, so it must not die on the signal that is meant to end
    # the run: it passes it down and then waits for all three to save. Without this,
    # stopping a run from the app killed the parent and orphaned three children that
    # went on training with nobody reading them.
    def forward(signum, frame):
        print("stopping all three at the next iteration…", flush=True)
        for pr in procs:
            with_suppressed(pr.terminate)
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    codes = [p.wait() for p in procs]
    print("exit codes:", dict(zip(SCHOOLS, codes)), flush=True)


if __name__ == "__main__":
    main()
