"""Train the schools.

    python trainer/scripts/train.py --minutes 30                 # all three, in parallel
    python trainer/scripts/train.py --school ppo --minutes 5     # just one

All three run at once in separate processes. That is not only faster — it is the
fairness protocol: identical wall-clock, on the same machine, under the same load.
Each process is pinned to a slice of the cores so one school cannot starve another.

Every run writes, under `runs/<tag>/<school>/`:
    checkpoints.npz   the three Academy checkpoints, both roles, as flat weight vectors
    history.json      the Examiner curve against both budget clocks
    events.jsonl      every telemetry event, including the algorithm internals the
                      on-screen explainers draw. This is also what a TRAIN-mode replay
                      is served from.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "trainer"))

SCHOOLS = ("ppo", "ga", "cmaes")


def build(school: str, maps, emaps, device, budget, seed, on_event, out):
    from catmouse.cmaes import CMAESSchool
    from catmouse.ga import GASchool
    from catmouse.ppo import PPOSchool
    cls = {"ppo": PPOSchool, "ga": GASchool, "cmaes": CMAESSchool}[school]
    return cls(maps, emaps, device, budget, seed=seed, on_event=on_event, out_dir=out)


def run_one(school: str, minutes: float, seed: int, out: Path, threads: int, device: str) -> None:
    import torch
    torch.set_num_threads(threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))

    from catmouse import arena, nets, vec
    from catmouse.school import Budget

    dev = nets.pick_device(device)
    maps = vec.MapSet(arena.TRAIN_SEEDS)
    emaps = vec.MapSet(arena.EVAL_SEEDS)
    out.mkdir(parents=True, exist_ok=True)
    (out / school).mkdir(parents=True, exist_ok=True)
    log = (out / school / "events.jsonl").open("w")

    t0 = time.time()

    def on_event(ev: dict) -> None:
        log.write(json.dumps(ev) + "\n")
        if ev["kind"] == "eval":
            print(f"[{school:5}] {ev['frac']:5.0%} it={ev['iter']:5d} "
                  f"steps={ev['steps'] / 1e6:6.2f}M  cat={ev['catExam']:6.1%} "
                  f"mouse={ev['mouseExam']:6.1%}  traps={ev['catTraps']:.2f}/{ev['mouseTraps']:.2f}",
                  flush=True)

    s = build(school, maps, emaps, dev, Budget(seconds=minutes * 60), seed, on_event, out)
    final = s.train(eval_every=0.04)
    log.close()
    dt = time.time() - t0
    print(f"[{school:5}] DONE {dt / 60:.1f} min  iters={s.run.iters}  "
          f"envSteps={s.run.steps / 1e6:.1f}M  "
          f"cat={final['catExam']:.1%} mouse={final['mouseExam']:.1%}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--school", default="all", choices=("all",) + SCHOOLS)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()

    out = ROOT / "runs" / a.tag
    if a._child:
        run_one(a._child, a.minutes, a.seed, out, a.threads, a.device)
        return
    if a.school != "all":
        run_one(a.school, a.minutes, a.seed, out, a.threads, a.device)
        return

    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "minutes": a.minutes, "seed": a.seed, "schools": list(SCHOOLS),
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    print(f"training {', '.join(SCHOOLS)} in parallel for {a.minutes:.0f} min each "
          f"-> runs/{a.tag}", flush=True)
    procs = [
        subprocess.Popen(
            [sys.executable, __file__, "--_child", s, "--minutes", str(a.minutes),
             "--seed", str(a.seed), "--tag", a.tag, "--device", a.device,
             "--threads", str(a.threads)])
        for s in SCHOOLS
    ]
    codes = [p.wait() for p in procs]
    print("exit codes:", dict(zip(SCHOOLS, codes)), flush=True)


if __name__ == "__main__":
    main()
