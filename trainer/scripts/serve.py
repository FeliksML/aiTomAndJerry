"""WebSocket front door for the recording app.

    python trainer/scripts/serve.py --run runs/latest
    python trainer/scripts/serve.py --replay runs/journals/<file>.jsonl

Commands the app sends:

    {"cmd":"hello"}                                     catalogue: schools, checkpoints,
                                                        level set, tournament, budgets
    {"cmd":"play","school":"ppo","checkpoint":"trained","mouseSchool":"ga",
     "levels":[3,1],"seeds":[20263559,20261737]}   a reproducible pairing, arena by arena
    {"cmd":"play","school":"ppo","checkpoint":"trained",
     "opponent":"self|examiner-mouse|examiner-cat"}     run the shared level set
    {"cmd":"final"}                                     champion vs champion, 5 rounds
    {"cmd":"race","checkpoint":"trained"}               all three schools, same room, side by side
    {"cmd":"train","school":"ga","minutes":10}          train live, on camera
    {"cmd":"speed","value":4}  {"cmd":"pause"}  {"cmd":"resume"}
    {"cmd":"skip"}                                      finish this episode instantly
    {"cmd":"next"}                                      jump to the next arena
    {"cmd":"reset"}                                     drop every weight back to a random init\n    {"cmd":"stop"}

Every outgoing message is journalled first, so any take can be replayed frame for frame.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "trainer"))

BASE_STEPS_PER_SEC = 9.0     # matches the Academy's default pace


async def pump(session, send, stop: asyncio.Event) -> None:
    """The simulation clock. Steps at `speed x 9` per second and holds on a result
    long enough for the banner to read on camera.

    Wrapped so a failure is loud. An exception inside a bare `create_task` is swallowed
    until the task is awaited, which during a shoot looks like the arena quietly
    freezing with no explanation anywhere.
    """
    try:
        await _pump(session, send, stop)
    except asyncio.CancelledError:
        raise
    except Exception:
        traceback.print_exc()
        with contextlib.suppress(Exception):
            await send({"type": "error", "message": "the simulation clock stopped — see the server log"})
        raise


async def _pump(session, send, stop: asyncio.Event) -> None:
    acc = 0.0
    last = time.perf_counter()
    hold_until = 0.0
    while not stop.is_set():
        await asyncio.sleep(1 / 120)
        now = time.perf_counter()
        dt, last = now - last, now

        while not session._train_events.empty():
            await send(session._train_events.get_nowait())

        if session.mode not in ("play", "final", "train", "race") or not session.playing:
            continue
        env = session.env
        if env is None:
            continue
        if env.done.all():
            if hold_until == 0.0:
                hold_until = now + 0.9 / max(1.0, session.speed * 0.55)
                for m in (session.race_tick() if session.mode == "race" else session.tick()):
                    await send(m)
            elif now >= hold_until:
                hold_until = 0.0
                end = session.advance()
                if end:
                    await send(end)
            continue

        acc += dt * BASE_STEPS_PER_SEC * session.speed
        guard = 0
        while acc >= 1 and guard < 240 and not env.done.all():
            acc -= 1
            guard += 1
            for m in (session.race_tick() if session.mode == "race" else session.tick()):
                await send(m)
        if env.done.all():
            acc = 0.0


async def handler(ws, session, journal, clients, send):
    """One connection. The clock is NOT started here.

    There is one Session, so there must be exactly one pump driving it. A pump per
    connection meant a second window — a stale tab, a reload that left a zombie socket,
    a preview on another monitor — silently stole half the work: the simulation advanced
    once per pump, so it ran at N times the pace the speed chip claimed, and each window
    received only its own share of the frames and results. On screen that reads as
    episodes flying past and gaps in the level strip, with nothing to point at. Clients
    now share one clock, and every one of them sees every message.
    """
    clients.add(ws)
    try:
        # The greeting is the only per-connection message: it is a snapshot of the run,
        # and a joining window needs it without interrupting anybody else's.
        await ws.send(json.dumps(session.hello(), default=lambda o: o.tolist()))
        async for raw in ws:
            try:
                m = json.loads(raw)
            except Exception:
                continue
            cmd = m.get("cmd")
            if cmd == "hello":
                await ws.send(json.dumps(session.hello(), default=lambda o: o.tolist()))
            elif cmd == "play":
                await send(session.start_play(m.get("school", "ppo"),
                                              m.get("checkpoint", "trained"),
                                              m.get("opponent", "self"),
                                              m.get("levels"),
                                              m.get("mouseSchool"),
                                              m.get("seeds")))
            elif cmd == "race":
                await send(session.start_race(m.get("checkpoint", "trained"), m.get("levels")))
            elif cmd == "final":
                await send(session.start_final(int(m.get("rounds", 5))))
            elif cmd == "train":
                await send(session.start_train(m.get("school", "ppo"),
                                               float(m.get("minutes", 5)),
                                               int(m.get("seed", 11))))
            elif cmd == "speed":
                session.speed = max(0.25, float(m.get("value", 4)))
                await send({"type": "state", **session.state()})
            elif cmd in ("pause", "resume"):
                session.playing = cmd == "resume"
                await send({"type": "state", **session.state()})
            elif cmd == "skip":
                # Run the episode out silently, then send only its final frame and its
                # result. The result message is emitted by the tick that ENDS the
                # episode, so that tick's output is what has to be forwarded — a fresh
                # tick afterwards finds the result already recorded and stays quiet.
                env = session.env
                if env is not None:
                    tail: list[dict] = []
                    guard = 0
                    while not env.done[0] and guard < 400:
                        guard += 1
                        tail = session.tick()
                    for msg in (tail or session.tick()):
                        await send(msg)
            elif cmd == "next":
                end = session.advance()
                await send(end or {"type": "state", **session.state()})
            elif cmd == "reset":
                # Everyone watching should land in the same empty state at the same
                # moment, so this goes out to every client rather than just the one
                # that pressed the button.
                await send(session.reset_to_zero(int(m.get("seed", 0))))
                await send({"type": "state", **session.state()})
            elif cmd == "stop":
                session.mode = "idle"
                await send({"type": "state", **session.state()})
    finally:
        clients.discard(ws)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/latest")
    ap.add_argument("--replay", default=None)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-journal", action="store_true")
    a = ap.parse_args()

    from catmouse import nets
    from catmouse.server import Journal, Session, replay

    async def run() -> None:
        if a.replay:
            path = Path(a.replay)

            async def serve_replay(ws):
                async def send(msg):
                    await ws.send(json.dumps(msg))
                await replay(path, send)

            async with websockets.serve(serve_replay, "127.0.0.1", a.port,
                                        max_size=None, compression=None):
                print(f"replaying {path} on ws://127.0.0.1:{a.port}", flush=True)
                await asyncio.Future()
            return

        dev = nets.pick_device(a.device)
        run_dir = ROOT / a.run if not Path(a.run).is_absolute() else Path(a.run)
        stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        journal = None if a.no_journal else Journal(ROOT / "runs" / "journals" / f"{stamp}.jsonl")
        session = Session(run_dir, dev, journal)
        print(f"serving {run_dir} on ws://127.0.0.1:{a.port}  (device {dev})", flush=True)
        if journal:
            print(f"journalling to runs/journals/{stamp}.jsonl", flush=True)

        clients: set = set()

        async def broadcast(msg: dict) -> None:
            """Journal once, then fan out. A window that has gone away is dropped rather
            than left to stall the clock everyone else is on."""
            if journal:
                journal.write(msg)
            if not clients:
                return
            data = json.dumps(msg, default=lambda o: o.tolist())
            for sock in list(clients):
                try:
                    await sock.send(data)
                except Exception:
                    clients.discard(sock)

        stop = asyncio.Event()
        clock = asyncio.create_task(pump(session, broadcast, stop))
        try:
            async with websockets.serve(
                    lambda ws: handler(ws, session, journal, clients, broadcast),
                    "127.0.0.1", a.port, max_size=None, compression=None):
                await asyncio.Future()
        finally:
            stop.set()
            clock.cancel()

    asyncio.run(run())


if __name__ == "__main__":
    main()
