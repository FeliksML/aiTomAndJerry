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
    {"cmd":"trainAll","steps":"500M","envs":2048,"tag":"v5"}  the real three-school run
    {"cmd":"stopAll"}                                   ... ended early, but saved
    {"cmd":"score","tag":"v5","checkpoint":"best"}      tournament + highlight scan
    {"cmd":"useRun","tag":"v5"}                         watch a different run
    {"cmd":"train","school":"ga","minutes":10,"shaping":{...},"hyper":{...}}
                                                        one academy, its own knobs
    {"cmd":"pin","at":12}   {"cmd":"pin","at":null}     scrub the run / back to live
    {"cmd":"shadow"}                                    back to a live run's arena
    {"cmd":"popRace","role":"cat"}                      best vs worst of one generation
    {"cmd":"train","school":"ga","minutes":null,"steps":"500M"}   ... to a step budget
    {"cmd":"speed","value":4}  {"cmd":"pause"}  {"cmd":"resume"}
    {"cmd":"skip"}                                      finish this episode instantly
    {"cmd":"next"}                                      jump to the next arena
    {"cmd":"reset"}                                     drop every weight back to a random init
    {"cmd":"stop"}                                      idle; a live run finishes cleanly

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

from catmouse.school import parse_steps      # noqa: E402  (needs the path above)

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
            msg = session._train_events.get_nowait()
            await send(msg)
            # The scan rewrites tournament.json / highlights.json, and switching runs
            # replaces the whole catalogue. Push the new greeting rather than making the
            # app guess when to ask for it.
            if msg.get("type") in ("scoreDone", "trainAllDone"):
                await send(session.hello())

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
            elif cmd == "trainAll":
                await send(session.start_train_all(
                    asyncio.get_running_loop(),
                    steps=parse_steps(m.get("steps")),
                    minutes=m.get("minutes") or None,
                    envs=m.get("envs") or None,
                    seed=int(m.get("seed", 7)),
                    tag=m.get("tag", "v5"),
                    nests=str(m.get("nests", "2")),
                    device=m.get("device", "auto")))
            elif cmd == "stopAll":
                await send(session.stop_train_all())
            elif cmd == "score":
                await send(session.start_score(asyncio.get_running_loop(),
                                               m.get("tag"),
                                               m.get("checkpoint", "trained")))
            elif cmd == "useRun":
                tag = str(m.get("tag", "")).strip()
                target = (ROOT / "runs" / tag) if tag else None
                busy = session.busy_reason()
                if busy:
                    # Switching runs rebinds the arenas and the policies under whatever is
                    # training, and the take would then save into the run it was moved to.
                    await send({"type": "error", "message": busy})
                elif not target or not target.exists():
                    await send({"type": "error", "message": f"no run called {tag!r}"})
                else:
                    session.mode = "idle"
                    session.load_run_dir(target)
                    await send({"type": "runSwitched", "tag": tag})
                    await send(session.hello())
            elif cmd == "train":
                # A live take can be budgeted in minutes, in environment steps, or both.
                # `minutes: null` with a step budget is the overnight form.
                mins = m.get("minutes", 5)
                await send(session.start_train(m.get("school", "ppo"),
                                               None if mins is None else float(mins),
                                               int(m.get("seed", 11)),
                                               steps=parse_steps(m.get("steps")),
                                               shaping=m.get("shaping"),
                                               hyper=m.get("hyper")))
            elif cmd == "popRace":
                # Six of one generation in twelve identical rooms: the best three and the
                # three that were about to be replaced, same opponent, same dice.
                await send(session.start_pop_race(m.get("role", "cat")))
            elif cmd == "shadow":
                # Back to the training arena of a run that is still going, without
                # touching the optimiser. A checkpoint pill moves the session to `play`,
                # and that used to end scrubbing for the rest of the run — this is the
                # way back, and the reel sends it itself before a pin.
                await send(session.shadow())
            elif cmd == "pin":
                # Scrub the run: play the arena on the weights it had at one evaluation,
                # or `null` to go back to whatever the optimiser holds now.
                await send(session.pin(m.get("at"), m.get("school")))
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
                if env is None:
                    await send({"type": "error", "message": "nothing is playing to skip"})
                else:
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
                # Ends the ON-CAMERA take. Three things this deliberately does NOT do:
                #
                #  * it does not touch the three-school run — `stop` is what the arena's
                #    own button and shift+S send, and one keystroke meant for a take must
                #    not end an hour-long background run. That is `stopAll`.
                #  * it does not idle the arena while a take is wrapping up. Stopping is
                #    a request honoured at the next iteration boundary, and the tail after
                #    it (final scoring, the peak-versus-finish run-off) is tens of seconds
                #    more. Freezing the arena at the key press made every stop look like a
                #    crash, seconds before anything had actually stopped.
                #  * it does not stay silent when there is nothing to stop.
                sch = session.train_school
                if sch is not None:
                    sch.request_stop()
                    await send({"type": "trainStopping", "school": sch.key})
                elif session.runner is not None and session.runner.poll() is None:
                    await send({"type": "error", "message":
                                "that stops a live take — the three-school run is ended "
                                "with STOP on the RUNS screen"})
                else:
                    await send({"type": "error", "message": "nothing is training"})
                    session.mode = "idle"
                await send({"type": "state", **session.state()})
            else:
                # No `else` at all meant a typo, a stale client and a dropped packet were
                # the same event: nothing happened and nothing said so.
                await ws.send(json.dumps({"type": "error",
                                          "message": f"unknown command {cmd!r}"}))
    finally:
        clients.discard(ws)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/latest")
    ap.add_argument("--replay", default=None)
    ap.add_argument("--port", type=int, default=8765)
    # "localhost" binds both 127.0.0.1 and ::1. Binding only the IPv4 address meant a
    # browser that resolved `localhost` to ::1 first — which Chrome does, and not
    # predictably — could not reach a server that was plainly running.
    ap.add_argument("--host", default="localhost")
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

            async with websockets.serve(serve_replay, a.host, a.port,
                                        max_size=None, compression=None):
                print(f"replaying {path} on ws://{a.host}:{a.port}", flush=True)
                await asyncio.Future()
            return

        dev = nets.pick_device(a.device)
        run_dir = ROOT / a.run if not Path(a.run).is_absolute() else Path(a.run)
        stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        journal = None if a.no_journal else Journal(ROOT / "runs" / "journals" / f"{stamp}.jsonl")
        session = Session(run_dir, dev, journal)
        print(f"serving {run_dir} on ws://{a.host}:{a.port}  (device {dev})", flush=True)
        # A missing or empty run directory is tolerated on purpose — you can serve one and
        # train into it from the app — but silently is the wrong way to tolerate it. The
        # symptom otherwise is a green TRAINER LIVE chip over three greyed-out schools.
        have = sorted(session.policies["trained"].keys())
        if not have:
            print(f"  !! no checkpoints under {run_dir} — every school will read "
                  f"NOT IN THIS RUN until one is trained", flush=True)
        else:
            print(f"  schools present: {', '.join(have)}", flush=True)
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

        async def keep_clock() -> None:
            """The pump is the only thing that steps the arena and the only thing that
            drains training telemetry. It used to be created once: one exception and the
            app stayed connected, still answering commands, with an arena that never moved
            again and a run whose events piled up unseen."""
            while not stop.is_set():
                try:
                    await pump(session, broadcast, stop)
                    return                      # asked to stop
                except asyncio.CancelledError:
                    raise
                except Exception:
                    with contextlib.suppress(Exception):
                        await broadcast({"type": "error",
                                         "message": "the simulation clock restarted"})
                    await asyncio.sleep(0.5)

        clock = asyncio.create_task(keep_clock())
        try:
            async with websockets.serve(
                    lambda ws: handler(ws, session, journal, clients, broadcast),
                    a.host, a.port, max_size=None, compression=None):
                await asyncio.Future()
        finally:
            stop.set()
            clock.cancel()

    asyncio.run(run())


if __name__ == "__main__":
    main()
