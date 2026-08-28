#!/usr/bin/env python3
"""Five AI walk clips of Tom -> one 8-direction walking sprite sheet.

    python3 tools/build_cat_sprites.py --analyze    # measure the gait, write the config
    python3 tools/build_cat_sprites.py              # build assets from the config
    python3 tools/build_cat_sprites.py --all        # both

The clips are immutable source material and are never written to. Everything the build
produces lands under build/ (scratch) and app/assets/cat/ (the shipped assets).

Only five directions were generated: DOWN, UP, LEFT, DOWN_LEFT and UP_LEFT. The three
right-facing directions are exact horizontal mirrors of their left-facing partners, on
purpose — a sixth and seventh generated clip would be a different cat.

`--analyze` measures each clip and writes cat-animation.config.json. The build reads only
that file, so the chosen frames are data, not magic numbers buried in code, and a pose
that reads badly can be nudged by editing one integer and rebuilding.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from catlib import gait                     # noqa: E402
from catlib import sheet as SH               # noqa: E402
from catlib.cache import alpha_cache, frame_paths, DIRS5   # noqa: E402
from catlib.dedupe import hold_factor, keyframes           # noqa: E402
from catlib.frames import geometry, quality, height_trend, place  # noqa: E402
from catlib.matte import cutout             # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / 'source-animation' / 'cat' / 'walk'
BUILD = ROOT / 'build'
OUT = ROOT / 'app' / 'assets' / 'cat'
CONFIG = ROOT / 'cat-walk.config.json'

# Sheet layout, mirroring and naming are shared with the mouse — see catlib/sheet.py.
ROWS, MIRROR, CAMEL = SH.ROWS, SH.MIRROR, SH.CAMEL

FPS_SRC = 24.0


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ frame extraction

def probe(path: pathlib.Path) -> dict:
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-of', 'json',
         '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration', str(path)],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)['streams'][0]
    num, den = (int(v) for v in s['r_frame_rate'].split('/'))
    return dict(width=int(s['width']), height=int(s['height']), fps=num / den,
                frames=int(s['nb_frames']), duration=float(s['duration']))


def extract(d: str) -> int:
    dest = BUILD / 'frames-raw' / d
    dest.mkdir(parents=True, exist_ok=True)
    have = len(list(dest.glob('*.png')))
    meta = probe(SRC / (d + '.mp4'))
    if have == meta['frames']:
        return have
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(SRC / (d + '.mp4')),
                    '-fps_mode', 'passthrough', '-start_number', '0',
                    str(dest / '%03d.png')], check=True)
    return len(list(dest.glob('*.png')))


# ------------------------------------------------------------------ analysis

def analyse(prior: dict) -> dict:
    cfg = {
        '_note': ('Written by tools/build_cat_sprites.py --analyze, then safe to hand-edit. '
                  '"frames" are the source frame indices actually used; "timestamps" are the '
                  'same instants in seconds and are what the build prints for review. '
                  'F1 contact A, F2 passing A, F3 contact B, F4 passing B.'),
        'frameSize': 256, 'fps': 8, 'charHeight': 0.80, 'groundY': 0.90, 'anchorX': 0.5,
        'source': {d: d + '.mp4' for d in DIRS5},
        # Hand-set overrides, honoured by --analyze so a corrected reading survives a
        # re-run. Empty means "use whatever the detector measures".
        'strideOverride': prior.get('strideOverride', {}),
        'poseOverride': prior.get('poseOverride', {}),
        'stride': {}, 'frames': {}, 'timestamps': {}, 'scale': {}, 'idle': {},
        'measured': {},
    }
    heights = {}
    for d in DIRS5:
        n = extract(d)
        meta = probe(SRC / (d + '.mp4'))
        log('%-10s %dx%d  %.2fs  %d frames @ %.0f fps' %
            (d, meta['width'], meta['height'], meta['duration'], n, meta['fps']))

        A = alpha_cache(ROOT, d, log=None).astype(np.float32) / 255.0
        paths = frame_paths(ROOT, d)
        rgbs = [np.asarray(Image.open(p).convert('RGB')).astype(np.float32) for p in paths]

        # The clip is 8 fps of animation in a 24 fps container. Collapse the held copies
        # or every measurement below is taken against two duplicates of itself.
        hold, phase, _ = hold_factor((A * 255).astype(np.uint8))
        kf = keyframes((A * 255).astype(np.uint8), rgbs)
        log('   %d fps of drawn animation (each pose held %d frames) -> %d poses'
            % (round(meta['fps'] / hold), hold, len(kf)))

        kA = A[kf]
        kR = [rgbs[i] for i in kf]
        geos = [geometry(a) for a in kA]
        feats = [gait.features(kA[i], geos[i]) for i in range(len(kA))]
        q = quality(kA, kR, geos)

        pick = gait.choose_cycle(feats, q, cfg['strideOverride'].get(CAMEL[d]))
        pinned = cfg['poseOverride'].get(CAMEL[d])
        if pinned:
            pose_idx = [int(i) for i in pinned]
        else:
            pose_idx = [gait.refine(feats, q, i, pick['period']) for i in pick['idx']]
        idx = [kf[i] for i in pose_idx]

        hs = np.array([g['h'] for g in geos], float)
        trend = height_trend(hs, pick['period'])
        heights[d] = float(np.median(hs / trend))

        secs = pick['period'] * hold / meta['fps']
        cfg['stride'][CAMEL[d]] = int(pick['period'])
        cfg['frames'][CAMEL[d]] = [int(i) for i in idx]
        cfg['timestamps'][CAMEL[d]] = [round(i / meta['fps'], 3) for i in idx]
        cfg['scale'][CAMEL[d]] = 1.0
        cfg['measured'][CAMEL[d]] = dict(
            videoFrames=n, drawnPoses=len(kf), holdFrames=hold, poses=[int(i) for i in pose_idx],
            strideSeconds=round(secs, 3),
            spread=[round(float(feats[i]['spread']), 3) for i in pose_idx],
            quality=[round(float(q[i]), 3) for i in pose_idx],
            heightDriftPct=round(100 * (trend[-1] / trend[0] - 1), 2),
            medianHeightPx=round(float(np.median(trend)), 1))
        log('   stride %2d poses (%.2fs)  ->  poses %s = frames %s   spread %s   q %s' % (
            pick['period'], secs, pose_idx, idx,
            cfg['measured'][CAMEL[d]]['spread'], cfg['measured'][CAMEL[d]]['quality']))

    # Character size needs no per-clip factor. Dividing each frame by its own clip's
    # height trend already states its size as a fraction of that clip's framing, so all
    # five land on the same rendered height however differently they were shot. `scale`
    # stays in the config as a hand-tuning knob for a direction that still reads wrong.
    for d in DIRS5:
        cfg['scale'][CAMEL[d]] = prior.get('scale', {}).get(CAMEL[d], 1.0)
    log('relative clip heights %s (scale kept at %s)' %
        ({k: round(heights[k], 3) for k in DIRS5},
         {k: cfg['scale'][CAMEL[k]] for k in DIRS5}))
    return cfg


# ------------------------------------------------------------------ build

def build(cfg: dict) -> dict:
    size = int(cfg['frameSize'])
    OUT.mkdir(parents=True, exist_ok=True)
    (BUILD / 'qa').mkdir(parents=True, exist_ok=True)

    canvases: dict[str, list[np.ndarray]] = {}
    report = {}
    for d in DIRS5:
        extract(d)
        key = CAMEL[d]
        idx = cfg['frames'][key]
        paths = frame_paths(ROOT, d)
        A = alpha_cache(ROOT, d, log=None).astype(np.float32) / 255.0
        kf = keyframes((A * 255).astype(np.uint8))
        hs = np.array([geometry(A[i])['h'] for i in kf], float)
        tr = height_trend(hs, cfg['stride'][key])
        trend = np.interp(np.arange(len(A)), kf, tr)
        ref = float(np.median(tr))

        # Character height in the finished frame. `scale` equalises the eight directions
        # against each other; the per-frame `ref/trend[i]` term undoes the camera dolly.
        base = cfg['charHeight'] * size / ref * cfg['scale'][key]

        frames, notes = [], []
        for i in idx:
            rgb, alpha = cutout(np.asarray(Image.open(paths[i]).convert('RGB')))
            geo = geometry(alpha)
            s = base * (ref / trend[i])          # undo the camera dolly at this instant
            frames.append(place(rgb, alpha, geo, s, size, cfg['anchorX'], cfg['groundY']))
            notes.append(dict(frame=int(i), scale=round(s, 4),
                              groundPx=geo['ground'], torsoCx=round(geo['torso_cx'], 1),
                              heightPx=geo['h']))
        canvases[d] = frames
        report[key] = notes
        log('%-10s frames %s  scale %.3f' % (d, idx, base))

    SH.mirror_all(canvases)
    for dst, src in MIRROR.items():
        report[CAMEL[dst]] = {'mirrorOf': CAMEL[src]}

    meta = SH.write(canvases, OUT, 'tom-walk.png', 'tom-walk.json', size, cfg['fps'],
                    {'x': cfg['anchorX'], 'y': cfg['groundY']}, cfg['idle'],
                    {'videos': cfg['source'], 'config': CONFIG.name})
    (BUILD / 'qa').mkdir(parents=True, exist_ok=True)
    (BUILD / 'qa' / 'build-report.json').write_text(json.dumps(report, indent=2) + '\n')
    log('sheet %dx%d -> %s' % (size * 4, size * len(ROWS), OUT / meta['image']))
    return dict(canvases=canvases, meta=meta, size=size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analyze', action='store_true', help='measure the clips, write the config')
    ap.add_argument('--all', action='store_true', help='analyse then build')
    a = ap.parse_args()

    prior = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    if a.analyze or a.all or not CONFIG.exists():
        cfg = analyse(prior)
        if CONFIG.exists():                       # keep hand-edits to tunables
            for k in ('frameSize', 'fps', 'charHeight', 'groundY', 'anchorX', 'idle',
                      'strideOverride', 'poseOverride', 'scale'):
                if k in prior:
                    cfg[k] = prior[k]
        CONFIG.write_text(json.dumps(cfg, indent=2) + '\n')
        log('wrote %s' % CONFIG)
        if a.analyze and not a.all:
            return

    cfg = json.loads(CONFIG.read_text())
    res = build(cfg)
    SH.qa(res['canvases'], BUILD / 'qa', res['size'], cfg['fps'])
    log('QA written to %s' % (BUILD / 'qa'))


if __name__ == '__main__':
    main()
