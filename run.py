#!/usr/bin/env python3
"""Camera-based localization: raw mcap -> submission CSV (timestamp_ns, x, y, yaw), causal, online.

    python run.py BAG out/submission.csv [--maps maps] [--cams fl] [--init-pose x,y,yaw]

1. One pass over the bag: camera calibration, track borders, car state, image stamps -> <work>/.
   The initial pose is the first /tf map -> base_link of the run (allowed: first pose), or --init-pose.
2. For every image set (fl stamps, cameras are hardware-synced), in time order:
   predict with car-state odometry up to the stamp -> track-map update -> relocalize the image(s)
   against the keyframe map near the predicted pose (ALIKED + LightGlue + PnP) -> fix update -> output.
   A pose at stamp t uses only images and state with stamps <= t.
Per-frame processing time is measured and written next to the CSV.
"""
import argparse, json, os, subprocess, sys, time
import cv2
import numpy as np
import pandas as pd
import torch
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from camloc.camera import load_cameras
from camloc.body_masks import body_masks
from camloc.trackmap import TrackMap
from camloc.motion import increments
from camloc.pf import ParticleFilter, PFParams
from camloc.features import Features
from camloc.kfmap import KeyframeMap
from camloc.augment import TRANSFORMS

NS = 'constructor0'


def first_pose(bag):
    with open(bag, 'rb') as f:
        for _, _, _, msg in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(topics=['/tf']):
            for t in msg.transforms:
                if t.header.frame_id == 'map' and t.child_frame_id == f'{NS}/base_link':
                    q = t.transform.rotation
                    yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                    return t.header.stamp.sec * 10**9 + t.header.stamp.nanosec, t.transform.translation.x, \
                        t.transform.translation.y, yaw
    raise RuntimeError('no /tf map -> base_link in the bag; pass --init-pose')


SET_TOL_NS = 5_000_000      # images of different cameras within 5 ms form one image set


def image_sets(frames, cams):
    """Group image stamps of `cams` into sets around the first camera's stamps.

    The cameras are hardware-synced but their stamps differ by tens of microseconds. Returns
    member {(cam, stamp): set_id}, set_stamp {set_id: max member stamp} (causal output stamp), size {set_id: n}.
    """
    ref = np.sort(frames[frames.cam == cams[0]].timestamp_ns.values)
    member = {(cams[0], int(t)): int(t) for t in ref}
    for c in cams[1:]:
        ts = np.sort(frames[frames.cam == c].timestamp_ns.values)
        i = np.clip(np.searchsorted(ref, ts), 1, len(ref) - 1)
        j = np.where(np.abs(ref[i] - ts) < np.abs(ref[i - 1] - ts), i, i - 1)
        for t, r in zip(ts, ref[j]):
            if abs(int(t) - int(r)) <= SET_TOL_NS:
                member[(c, int(t))] = int(r)
    set_stamp, size = {}, {}
    for (c, t), s in member.items():
        set_stamp[s] = max(set_stamp.get(s, t), t)
        size[s] = size.get(s, 0) + 1
    return member, set_stamp, size


def images(bag, cams, member, size, t_start=None, t_end=None):
    """Yield (set_id, {cam: jpeg}) once per image set, when all its images have arrived (or it is stale)."""
    topics = {f'/{NS}/sensor/camera_{c}/compressed_image': c for c in cams}
    pending, done = {}, set()
    window = {} if t_start is None else dict(start_time=t_start - 10**9, end_time=t_end + 10**9)
    with open(bag, 'rb') as f:
        for i, (_, ch, _, msg) in enumerate(make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                topics=list(topics), **window)):
            if i % 2000 == 0 and hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            c = topics[ch.topic]
            s = member.get((c, msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec))
            if s is None or s in done:
                continue
            pending.setdefault(s, {})[c] = bytes(msg.data)
            ready = [k for k in pending if len(pending[k]) == size[k] or k < s - 300_000_000]
            for k in sorted(ready):
                done.add(k)
                yield k, pending.pop(k)
    for k in sorted(pending):
        yield k, pending.pop(k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag'); ap.add_argument('out')
    ap.add_argument('--maps', default=os.path.join(HERE, 'maps'))
    ap.add_argument('--cams', default='fl')
    ap.add_argument('--work', default=None, help='scratch dir for the exported bag data (default: next to out)')
    ap.add_argument('--init-pose', default=None, help='x,y,yaw; default: first /tf of the bag')
    ap.add_argument('--n-kf', type=int, default=2)
    ap.add_argument('--min-inl', type=int, default=15)
    ap.add_argument('--reproj', type=float, default=3.5, help='PnP reprojection error threshold')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pf-config', default=None, help='JSON with {"params": {...}} overriding PFParams defaults')
    ap.add_argument('--roma-fallback', type=int, default=0, help='RoMa relocalization when LightGlue gives no fix')
    ap.add_argument('--roma-cam', default='fl')
    ap.add_argument('--roma-period', type=float, default=0.5, help='s, at most one RoMa call per period')
    ap.add_argument('--roma-lat', type=float, default=0.3, help='m, sigma of RoMa fixes across the car')
    ap.add_argument('--roma-lon', type=float, default=1.2, help='m, sigma of RoMa fixes along the car')
    ap.add_argument('--roma-gate', type=float, default=10.0, help='m, RoMa fixes farther than this are ignored')
    ap.add_argument('--roma-cert', type=float, default=0.05, help='RoMa certainty threshold for the sampled matches')
    # robustness tests only (not used for the submission)
    ap.add_argument('--transform', default='none', help='synthetic appearance change, see camloc/augment.py')
    ap.add_argument('--t-start', type=int, default=None, help='ns; run only from here (needs --init-from)')
    ap.add_argument('--t-end', type=int, default=None, help='ns; stop here')
    ap.add_argument('--init-from', default=None, help='CSV of an earlier causal run: initial pose at --t-start')
    a = ap.parse_args()
    cams = a.cams.split(',')
    work = a.work or os.path.splitext(a.out)[0] + '_work'

    # ---- 1. bag -> calibration, borders, state, stamps
    if not os.path.exists(f'{work}/state.csv'):
        subprocess.run([sys.executable, os.path.join(HERE, 'tools', 'export_cache.py'), a.bag, work,
                        '--img-every', '0'], check=True)
    if a.init_from:
        prev = pd.read_csv(a.init_from)
        t0 = a.t_start
        x0, y0 = (float(np.interp(t0, prev.timestamp_ns, prev[k])) for k in ('x', 'y'))
        yaw0 = float(np.interp(t0, prev.timestamp_ns, np.unwrap(prev['yaw'].values)))
    elif a.init_pose:
        x0, y0, yaw0 = map(float, a.init_pose.split(','))
        t0 = None
    else:
        t0, x0, y0, yaw0 = first_pose(a.bag)
    calib = load_cameras(f'{work}/calib.json')
    tm = TrackMap(f'{work}/track_bounds.npz')
    fr = pd.read_csv(f'{work}/frames.csv')
    member, set_stamp, size = image_sets(fr, cams)
    img_t = np.array(sorted(set_stamp.values()))
    t0 = t0 if t0 is not None else int(img_t[0]) - 1
    img_t = img_t[(img_t > t0) & (img_t <= (a.t_end or img_t[-1]))]
    inc = increments(f'{work}/state.csv', img_t, t0).set_index('timestamp_ns')

    feats = Features()
    masks = body_masks()
    kfmaps = {c: KeyframeMap(f'{a.maps}/kf_{c}.npz', calib[c]) for c in cams}
    params = PFParams()
    if a.pf_config:
        with open(a.pf_config) as f:
            for k, v in json.load(f)['params'].items():
                setattr(params, k, v)
    pf = ParticleFilter(tm, (x0, y0, yaw0), params, a.seed)
    roma, last_roma = None, -10**19
    if a.roma_fallback:
        from camloc.roma_fallback import RomaFallback
        roma = RomaFallback(kfmaps[a.roma_cam], f'{a.maps}/kf_{a.roma_cam}_img', masks[a.roma_cam],
                            cert_thr=a.roma_cert)

    # ---- 2. online loop
    rows, log = [], []
    transform = TRANSFORMS[a.transform]
    for sid, jpegs in images(a.bag, cams, member, size, a.t_start, a.t_end):
        t = set_stamp[sid]                       # output stamp: the latest image of the set (causal)
        if t not in inc.index:
            continue
        torch.cuda.synchronize(); tic = time.perf_counter()
        dx, dy, dyaw, dt, dist = inc.loc[t, ['dx', 'dy', 'dyaw', 'dt', 'dist']]
        pf.predict(dx, dy, dyaw, dt)
        if dist > 0.05:
            pf.update_map()
        n_used, grays = 0, {}
        for c, jpeg in jpegs.items():
            gray = grays[c] = transform(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE))
            r = kfmaps[c].localize(feats, feats.extract(gray, masks[c]), pf.estimate(),
                                   n_kf=a.n_kf, reproj=a.reproj, min_inliers=a.min_inl)
            if r['ok'] and r['n_inl'] >= a.min_inl and pf.update_fix(r['x'], r['y'], r['yaw']):
                n_used += 1
            log.append(dict(timestamp_ns=t, cam=c, ok=int(r['ok']), n_inl=r['n_inl'], x=r['x'], y=r['y'], yaw=r['yaw']))
        if roma is not None and n_used == 0 and a.roma_cam in grays and t - last_roma >= a.roma_period * 1e9:
            last_roma = t
            r = roma.localize(grays[a.roma_cam], pf.estimate())
            used = r['ok'] and pf.update_fix(r['x'], r['y'], r['yaw'], a.roma_lat, a.roma_lon, a.roma_gate)
            n_used += int(bool(used))
            log.append(dict(timestamp_ns=t, cam=a.roma_cam + '_roma', ok=int(r['ok']), n_inl=r['n_inl'],
                            x=r['x'], y=r['y'], yaw=r['yaw']))
        rows.append((t,) + pf.estimate())
        torch.cuda.synchronize()
        log[-1]['ms'] = (time.perf_counter() - tic) * 1e3
        log[-1]['used'] = n_used
        if len(rows) % 1000 == 0:
            ms = [l['ms'] for l in log if 'ms' in l]
            print(f'{len(rows)} poses, {np.mean(ms[-1000:]):.0f} ms/frame', flush=True)

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    pd.DataFrame(rows, columns=['timestamp_ns', 'x', 'y', 'yaw']).to_csv(a.out, index=False, float_format='%.6f')
    lg = pd.DataFrame(log)
    lg.to_csv(os.path.splitext(a.out)[0] + '_log.csv', index=False, float_format='%.6f')
    ms = lg.ms.dropna()
    timing = dict(frames=len(rows), mean_ms=float(ms.mean()), median_ms=float(ms.median()), p95_ms=float(ms.quantile(0.95)),
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none', cams=cams)
    with open(os.path.splitext(a.out)[0] + '_timing.json', 'w') as o:
        json.dump(timing, o, indent=1)
    print(f'{len(rows)} poses -> {a.out}   {timing["mean_ms"]:.0f} ms/frame mean ({timing["gpu"]})')


if __name__ == '__main__':
    main()
