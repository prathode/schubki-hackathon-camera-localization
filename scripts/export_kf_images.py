#!/usr/bin/env python3
"""Save the keyframe images of a keyframe map (needed by the RoMa fallback, which matches whole images).

    python3 scripts/export_kf_images.py BAG cache gt.csv maps_no26/kf_fl.npz --cam fl

Keyframe stamps are recovered from the keyframe poses (each is the GT pose at its frame; training data).
Images are stored at half resolution as JPEG in <map>_img/<k>.jpg; RoMa works in normalized coordinates,
so the scale does not matter. The stamps are saved next to them (stamps.npy).
"""
import argparse, os
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag'); ap.add_argument('cache'); ap.add_argument('gt'); ap.add_argument('map')
    ap.add_argument('--cam', default='fl')
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--exclude-laps', default='', help='the laps left out of this map (same as build_kf_map)')
    a = ap.parse_args()
    kf = np.load(a.map)['kf_pose']
    g = pd.read_csv(a.gt)
    # search only the laps this map was built from (laps overlap in space; never pick another lap's frame)
    laps = pd.read_csv(f'{a.cache}/laps.csv')
    excl = [int(x) for x in a.exclude_laps.split(',') if x]
    use = laps[(laps.split == 'train') & ~laps.lap.isin(excl)]
    m = np.zeros(len(g), bool)
    for r in use.itertuples():
        m |= (g.timestamp_ns >= r.t0_ns) & (g.timestamp_ns <= r.t1_ns)
    g_all = g
    g = g[m].reset_index(drop=True)
    fr = pd.read_csv(f'{a.cache}/frames.csv')
    ts = np.sort(fr[fr.cam == a.cam].timestamp_ns.values)
    # keyframes are stored in time order: take the first GT sample within 0.3 m after the previous keyframe
    # (a position alone is ambiguous while the car stands still in the pit lane)
    tree, gt_t = cKDTree(np.c_[g.x, g.y]), g.timestamp_ns.values
    kf_t, prev = np.zeros(len(kf), np.int64), -1
    for k, (x, y, _) in enumerate(kf):
        cand = np.array(sorted(i for i in tree.query_ball_point([x, y], 0.3) if gt_t[i] > prev))
        if len(cand) == 0:
            raise RuntimeError(f'keyframe {k} has no GT match after the previous keyframe')
        tg = gt_t[cand[0]]
        j0 = int(np.argmin(np.abs(ts - tg)))
        js = [j for j in range(max(j0 - 3, 0), min(j0 + 4, len(ts))) if ts[j] > prev]
        err = [np.hypot(np.interp(ts[j], g_all.timestamp_ns, g_all.x) - x, np.interp(ts[j], g_all.timestamp_ns, g_all.y) - y)
               for j in js]                                   # the keyframe pose is the GT pose at its own frame
        j = js[int(np.argmin(err))]
        kf_t[k] = ts[j]
        prev = ts[j] + 50_000_000                      # next keyframe is at least one frame later
    assert len(np.unique(kf_t)) == len(kf_t), 'duplicate keyframe stamps'
    want = {int(t): k for k, t in enumerate(kf_t)}
    out = os.path.splitext(a.map)[0] + '_img'
    os.makedirs(out, exist_ok=True)
    np.save(f'{out}/stamps.npy', kf_t)
    n = 0
    with open(a.bag, 'rb') as f:
        fd = f.fileno()
        for i, (_, _, _, msg) in enumerate(make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                topics=[f'/constructor0/sensor/camera_{a.cam}/compressed_image'])):
            t = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if t in want:
                g8 = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_GRAYSCALE)
                g8 = cv2.resize(g8, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
                cv2.imwrite(f'{out}/{want[int(t)]}.jpg', g8, [cv2.IMWRITE_JPEG_QUALITY, 90])
                n += 1
            if i % 2000 == 0 and hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)     # do not let the bag fill the page cache
    print(f'{n}/{len(kf_t)} keyframe images -> {out}')


if __name__ == '__main__':
    main()
