#!/usr/bin/env python3
"""One pass over the mcap -> small cached files for development.

    python3 tools/export_cache.py c0_2026-10-13_22-53-42_camloc_0.mcap cache --img-every 10

cache/
  calib.json          per camera: K, D, width, height, T_base_cam (4x4, base_link <- camera optical)
  track_bounds.npz    track_left, track_right, pitlane_left, pitlane_right: (N, 2) x, y in map
  state.csv           selected car state fields (velocities, wheel speeds, steering, ...), no pose
  frames.csv          every image: cam, timestamp_ns (header.stamp), log_time_ns
  img/<cam>/<timestamp_ns>.jpg   every Nth image per camera (raw jpeg bytes)
"""
import argparse, csv, json, os
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

NS = 'constructor0'
CAMS = ['fl', 'fr', 'cl', 'cr', 'r', 'rl', 'rr']
STATE_FIELDS = ['vx_mps', 'vy_mps', 'vz_mps', 'wx_radps', 'wy_radps', 'wz_radps', 'v_mps', 'v_raw_mps',
                'beta_rad', 'ax_mps2', 'ay_mps2', 'az_mps2', 'yaw_vel_rad', 'kappa_radpm',
                'omega_w_fl', 'omega_w_fr', 'omega_w_rl', 'omega_w_rr', 'delta_wheel_rad',
                'gear', 'rpm', 'gas', 'brake', 'valid_imu_b', 'valid_wheelsspeeds_b', 'se_status', 'se_state']


def stamp_ns(h):
    return h.stamp.sec * 10**9 + h.stamp.nanosec


def tf_matrix(t):
    q, p = t.rotation, t.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, [p.x, p.y, p.z]
    return T


def chain_to_base(static, child):
    """Compose parent <- child static transforms up to base_link."""
    T = np.eye(4)
    while child != f'{NS}/base_link':
        parent, Tpc = static[child]
        T = Tpc @ T
        child = parent
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('out')
    ap.add_argument('--img-every', type=int, default=10, help='save every Nth image per camera (0 = none)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for c in CAMS:
        os.makedirs(f'{a.out}/img/{c}', exist_ok=True)

    img_topic = {f'/{NS}/sensor/camera_{c}/compressed_image': c for c in CAMS}
    info_topic = {f'/{NS}/sensor/camera_{c}/camera_info': c for c in CAMS}
    static, info, bounds = {}, {}, None
    img_count = {c: 0 for c in CAMS}

    with open(a.bag, 'rb') as f, \
            open(f'{a.out}/state.csv', 'w', newline='') as fs, \
            open(f'{a.out}/frames.csv', 'w', newline='') as ff:
        ws, wf = csv.writer(fs), csv.writer(ff)
        ws.writerow(['timestamp_ns'] + STATE_FIELDS)
        wf.writerow(['cam', 'timestamp_ns', 'log_time_ns'])
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for i, (_, ch, m, msg) in enumerate(r.iter_decoded_messages(
                topics=list(img_topic) + list(info_topic) + ['/tf_static', f'/{NS}/state_estimation',
                                                             f'/{NS}/track_bounds', f'/{NS}/validation/twist'])):
            topic = ch.topic
            if topic in img_topic:
                c = img_topic[topic]
                ts = stamp_ns(msg.header)
                wf.writerow([c, ts, m.log_time])
                if a.img_every and img_count[c] % a.img_every == 0:
                    with open(f'{a.out}/img/{c}/{ts}.jpg', 'wb') as o:
                        o.write(bytes(msg.data))
                img_count[c] += 1
            elif topic == f'/{NS}/state_estimation':
                ws.writerow([stamp_ns(msg.header)] + [getattr(msg, k) for k in STATE_FIELDS])
            elif topic == f'/{NS}/validation/twist':
                row = {k: 0.0 for k in STATE_FIELDS}
                row['vx_mps'] = msg.twist.linear.x
                row['vy_mps'] = msg.twist.linear.y
                row['wz_radps'] = msg.twist.angular.z
                row['v_mps'] = float(np.hypot(msg.twist.linear.x, msg.twist.linear.y))
                ws.writerow([stamp_ns(msg.header)] + [row[k] for k in STATE_FIELDS])
            elif topic in info_topic:
                c = info_topic[topic]
                if c not in info:
                    info[c] = dict(K=list(msg.k), D=list(msg.d), distortion_model=msg.distortion_model,
                                   width=msg.width, height=msg.height, frame_id=msg.header.frame_id)
            elif topic == '/tf_static':
                for t in msg.transforms:
                    static[t.child_frame_id] = (t.header.frame_id, tf_matrix(t.transform))
            elif topic == f'/{NS}/track_bounds' and bounds is None:
                bounds = {mk.ns or str(mk.id): np.array([[p.x, p.y] for p in mk.points]) for mk in msg.markers}
            if i % 50000 == 0:
                print(f'{i} msgs, t={m.log_time / 1e9:.1f}', flush=True)

    if bounds is None:
        for cand in ['cache/track_bounds.npz', '../Hackathon-schubki-gemi/data/track_and_calibration.json',
                     'd:/Parikshit/Hackathon-schubki-gemi/data/track_and_calibration.json']:
            if os.path.exists(cand):
                if cand.endswith('.npz'):
                    bounds = dict(np.load(cand))
                else:
                    with open(cand) as bf:
                        bounds = {k: np.array(v) for k, v in json.load(bf)['track_bounds'].items()}
                break

    calib = {}
    for c in CAMS:
        d = dict(info[c])
        d['T_base_cam'] = chain_to_base(static, f'{NS}/camera_{c}').tolist()
        calib[c] = d
    calib['_static_frames'] = {k: v[0] for k, v in static.items()}
    with open(f'{a.out}/calib.json', 'w') as o:
        json.dump(calib, o, indent=1)
    np.savez(f'{a.out}/track_bounds.npz', **bounds)
    print('track bounds:', {k: v.shape for k, v in bounds.items()})
    print('images per camera:', img_count)


if __name__ == '__main__':
    main()
