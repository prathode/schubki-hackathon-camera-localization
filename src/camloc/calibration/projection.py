"""Geometry: ground-plane projection and map<->image transforms.

Conventions:
  * pose = [x, y, yaw] of base_link in map (REP 103: x fwd, y left, z up).
  * camera optical frame: z fwd, x right, y down.
  * extrinsics give T_bl_cam (point in camera optical -> base_link).

Flat-track assumption: the ground is the plane z = 0 in base_link's local
frame (i.e. the road surface is at the height of base_link's origin minus the
ride height). We treat the ground plane as z = z_ground in base_link; by
default z_ground = 0 which corresponds to base_link at road level. Because
base_link z is defined at the vehicle reference, we use the camera height
over ground from extrinsics directly (cam z in base_link) and intersect rays
with the plane z = -h_offset. See `ground_project_pixels`.
"""
from __future__ import annotations

import numpy as np


def pose_to_T_map_bl(pose: np.ndarray) -> np.ndarray:
    """2D pose [x,y,yaw] -> 4x4 transform base_link -> map (planar)."""
    x, y, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T[:3, 3] = [x, y, 0.0]
    return T


def ground_project_pixels(
    pixels: np.ndarray,
    calib,
    pose: np.ndarray,
    z_ground_bl: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project pixels to the ground plane, returning map-frame XY.

    pixels : (N,2) pixel coords in the camera `calib`.
    pose   : current base_link pose [x,y,yaw] in map.
    z_ground_bl : height of the ground plane in base_link z (default 0 =
                  road at base_link origin height).

    Returns (xy_map (M,2), valid_mask over input (N,)). Rays not hitting the
    ground in front of the camera are dropped.
    """
    extr = calib.extrinsics
    if extr is None or len(pixels) == 0:
        return np.zeros((0, 2)), np.zeros(len(pixels), dtype=bool)

    rays_cam = calib.undistort_points(pixels)          # (N,3) optical frame
    R_bl_cam = extr.T_bl_cam[:3, :3]
    o_bl = extr.T_bl_cam[:3, 3]                          # camera centre in bl

    # rotate rays into base_link
    rays_bl = (R_bl_cam @ rays_cam.T).T                  # (N,3)

    # intersect with plane z = z_ground_bl in base_link:
    #   o_bl.z + t * ray_bl.z = z_ground_bl  ->  t = (z_ground - o.z)/ray.z
    denom = rays_bl[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z_ground_bl - o_bl[2]) / denom
    valid = np.isfinite(t) & (t > 0)                     # in front, hits ground
    t = np.where(valid, t, np.nan)
    pts_bl = o_bl[None, :] + rays_bl * t[:, None]        # (N,3) in base_link

    # keep only points ahead/behind within a sane range handled by caller.
    valid &= np.isfinite(pts_bl).all(axis=1)
    pts_bl = pts_bl[valid]

    # base_link -> map
    T = pose_to_T_map_bl(pose)
    pts_h = np.concatenate([pts_bl[:, :2], np.zeros((len(pts_bl), 1)),
                            np.ones((len(pts_bl), 1))], axis=1)
    # use planar: only x,y of base_link matter
    xy_bl = pts_bl[:, :2]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    Rm = np.array([[c, -s], [s, c]])
    xy_map = (Rm @ xy_bl.T).T + pose[:2]
    return xy_map, valid


def project_map_points_to_image(
    pts_map: np.ndarray,
    calib,
    pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project map-frame ground points (z=0) into image pixels.

    Returns (uv (M,2), mask of points in front of camera & on image).
    Used for visualization / verification.
    """
    import cv2
    extr = calib.extrinsics
    if extr is None or len(pts_map) == 0:
        return np.zeros((0, 2)), np.zeros(len(pts_map), dtype=bool)

    # map -> base_link (planar)
    c, s = np.cos(pose[2]), np.sin(pose[2])
    Rm = np.array([[c, s], [-s, c]])      # inverse rotation
    xy_bl = (Rm @ (pts_map - pose[:2]).T).T
    pts_bl = np.concatenate([xy_bl, np.zeros((len(xy_bl), 1))], axis=1)  # z=0

    # base_link -> camera optical
    T_cam_bl = extr.T_cam_bl
    pts_h = np.concatenate([pts_bl, np.ones((len(pts_bl), 1))], axis=1)
    pts_cam = (T_cam_bl @ pts_h.T).T[:, :3]
    front = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[front]
    if len(pts_cam) == 0:
        return np.zeros((0, 2)), front
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    uv, _ = cv2.projectPoints(pts_cam.reshape(-1, 1, 3), rvec, tvec,
                              calib.K, calib.D)
    uv = uv.reshape(-1, 2)
    on = (uv[:, 0] >= 0) & (uv[:, 0] < calib.width) & \
         (uv[:, 1] >= 0) & (uv[:, 1] < calib.height)
    full_mask = front.copy()
    full_mask[front] = on
    return uv[on], full_mask
