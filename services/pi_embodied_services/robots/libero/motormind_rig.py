# Copyright 2026 The pi-embodied Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MotorMind mode's LIBERO rig: the pure functions behind ``env.mm_observe`` and
``env.mm_run_step`` (numpy and Pillow only, no simulator).

Written from the behaviour specs in pi-embodied-work/docs/motormind (spec-adapter section 4,
spec-locate sections 3 and 7, spec-executor sections 5 and 6) and DESIGN.md sections 2 to 5;
the original MotorMind code is not copied. The constants are the numbers those specs record.

Frames and units: everything here is metres, radians and 3x3 rotation matrices. "base" is the
robot's base body frame (``robot0_base``), read once; poses arrive in it already converted.
Camera extrinsics follow the OpenCV convention (x right, y down, z forward) for an image that
is flipped vertically from MuJoCo's render (row 0 = top). Gripper channel: +1 close, -1 open,
0 hold.

The pieces:

- a segment's servo step, its per-tick judge (blocked / skidded / creeping) and its result
  (:func:`servo_step`, :class:`SegmentJudge`, :func:`segment_result`, :func:`replay_segment`);
- the lateral limit of a translation and the grasp judgement (:func:`lateral_limit`,
  :func:`grasp_state`);
- the side camera's mount and the table-top pick it aims at (:func:`side_mount`,
  :func:`table_top`), the table height from scene depth (:func:`table_height`);
- the wrist clearance cylinder and the aim marker's spot (:func:`clearance`, :func:`aim_point`);
- camera calibration and projection (:func:`intrinsic`, :func:`cam2world`, :func:`project`);
- the images the model is shown: JPEG round trip, the aim marker, 512 versions and the
  three-view mosaic (:func:`camera_images`, :func:`mosaic`);
- LIBERO's capability table (:func:`libero_caps`).
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants (spec-adapter 4.3-4.9, spec-locate 3 and 11, spec-executor 5-6)
# ---------------------------------------------------------------------------

#: Control period of LIBERO's OSC (20 Hz); the frame clock advances by it per env tick.
CONTROL_DT = 0.05
#: A segment has arrived within these.
ARRIVE_M = 0.003
ARRIVE_RAD = 0.03
#: A segment ends ``limited`` beyond these (three times the arrival band).
LIMITED_M = 0.009
LIMITED_RAD = 0.09
#: The most env ticks one segment servos.
SEGMENT_MAX_TICKS = 200
#: Stall (blocked): checked after this many ticks, a tick counts when it advanced less than
#: this share of the best tick so far, and this many counting ticks in a row abort.
STALL_SETTLE_TICKS = 3
STALL_FRACTION = 0.25
STALL_TICKS = 8
#: Lateral limit: a share of the segment's length with a floor; holding something widens it,
#: a vertical move without a payload narrows it.
LATERAL_FRACTION = 0.30
LATERAL_FLOOR_M = 0.004
PAYLOAD_LATERAL_M = 0.015
VERTICAL_LATERAL_M = 0.005
#: A downward move that drifts sideways having advanced at most this along its axis is blocked.
DOWN_BLOCKED_ALONG_M = 0.003
#: |axis_z| above this counts as a vertical move (the original's test; its up/down words are
#: exact, an axis 8 deg or more off vertical is not vertical).
VERTICAL_AXIS_Z = 0.99
#: Creep: position change + this weight x turn below CREEP_EPS for more than CREEP_TICKS ticks
#: ends the loop (no abort reason).
CREEP_EPS = 1e-5
CREEP_ROT_WEIGHT = 0.05
CREEP_TICKS = 12
#: Zero ticks held after an abort.
HOLD_TICKS = 2
#: Gripper: ticks a close / full open settles, the most ticks an open-to-width opens, the empty
#: band, the share of the full span above which a close did not happen, the per-tick change that
#: counts as still moving, and the full span LIBERO's caps declare.
GRIPPER_SETTLE_TICKS = 32
OPEN_TO_MAX_TICKS = 32
EMPTY_BAND_M = 0.003
WIDE_OPEN_FRACTION = 0.90
MOVING_PER_TICK_M = 0.0005
GRIPPER_OPEN_SPAN_M = 0.0788
GRIPPER_PAD_LENGTH_M = 0.017
#: Table height from scene depth: every STRIDE-th pixel, BIN_M bins, at least MIN_POINTS points.
TABLE_STRIDE = 2
TABLE_BIN_M = 0.005
TABLE_MIN_POINTS = 100
#: Side camera mount: aim at base xy + this (world axes), on the table top; eye on base +Y at
#: this elevation; a free body stands on a geom when its origin is at most STANDS_BELOW_M below
#: the geom's top.
SIDE_AIM_BASE_XY = (0.47, 0.0)
SIDE_ELEVATION_DEG = 35.0
SIDE_SIGN_Y = 1.0
STANDS_BELOW_M = 0.03
#: Wrist depth: the simulated minimum range (compared with each pixel's depth, camera z), the
#: depth image's quantum, the band above the minimum range that is a floor and not a surface
#: (in quanta), and the clearance cylinder around the approach axis (radius, the band beyond the
#: tool point that is skipped, the fewest returns).
DEPTH_MIN_RANGE_M = 0.21
DEPTH_QUANTUM_M = 0.001
FLOOR_BAND_QUANTA = 2
CLEARANCE_RADIUS_M = 0.040
CLEARANCE_BEYOND_M = 0.005
CLEARANCE_MIN_POINTS = 50
NEAR_MIN_RANGE_FACTOR = 1.25
#: Jaws count as pointing down within this of vertical: the approach axis then snaps to straight
#: down (heights, the aim marker).
DOWN_WITHIN_DEG = 45.0
#: The aim marker's depth below the tool: at least this above the table; without a table or a
#: clearance reading, this guess.
AIM_MIN_DEPTH_M = 0.02
AIM_DEFAULT_DEPTH_M = 0.20
#: Render size, the size most roles are sent, JPEG quality, and the aim marker's colour.
CAMERA_PX = 768
MODEL_PX = 512
JPEG_QUALITY = 90
AIM_RGB = (255, 0, 255)
#: MuJoCo geom types the table-top pick considers (plane, capsule, cylinder, box).
GEOM_PLANE, GEOM_CAPSULE, GEOM_CYLINDER, GEOM_BOX = 0, 3, 5, 6
TABLE_GEOM_TYPES = (GEOM_PLANE, GEOM_CAPSULE, GEOM_CYLINDER, GEOM_BOX)


# ---------------------------------------------------------------------------
# Rotations and frames
# ---------------------------------------------------------------------------


def quat_xyzw_matrix(q: Any) -> np.ndarray:
    """The rotation matrix of an xyzw quaternion (normalised first)."""
    x, y, z, w = np.asarray(q, dtype=np.float64).reshape(4)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_quat_wxyz(r: Any) -> np.ndarray:
    """The wxyz quaternion (w >= 0) of a rotation matrix (MuJoCo's ``cam_quat`` order)."""
    m = np.asarray(r, dtype=np.float64)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        ]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        ]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        ]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        ]
    out = np.asarray(q) / np.linalg.norm(q)
    return -out if out[0] < 0 else out


def matrix_quat_xyzw(r: Any) -> np.ndarray:
    w, x, y, z = matrix_quat_wxyz(r)
    return np.array([x, y, z, w])


def rotvec(r: Any) -> np.ndarray:
    """Axis x angle (angle in [0, pi]) of a rotation matrix; atan2 keeps small angles exact."""
    m = np.asarray(r, dtype=np.float64)
    w = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    s = float(np.linalg.norm(w))  # 2 sin(angle)
    c = float(np.trace(m) - 1.0)  # 2 cos(angle)
    angle = math.atan2(s, c)
    if angle < 1e-12:
        return np.zeros(3)
    if s < 1e-6:  # near a half turn: the axis from the symmetric part
        cols = m + np.eye(3)
        axis = cols[:, int(np.argmax(np.linalg.norm(cols, axis=0)))]
        return axis / np.linalg.norm(axis) * angle
    return w / s * angle


def axis_angle_matrix(v: Any) -> np.ndarray:
    """The rotation matrix of a rotation vector (Rodrigues)."""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    a = float(np.linalg.norm(v))
    if a < 1e-15:
        return np.eye(3)
    k = v / a
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * kx + (1 - math.cos(a)) * kx @ kx


def angle_between(ra: Any, rb: Any) -> float:
    """The angle (rad) of the rotation taking ``ra`` to ``rb``."""
    return float(np.linalg.norm(rotvec(np.asarray(rb) @ np.asarray(ra).T)))


@dataclass(frozen=True)
class BaseFrame:
    """The base body's world pose: ``p`` (3,) and ``R`` (3x3). ``to_base`` maps world poses into
    it (``R^T (p - p_b)``, ``R^T R``); ``vec_to_world`` maps a base vector to world axes."""

    p: np.ndarray
    R: np.ndarray

    def point(self, p_world: Any) -> np.ndarray:
        return self.R.T @ (np.asarray(p_world, dtype=np.float64).reshape(3) - self.p)

    def points(self, p_world: np.ndarray) -> np.ndarray:
        return (np.asarray(p_world, dtype=np.float64) - self.p) @ self.R

    def rotation(self, r_world: Any) -> np.ndarray:
        return self.R.T @ np.asarray(r_world, dtype=np.float64)

    def world_point(self, p_base: Any) -> np.ndarray:
        return self.R @ np.asarray(p_base, dtype=np.float64).reshape(3) + self.p

    def vec_to_world(self, v_base: Any) -> np.ndarray:
        return self.R @ np.asarray(v_base, dtype=np.float64).reshape(3)

    def transform(self) -> np.ndarray:
        """T_base<-world (4x4)."""
        t = np.eye(4)
        t[:3, :3] = self.R.T
        t[:3, 3] = -self.R.T @ self.p
        return t


def approach_axis(r: Any) -> np.ndarray:
    """The tool's approach direction (its +z column)."""
    return np.asarray(r, dtype=np.float64)[:, 2].copy()


def jaws_down(r: Any) -> bool:
    """Whether the approach axis is within DOWN_WITHIN_DEG of vertical (it then counts as
    straight down: :func:`snapped_axis`)."""
    a = approach_axis(r)
    return abs(float(a[2])) >= math.cos(math.radians(DOWN_WITHIN_DEG)) * float(
        np.linalg.norm(a)
    )


def snapped_axis(r: Any) -> np.ndarray:
    """The approach axis as heights and the aim read it: straight down (0, 0, -1) within
    DOWN_WITHIN_DEG of vertical, else the unit axis itself."""
    a = approach_axis(r)
    a = a / float(np.linalg.norm(a))
    return np.array([0.0, 0.0, -1.0]) if jaws_down(r) else a


def aim_point(
    p: Any,
    r: Any,
    table_z: float | None,
    clearance_mm: float | None = None,
    tracked_m: float | None = None,
) -> tuple[np.ndarray, float, str]:
    """The spot the aim marker shows (base frame): the tool point moved along the snapped approach
    axis by a depth. The depth is the height above the table when the table is known and the
    jaws point down (at least AIM_MIN_DEPTH_M), else the wrist clearance, else the caller's
    tracked clearance, else AIM_DEFAULT_DEPTH_M. Returns the point, the depth and its source."""
    p = np.asarray(p, dtype=np.float64).reshape(3)
    axis = snapped_axis(r)
    if table_z is not None and jaws_down(r):
        depth, source = max(AIM_MIN_DEPTH_M, float(p[2]) - float(table_z)), "table"
    elif clearance_mm:
        depth, source = float(clearance_mm) / 1000.0, "clearance"
    elif tracked_m:
        depth, source = float(tracked_m), "tracked"
    else:
        depth, source = AIM_DEFAULT_DEPTH_M, "default"
    return p + axis * depth, depth, source


def pose_dict(p: Any, r: Any) -> dict:
    """A pose for the wire: ``position_m``, ``rotation`` (rows), ``quat_xyzw``."""
    return {
        "position_m": [float(v) for v in np.asarray(p).reshape(3)],
        "rotation": np.asarray(r, dtype=np.float64).tolist(),
        "quat_xyzw": [float(v) for v in matrix_quat_xyzw(r)],
    }


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------


def intrinsic(fovy_deg: float, height: int, width: int) -> np.ndarray:
    """Pinhole K of a MuJoCo camera: f = 0.5 H / tan(fovy / 2), principal point at the centre."""
    f = 0.5 * height / math.tan(math.radians(float(fovy_deg)) / 2)
    return np.array([[f, 0.0, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]])


#: MuJoCo cameras look along their -z with +y up; the OpenCV frame flips y and z.
_OPENCV_FROM_MUJOCO = np.diag([1.0, -1.0, -1.0])


def cam2world(cam_xpos: Any, cam_xmat: Any) -> np.ndarray:
    """Camera-to-world (4x4, OpenCV axes) from MuJoCo's camera position and orientation."""
    t = np.eye(4)
    t[:3, :3] = (
        np.asarray(cam_xmat, dtype=np.float64).reshape(3, 3) @ _OPENCV_FROM_MUJOCO
    )
    t[:3, 3] = np.asarray(cam_xpos, dtype=np.float64).reshape(3)
    return t


def project(k: Any, cam2x: Any, p: Any) -> tuple[float, float, float] | None:
    """Pixel (u = column, v = row, both continuous) and camera z of point ``p`` given in the
    frame ``cam2x`` maps the camera into, for the upright image; None behind the camera."""
    t = np.asarray(cam2x, dtype=np.float64)
    pc = t[:3, :3].T @ (np.asarray(p, dtype=np.float64).reshape(3) - t[:3, 3])
    if pc[2] <= 1e-6:
        return None
    k = np.asarray(k, dtype=np.float64)
    return (
        float(k[0, 0] * pc[0] / pc[2] + k[0, 2]),
        float(k[1, 1] * pc[1] / pc[2] + k[1, 2]),
        float(pc[2]),
    )


def back_project(depth: np.ndarray, k: Any, cam2x: Any, stride: int = 1):
    """Points (N, 3) in ``cam2x``'s frame and their depths (N,) for every ``stride``-th pixel
    with depth > 0 of an upright depth image (metres)."""
    d = np.asarray(depth, dtype=np.float64)[::stride, ::stride]
    rows, cols = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    ok = np.isfinite(d) & (d > 0)
    z = d[ok]
    k = np.asarray(k, dtype=np.float64)
    x = (cols[ok] - k[0, 2]) * z / k[0, 0]
    y = (rows[ok] - k[1, 2]) * z / k[1, 1]
    t = np.asarray(cam2x, dtype=np.float64)
    pts = np.stack([x, y, z], axis=1) @ t[:3, :3].T + t[:3, 3]
    return pts, z


def quantize_depth(depth_m: np.ndarray) -> np.ndarray:
    """Depth as the 16-bit millimetre image carries it: NaN to 0, x1000, clipped to 0..65535,
    truncated; back in metres."""
    d = np.nan_to_num(np.asarray(depth_m, dtype=np.float64), nan=0.0, posinf=0.0)
    mm = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
    return mm.astype(np.float64) * DEPTH_QUANTUM_M


# ---------------------------------------------------------------------------
# Side camera mount and the table top it aims at
# ---------------------------------------------------------------------------


def geom_top_and_footprint(gtype: int, pos: Any, xmat: Any, size: Any):
    """A static geom's top z and its world-axis xy footprint ``(xmin, xmax, ymin, ymax)``
    (None for an infinite plane side)."""
    p = np.asarray(pos, dtype=np.float64).reshape(3)
    r = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    s = np.asarray(size, dtype=np.float64).reshape(-1)
    if gtype == GEOM_BOX:
        half = np.abs(r) @ s[:3]
        return p[2] + half[2], (
            p[0] - half[0],
            p[0] + half[0],
            p[1] - half[1],
            p[1] + half[1],
        )
    if gtype in (GEOM_CYLINDER, GEOM_CAPSULE):
        radius, half_len = s[0], s[1]
        axis = r[:, 2]
        cap = radius if gtype == GEOM_CAPSULE else 0.0
        ext = (
            np.abs(axis) * (half_len + cap)
            + np.sqrt(np.clip(1 - axis**2, 0, 1)) * radius
        )
        return p[2] + ext[2], (
            p[0] - ext[0],
            p[0] + ext[0],
            p[1] - ext[1],
            p[1] + ext[1],
        )
    if gtype == GEOM_PLANE:
        # MuJoCo: a plane's half sizes are its first two size entries; 0 is infinite.
        if s[0] <= 0 or s[1] <= 0:
            return p[2], (-math.inf, math.inf, -math.inf, math.inf)
        half = np.abs(r[:2, :2]) @ s[:2]
        return p[2], (p[0] - half[0], p[0] + half[0], p[1] - half[1], p[1] + half[1])
    return None, None


def table_top(geoms: list[dict], free_xyz: Any) -> dict | None:
    """The static geom most free bodies stand on (ties: the higher top), as the side mount's
    aim height. ``geoms``: dicts with ``type``, ``pos``, ``xmat``, ``size`` (and ``name``);
    meshes and other types are skipped. A body stands on a geom when its origin's xy is inside
    the geom's world-axis footprint and the origin is at most STANDS_BELOW_M below its top.
    None when no geom carries a body."""
    pts = np.asarray(free_xyz, dtype=np.float64).reshape(-1, 3)
    best = None
    for i, g in enumerate(geoms):
        if int(g["type"]) not in TABLE_GEOM_TYPES:
            continue
        top, fp = geom_top_and_footprint(int(g["type"]), g["pos"], g["xmat"], g["size"])
        if top is None:
            continue
        inside = (
            (pts[:, 0] >= fp[0])
            & (pts[:, 0] <= fp[1])
            & (pts[:, 1] >= fp[2])
            & (pts[:, 1] <= fp[3])
            & (pts[:, 2] >= top - STANDS_BELOW_M)
        )
        n = int(inside.sum())
        if n == 0:
            continue
        key = (n, float(top))
        if best is None or key > best[0]:
            best = (
                key,
                {
                    "index": i,
                    "name": g.get("name"),
                    "body": g.get("body"),
                    "top_z": float(top),
                    "count": n,
                },
            )
    return None if best is None else best[1]


def look_at_wxyz(eye: Any, target: Any, up: Any = (0.0, 0.0, 1.0)) -> np.ndarray:
    """MuJoCo camera orientation (wxyz) looking from ``eye`` at ``target``: columns right, true
    up, minus forward."""
    f = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    f /= np.linalg.norm(f)
    right = np.cross(f, np.asarray(up, dtype=np.float64))
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("look-at direction is parallel to up")
    right /= np.linalg.norm(right)
    true_up = np.cross(right, f)
    return matrix_quat_wxyz(np.stack([right, true_up, -f], axis=1))


def side_mount(
    base_pos_world: Any, table_top_z: float, agentview_pos_world: Any
) -> dict:
    """The side camera's pose by the mount rule: aim = base xy + SIDE_AIM_BASE_XY (world axes)
    at the table top; standoff = the agentview camera's distance to the aim; eye = aim +
    (0, sign * d cos(elev), d sin(elev)). Returns ``aim``, ``eye`` (= cam_pos for a camera
    whose parent is the worldbody), ``quat_wxyz``, ``standoff``."""
    b = np.asarray(base_pos_world, dtype=np.float64).reshape(3)
    aim = np.array(
        [b[0] + SIDE_AIM_BASE_XY[0], b[1] + SIDE_AIM_BASE_XY[1], float(table_top_z)]
    )
    d = float(
        np.linalg.norm(
            np.asarray(agentview_pos_world, dtype=np.float64).reshape(3) - aim
        )
    )
    el = math.radians(SIDE_ELEVATION_DEG)
    eye = aim + np.array([0.0, SIDE_SIGN_Y * d * math.cos(el), d * math.sin(el)])
    return {"aim": aim, "eye": eye, "quat_wxyz": look_at_wxyz(eye, aim), "standoff": d}


def table_height(z: Any) -> float | None:
    """The table's height from scene points' z (base frame): the centre of the 5 mm histogram's
    mode bin, refined to the median of the points within one bin of it. None with fewer than
    TABLE_MIN_POINTS points."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    z = z[np.isfinite(z)]
    if len(z) < TABLE_MIN_POINTS:
        return None
    bins = np.floor(z / TABLE_BIN_M).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    centre = (values[int(np.argmax(counts))] + 0.5) * TABLE_BIN_M
    near = z[np.abs(z - centre) <= TABLE_BIN_M]
    return float(np.median(near))


# ---------------------------------------------------------------------------
# Wrist clearance
# ---------------------------------------------------------------------------


def clearance(
    points: Any,
    cam_points: Any,
    tool_p: Any,
    tool_r: Any,
    min_range: float = DEPTH_MIN_RANGE_M,
) -> dict:
    """The free distance below the tool from the wrist camera's returns: ``points`` in the base
    frame and the same returns in the camera frame (``cam_points``, OpenCV axes: z is the pixel's
    depth), metres, every pixel with a depth. The cylinder is the returns within
    CLEARANCE_RADIUS_M of the approach axis and more than CLEARANCE_BEYOND_M past the tool point;
    a return counts when its depth is at least ``min_range`` (the simulated sensor returns nothing
    nearer). Blind (``seen`` false, ``at_range_floor`` and ``near_min_range`` set) with fewer than
    CLEARANCE_MIN_POINTS counted returns; and when what is under the tool is inside the minimum
    range: a fifth of that many cylinder returns sit in the depth band just under it
    ([min_range - FLOOR_BAND_QUANTA quanta, min_range): ``n_cut``), or the nearest counted
    return's depth lies within FLOOR_BAND_QUANTA quanta above it (a lower bound, not a surface).
    The robot's own fingertips sit far nearer (about 0.1 m) and count in neither. Otherwise
    ``mm`` is the nearest distance along the approach axis, ``median_mm`` the median,
    ``surface_z_m`` (jaws down: :func:`snapped_axis`) the base-frame height of that surface, and
    ``near_min_range`` is set when the mean counted return (camera frame) is at most
    NEAR_MIN_RANGE_FACTOR times the minimum range from the lens."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cam = np.asarray(cam_points, dtype=np.float64).reshape(-1, 3)
    depth = cam[:, 2]
    p0 = np.asarray(tool_p, dtype=np.float64).reshape(3)
    a = approach_axis(tool_r)
    rel = pts - p0
    along = rel @ a
    radial = np.linalg.norm(rel - np.outer(along, a), axis=1)
    cone = (along > CLEARANCE_BEYOND_M) & (radial < CLEARANCE_RADIUS_M)
    band = FLOOR_BAND_QUANTA * DEPTH_QUANTUM_M
    kept = cone & (depth >= min_range)
    cut = cone & (depth >= min_range - band) & (depth < min_range)
    n, n_cut = int(kept.sum()), int(cut.sum())
    out: dict[str, Any] = {
        "seen": False,
        "mm": None,
        "median_mm": None,
        "surface_z_m": None,
        "n_points": n,
        "n_cut": n_cut,
        "at_range_floor": False,
        "near_min_range": False,
        "min_range_m": float(min_range),
        "jaws_down": jaws_down(tool_r),
    }
    blind = {"at_range_floor": True, "near_min_range": True}
    if n < CLEARANCE_MIN_POINTS:
        return {
            **out,
            **blind,
            "reason": f"{n} returns in the cylinder (fewer than {CLEARANCE_MIN_POINTS})",
        }
    if n_cut >= max(CLEARANCE_MIN_POINTS // 5, 1):
        return {
            **out,
            **blind,
            "reason": f"{n_cut} returns just under the {min_range:.2f} m minimum range",
        }
    if float(depth[kept].min()) <= min_range + band:
        return {
            **out,
            **blind,
            "reason": "the nearest return sits at the minimum range",
        }
    d = along[kept]
    near = float(d.min())
    out.update(
        seen=True,
        mm=near * 1000.0,
        median_mm=float(np.median(d)) * 1000.0,
        near_min_range=bool(
            float(np.linalg.norm(cam[kept].mean(axis=0)))
            <= NEAR_MIN_RANGE_FACTOR * min_range
        ),
    )
    if out["jaws_down"]:
        out["surface_z_m"] = float(p0[2] - near)
    return out


# ---------------------------------------------------------------------------
# Gripper
# ---------------------------------------------------------------------------


def opening_m(gripper_qpos: Any) -> float:
    """The finger gap: |q0 - q1| of the two finger joints (metres)."""
    q = np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
    return float(abs(q[0] - q[1]))


def channel_name(channel: float) -> str:
    return "close" if channel > 0 else "open" if channel < 0 else "hold"


def grasp_state(
    commanded: str,
    opening: float,
    moving: bool,
    open_span: float = GRIPPER_OPEN_SPAN_M,
) -> dict:
    """Whether the closed fingers hold something: ``holding`` true / false / None (does not
    apply or cannot tell), with ``reason``, ``opening_mm`` and ``closed_fraction``. Only judged
    after a close; still moving (and not already at the empty band) or nearly wide open is
    None; at most EMPTY_BAND_M is empty, anything wider is holding."""
    out = {
        "opening_mm": opening * 1000.0,
        # 0 at the closed end, 1 wide open (the original's field: its rules read < 0.5 as shut).
        "closed_fraction": float(opening / open_span),
        "commanded": commanded,
    }
    if commanded != "close":
        return {
            **out,
            "holding": None,
            "reason": "the last gripper command was not close",
        }
    if moving and opening > EMPTY_BAND_M:
        return {**out, "holding": None, "reason": "the fingers are still moving"}
    if opening / open_span > WIDE_OPEN_FRACTION:
        return {
            **out,
            "holding": None,
            "reason": "nearly wide open: the close did not take effect",
        }
    if opening <= EMPTY_BAND_M:
        return {
            **out,
            "holding": False,
            "reason": f"closed to {opening * 1000:.1f} mm (empty at <= {EMPTY_BAND_M * 1000:.0f} mm)",
        }
    return {
        **out,
        "holding": True,
        "reason": f"the fingers stopped at {opening * 1000:.1f} mm",
    }


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


def is_vertical(axis: Any) -> bool:
    return (
        axis is not None
        and abs(float(np.asarray(axis).reshape(3)[2])) > VERTICAL_AXIS_Z
    )


def is_downward(axis: Any) -> bool:
    return is_vertical(axis) and float(np.asarray(axis).reshape(3)[2]) < 0


def lateral_limit(distance_m: float, holding: bool, axis: Any) -> float | None:
    """How far a translation may drift sideways before it aborts: max(30% of its length, 4 mm);
    holding something, at least 15 mm; a vertical move without a payload, at most 5 mm. None
    (no limit) for a zero-length request."""
    if not distance_m or distance_m <= 0:
        return None
    lim = max(LATERAL_FRACTION * distance_m, LATERAL_FLOOR_M)
    if holding:
        lim = max(PAYLOAD_LATERAL_M, lim)
    elif is_vertical(axis):
        lim = min(lim, VERTICAL_LATERAL_M)
    return lim


@dataclass
class Segment:
    """One segment for the servo loop (base frame). ``kind`` is ``translate`` or ``rotate``;
    ``speed`` is m/s for a translation and rad/s for a rotation and caps both channels' per-tick
    step (``speed * dt``), as the original does; ``axis`` and ``distance_m`` belong to a
    translation."""

    kind: str
    goal_p: np.ndarray
    goal_r: np.ndarray
    speed: float
    axis: np.ndarray | None = None
    distance_m: float = 0.0
    push: bool = False
    holding: bool = False


def servo_step(
    p: Any,
    r: Any,
    seg: Segment,
    max_translation: float,
    max_rotation: float,
    dt: float = CONTROL_DT,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """The next tick's base-frame increments toward the segment's goal: translation
    min(gap, speed dt, max) along the gap, rotation min(angle, speed dt, max) about the error
    axis; ``arrived`` (and zero increments) within ARRIVE_M and ARRIVE_RAD."""
    gap = seg.goal_p - np.asarray(p, dtype=np.float64)
    dist = float(np.linalg.norm(gap))
    rv = rotvec(seg.goal_r @ np.asarray(r, dtype=np.float64).T)
    ang = float(np.linalg.norm(rv))
    if dist <= ARRIVE_M and ang <= ARRIVE_RAD:
        return np.zeros(3), np.zeros(3), True
    step_t = min(max_translation, seg.speed * dt)
    step_r = min(max_rotation, seg.speed * dt)
    dp = gap / dist * min(dist, step_t) if dist > 0 else np.zeros(3)
    dr = rv / ang * min(ang, step_r) if ang > 0 else np.zeros(3)
    return dp, dr, False


def osc_action(
    dp_base, dr_base, output_max, grip: float, base_r=None, ref_frame="base"
) -> np.ndarray:
    """The 7-d OSC action: increments in the controller's reference frame (``base``: as given;
    ``world``: turned by the base rotation), divided by ``output_max`` and clipped to [-1, 1],
    then the gripper channel."""
    dp = np.asarray(dp_base, dtype=np.float64)
    dr = np.asarray(dr_base, dtype=np.float64)
    if ref_frame == "world" and base_r is not None:
        dp, dr = np.asarray(base_r) @ dp, np.asarray(base_r) @ dr
    om = np.asarray(output_max, dtype=np.float64).reshape(6)
    a = np.clip(np.concatenate([dp / om[:3], dr / om[3:]]), -1.0, 1.0)
    return np.concatenate([a, [float(grip)]]).astype(np.float64)


@dataclass
class SegmentJudge:
    """The per-tick checks of a segment, fed the tool pose after every tick (``tick``).

    - blocked: a translation that stopped advancing along its axis. Once past
      STALL_SETTLE_TICKS ticks, with a best tick so far and still LIMITED_M from the goal, a tick
      advancing less than STALL_FRACTION of the best counts and one advancing more resets the
      count; a tick outside those conditions (near the goal, say) leaves the count as it is.
      STALL_TICKS counts abort.
    - skidded: sideways drift from the segment's start beyond the lateral limit; a downward move
      that advanced at most DOWN_BLOCKED_ALONG_M is blocked instead.
    - creep: position change + CREEP_ROT_WEIGHT x turn below CREEP_EPS for more than CREEP_TICKS
      ticks: the loop ends without an abort."""

    seg: Segment
    p0: np.ndarray
    r0: np.ndarray
    limit: float | None = None
    ticks: int = 0
    best: float = 0.0
    stalls: int = 0
    creeps: int = 0
    abort: str = ""
    crept: bool = False
    _prev_p: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    _prev_r: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self):
        self.p0 = np.asarray(self.p0, dtype=np.float64).reshape(3)
        self.r0 = np.asarray(self.r0, dtype=np.float64).reshape(3, 3)
        self._prev_p, self._prev_r = self.p0.copy(), self.r0.copy()
        if self.limit is None and self.seg.kind == "translate":
            self.limit = lateral_limit(
                self.seg.distance_m, self.seg.holding, self.seg.axis
            )

    def tick(self, p: Any, r: Any) -> str:
        """Record the pose after one tick; returns ``"abort"``, ``"creep"`` or ``""``."""
        p = np.asarray(p, dtype=np.float64).reshape(3)
        r = np.asarray(r, dtype=np.float64).reshape(3, 3)
        self.ticks += 1
        step = p - self._prev_p
        turn = angle_between(self._prev_r, r)
        self._prev_p, self._prev_r = p, r
        a = self.seg.axis
        if self.seg.kind == "translate" and a is not None:
            rate = float(step @ a)
            self.best = max(self.best, rate)
            far = float(np.linalg.norm(self.seg.goal_p - p)) > LIMITED_M
            if self.ticks > STALL_SETTLE_TICKS and self.best > 0 and far:
                if rate < STALL_FRACTION * self.best:
                    self.stalls += 1
                    if self.stalls >= STALL_TICKS:
                        self.abort = "blocked"
                        return "abort"
                else:
                    self.stalls = 0
            if self.limit is not None:
                d = p - self.p0
                along = float(d @ a)
                lateral = float(np.linalg.norm(d - along * a))
                if lateral > self.limit:
                    down = is_downward(a) and along <= DOWN_BLOCKED_ALONG_M
                    self.abort = "blocked" if down else "skidded"
                    return "abort"
        if float(np.linalg.norm(step)) + CREEP_ROT_WEIGHT * turn < CREEP_EPS:
            self.creeps += 1
        else:
            self.creeps = 0
        if self.creeps > CREEP_TICKS:
            self.crept = True
            return "creep"
        return ""


def segment_result(seg: Segment, p0, r0, p1, r1, abort: str) -> dict:
    """The measured result of a segment: ``moved_m``, ``along_axis_m``, ``lateral_m`` (a
    translation's decomposition along its axis; otherwise along = moved, lateral = 0),
    ``turned_rad``, the residual to the goal, ``outcome`` (``limited`` beyond LIMITED_M or
    LIMITED_RAD, else ``done``; blocked is ``contact`` for a push and ``limited`` otherwise,
    skidded is ``limited``) and a one-line ``message`` for people (no rule parses it)."""
    p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    d = p1 - p0
    moved = float(np.linalg.norm(d))
    if seg.kind == "translate" and seg.axis is not None:
        along = float(d @ seg.axis)
        lateral = float(np.linalg.norm(d - along * seg.axis))
    else:
        along, lateral = moved, 0.0
    turned = angle_between(r0, r1)
    res_m = float(np.linalg.norm(seg.goal_p - p1))
    res_rad = angle_between(r1, seg.goal_r)
    outcome = "limited" if res_m > LIMITED_M or res_rad > LIMITED_RAD else "done"
    mm = along * 1000.0
    if abort == "blocked":
        if seg.push:
            outcome, msg = "contact", f"contact: pushed {mm:.1f} mm and met resistance"
        else:
            outcome = "limited"
            msg = (
                f"blocked after {mm:.1f} mm: something is in the way or the arm is at its limit "
                "(descending onto an object, this is the fingers touching it)"
            )
    elif abort == "skidded":
        outcome = "limited"
        msg = f"skidded {lateral * 1000:.1f} mm sideways after {mm:.1f} mm: the tool was pushed aside; locate the target again"
    elif seg.push:
        msg = f"pushed {mm:.1f} mm and touched nothing" + (
            "" if outcome == "done" else f"; stopped {res_m * 1000:.1f} mm short"
        )
    elif outcome == "done":
        msg = f"arrived: moved {moved * 1000:.1f} mm, turned {math.degrees(turned):.1f} deg"
    else:
        msg = f"stopped short: {res_m * 1000:.1f} mm and {math.degrees(res_rad):.1f} deg from the goal"
    return {
        "outcome": outcome,
        "abort_reason": abort,
        "message": msg,
        "moved_m": moved,
        "along_axis_m": along,
        "lateral_m": lateral,
        "turned_rad": turned,
        "residual_m": res_m,
        "residual_rad": res_rad,
    }


def replay_segment(seg: Segment, poses: list, end: tuple | None = None) -> dict:
    """Re-judge a recorded segment from its per-tick poses (``[(p, r), ...]``: the pose before
    the first tick, then one per servo tick, without the hold ticks): the abort reason, the tick
    it fired at (``abort_tick``), whether creep ended it, and the result measured at ``end``
    (the pose after the hold ticks; default the last pose given). ``env.mm_run_step`` with
    ``trace`` returns exactly this input, so its judgement can be checked offline."""
    (p0, r0), rest = poses[0], poses[1:]
    judge = SegmentJudge(seg, np.asarray(p0), np.asarray(r0))
    stop_at = None
    for i, (p, r) in enumerate(rest):
        if judge.tick(p, r):
            stop_at = i + 1
            break
    p_end, r_end = end if end is not None else poses[-1]
    out = segment_result(seg, p0, r0, p_end, r_end, judge.abort)
    out.update(abort_tick=stop_at if judge.abort else None, crept=judge.crept)
    return out


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def _pil():
    from PIL import Image

    return Image


def jpeg_bytes(rgb: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    _pil().fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(
        buf, format="JPEG", quality=quality
    )
    return buf.getvalue()


def jpeg_roundtrip(rgb: np.ndarray) -> np.ndarray:
    """The image as a JPEG q90 decoder returns it (the original ships frames as JPEG)."""
    return np.asarray(_pil().open(io.BytesIO(jpeg_bytes(rgb))).convert("RGB"))


def resize(rgb: np.ndarray, px: int) -> np.ndarray:
    """``rgb`` scaled to ``px`` on its longest side with LANCZOS."""
    img = _pil().fromarray(np.ascontiguousarray(rgb, dtype=np.uint8))
    s = px / max(img.size)
    if abs(s - 1) < 1e-9:
        return np.asarray(img)
    size = (max(1, round(img.size[0] * s)), max(1, round(img.size[1] * s)))
    return np.asarray(img.resize(size, _pil().LANCZOS))


def draw_aim(rgb: np.ndarray, uv: tuple[float, float] | None) -> np.ndarray:
    """A copy of ``rgb`` with the magenta aim marker (a ring and a cross, sized to the image) at
    pixel ``uv`` (column, row); unchanged when ``uv`` is None or off the image."""
    out = np.array(rgb, dtype=np.uint8, copy=True)
    if uv is None:
        return out
    h, w = out.shape[:2]
    u, v = float(uv[0]), float(uv[1])
    if not (0 <= u < w and 0 <= v < h):
        return out
    radius = max(5.0, 0.014 * max(h, w))
    thick = max(1.5, 0.003 * max(h, w))
    arm = 1.8 * radius
    # Only the window around the marker is drawn on.
    r0, r1 = max(0, int(v - arm - 2)), min(h, int(v + arm + 3))
    c0, c1 = max(0, int(u - arm - 2)), min(w, int(u + arm + 3))
    rows, cols = np.mgrid[r0:r1, c0:c1]
    dy, dx = rows + 0.5 - v, cols + 0.5 - u
    r = np.hypot(dx, dy)
    ring = np.abs(r - radius) <= thick
    cross = (r >= radius * 0.5) & (
        ((np.abs(dy) <= thick / 2 + 0.5) & (np.abs(dx) <= arm))
        | ((np.abs(dx) <= thick / 2 + 0.5) & (np.abs(dy) <= arm))
    )
    out[r0:r1, c0:c1][ring | cross] = AIM_RGB
    return out


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def camera_images(rgb: np.ndarray, uv: tuple[float, float] | None) -> dict:
    """The model's pictures of one camera render (upright, CAMERA_PX square), as base64 JPEG q90:
    ``marked768`` (aim marker), ``marked512`` and ``raw512`` (LANCZOS from the marked and the
    unmarked image). The render goes through one JPEG round trip first, as the original's frames
    do. Also returns the decoded marked 768 array (``_marked``) for the mosaic."""
    base = jpeg_roundtrip(rgb)
    marked = draw_aim(base, uv)
    return {
        "marked768": b64(jpeg_bytes(marked)),
        "marked512": b64(jpeg_bytes(resize(marked, MODEL_PX))),
        "raw512": b64(jpeg_bytes(resize(base, MODEL_PX))),
        "_marked": marked,
    }


def mosaic(images: list[np.ndarray], tile: int = 256) -> str:
    """The views side by side, each scaled to ``tile`` px (``3 x 256`` wide for three), as base64
    JPEG q90: the one picture a tool result carries."""
    tiles = [resize(im, tile) for im in images]
    if not tiles:
        return ""
    return b64(jpeg_bytes(np.concatenate(tiles, axis=1)))


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def libero_caps(
    output_max: Any, dt: float, supports_home: bool, table_z_m: float | None
) -> dict:
    """LIBERO's capability table (spec-adapter 4.9): limits from the controller's ``output_max``
    and the control period; the measured constants of the original rig; ``supports_home`` from
    ``--mm-home``. Notes are this port's own wording."""
    om = np.asarray(output_max, dtype=np.float64).reshape(6)
    max_t, max_r = float(om[:3].max()), float(om[3:].max())
    notes = [
        "A motion that stops making progress ends on its own and reports how far it went.",
        "The episode ends at success or at its step horizon; after that no action is accepted.",
        "home resets the whole scene (env.reset())."
        if supports_home
        else "There is no home: a new episode starts only from a fresh scene reset by the harness.",
    ]
    return {
        "max_translation_m": max_t,
        "max_rotation_rad": max_r,
        "max_speed_m_s": max_t / dt,
        "max_rotation_speed_rad_s": max_r / dt,
        "default_speed_m_s": 0.05,
        "default_rotation_speed_rad_s": math.radians(60.0),
        "control_point_offset_m": 0.0,
        "supports_contact_stop": True,
        "contact_stop_trusted": True,
        "supports_home": bool(supports_home),
        "gripper_gap_measured": True,
        "gripper_open_span_m": GRIPPER_OPEN_SPAN_M,
        "gripper_pad_length_m": GRIPPER_PAD_LENGTH_M,
        "table_z_m": table_z_m,
        "comfortable_reach_m": 0.727,
        "reach_m": 0.78,
        "tilt_limits_deg": {
            "roll_cw": 88.6,
            "roll_ccw": 83.7,
            "pitch_down": 91.7,
            "pitch_up": 53.6,
        },
        "tilted_floor_m": 0.100,
        "tilt_reach_deg": 80.0,
        "max_object_height_m": 0.16,
        "subgoal_median_s": 21.0,
        "subgoal_verdict_s": 4.6,
        "arrival_tolerance_m": ARRIVE_M,
        "mission_budget_s": None,
        "floor_above_table_m": None,
        "depth_min_range_m": DEPTH_MIN_RANGE_M,
        "control_dt_s": dt,
        "notes": notes,
    }
