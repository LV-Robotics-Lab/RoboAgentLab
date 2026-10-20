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

"""MotorMind mode's LIBERO rig: the pure functions (segment judgement, lateral limits,
grasp_state, table histogram, side mount geometry, clearance cylinder, projection, images) and
the facade's env.mm_observe / env.mm_run_step over a kinematic stand-in for the LIBERO worker
(no simulator; the real-LIBERO checks are pi's gpu-e2e `libero-mm`)."""

from __future__ import annotations

import base64
import io
import json
import math
import time

import numpy as np
import pytest
from PIL import Image

from pi_embodied_services.robots.libero import motormind_rig as rig
from pi_embodied_services.robots.libero.env_server import LiberoEnvFacade


def rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array(
        [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    )


DOWN = np.diag([1.0, -1.0, -1.0])  # jaws pointing down (half a turn about x)


def seg(**kw) -> rig.Segment:
    base = {
        "kind": "translate",
        "goal_p": np.array([0.05, 0.0, 0.0]),
        "goal_r": DOWN,
        "speed": 0.05,
        "axis": np.array([1.0, 0.0, 0.0]),
        "distance_m": 0.05,
    }
    base.update(kw)
    return rig.Segment(**base)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def test_lateral_limit_follows_the_four_rules():
    x, z = np.array([1.0, 0, 0]), np.array([0, 0, -1.0])
    assert rig.lateral_limit(0.05, False, x) == pytest.approx(0.015)  # 30% of 50 mm
    assert rig.lateral_limit(0.005, False, x) == pytest.approx(0.004)  # the 4 mm floor
    assert rig.lateral_limit(0.02, True, x) == pytest.approx(0.015)  # holding: >= 15 mm
    assert rig.lateral_limit(0.05, False, z) == pytest.approx(
        0.005
    )  # vertical: <= 5 mm
    assert rig.lateral_limit(0.05, True, z) == pytest.approx(0.015)  # payload wins
    assert rig.lateral_limit(0.0, False, x) is None
    # Vertical is the original's |axis_z| > 0.99: 10 deg off vertical is a slanted move.
    slant = np.array([math.sin(math.radians(10)), 0, -math.cos(math.radians(10))])
    assert not rig.is_vertical(slant) and rig.is_vertical([0, 0.1, -0.995])
    assert rig.lateral_limit(0.05, False, slant) == pytest.approx(0.015)


def test_servo_step_caps_both_channels_at_speed_dt_and_arrives_within_tolerance():
    s = seg(speed=0.025)
    dp, dr, arrived = rig.servo_step(np.zeros(3), DOWN, s, 0.05, 0.5)
    assert not arrived and np.linalg.norm(dp) == pytest.approx(0.025 * 0.05)
    assert np.allclose(dr, 0)
    # The orientation hold of a translation also steps speed * dt (m/s read as rad/s).
    tilted = rig.axis_angle_matrix([0.2, 0, 0]) @ DOWN
    _dp, dr, _ = rig.servo_step(np.zeros(3), tilted, s, 0.05, 0.5)
    assert np.linalg.norm(dr) == pytest.approx(0.025 * 0.05)
    near = s.goal_p - np.array([0.0029, 0, 0])
    assert rig.servo_step(
        near, rig.axis_angle_matrix([0.029, 0, 0]) @ DOWN, s, 0.05, 0.5
    )[2]
    # A rotation at 60 deg/s steps at most 3 deg (and translation min(speed dt, 50 mm)).
    rot = seg(
        kind="rotate",
        goal_p=np.zeros(3),
        goal_r=rz(30) @ DOWN,
        speed=math.radians(60),
        axis=None,
    )
    _dp, dr, _ = rig.servo_step(np.zeros(3), DOWN, rot, 0.05, 0.5)
    assert np.linalg.norm(dr) == pytest.approx(math.radians(3))
    assert dr[2] > 0  # yaw left: about +Z of the base


def test_osc_action_normalises_by_output_max_and_turns_world_frame_deltas():
    om = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
    a = rig.osc_action([0.0025, 0, 0], [0, 0, 0.1], om, 0.0)
    assert np.allclose(a, [0.05, 0, 0, 0, 0, 0.2, 0.0])
    big = rig.osc_action([1.0, 0, 0], [0, 0, 0], om, 1.0)
    assert big[0] == 1.0 and big[6] == 1.0
    w = rig.osc_action(
        [0.0025, 0, 0], [0, 0, 0], om, -1.0, base_r=rz(90), ref_frame="world"
    )
    assert np.allclose(w[:3], [0, 0.05, 0])


def advance(steps):
    """Poses from a list of per-tick x advances (m), jaws down."""
    p, out = np.zeros(3), [(np.zeros(3), DOWN)]
    for d in steps:
        p = p + np.array([d, 0.0, 0.0])
        out.append((p.copy(), DOWN))
    return out


def test_a_translation_that_stops_advancing_is_blocked_after_eight_slow_ticks():
    poses = advance([0.0025] * 4 + [0.0] * 20)
    out = rig.replay_segment(seg(), poses)
    assert out["abort_reason"] == "blocked"
    # Ticks 5..12 are slow (after the 3rd tick, < 25% of the best, > 9 mm to go).
    assert out["abort_tick"] == 12
    assert out["outcome"] == "limited" and out["along_axis_m"] == pytest.approx(0.01)
    pushed = rig.replay_segment(seg(push=True), poses)
    assert pushed["outcome"] == "contact" and "pushed" in pushed["message"]
    # Within 9 mm of the goal a slow tick is no stall.
    late = advance([0.0025] * 17 + [0.0] * 20)
    assert rig.replay_segment(seg(), late)["abort_reason"] == ""


def test_a_tick_outside_the_stall_conditions_keeps_the_count():
    # The goal is 9.5 mm away after four 2.5 mm ticks; four slow ticks count, a sideways step
    # brings the tool within 9 mm of the goal for three ticks (the count waits), then back out:
    # four more slow ticks make eight, at tick 15 (a reset count would need tick 19).
    s = seg(goal_p=np.array([0.01, 0.0095, 0.0]))
    xs = [0.0025 * i for i in range(1, 5)] + [0.01] * 11
    ys = [0.0] * 8 + [0.001] * 3 + [0.0] * 4
    poses = [(np.zeros(3), DOWN)] + [
        (np.array([x, y, 0.0]), DOWN) for x, y in zip(xs, ys)
    ]
    out = rig.replay_segment(s, poses)
    assert out["abort_reason"] == "blocked" and out["abort_tick"] == 15


def test_sideways_drift_beyond_the_limit_skids_and_a_downward_one_is_blocked():
    poses = [(np.zeros(3), DOWN)]
    for i in range(1, 40):
        poses.append((np.array([0.002 * i, 0.001 * i, 0.0]), DOWN))
    out = rig.replay_segment(seg(), poses)
    assert out["abort_reason"] == "skidded" and out["outcome"] == "limited"
    assert out["lateral_m"] > 0.015 and out["abort_tick"] == 16
    down = seg(goal_p=np.array([0, 0, -0.05]), axis=np.array([0, 0, -1.0]))
    drift = [(np.zeros(3), DOWN)] + [
        (np.array([0.0012 * i, 0.0, -0.0004 * i]), DOWN) for i in range(1, 20)
    ]
    out = rig.replay_segment(down, drift)
    # A vertical move without a payload may drift 5 mm; it had advanced <= 3 mm: blocked.
    assert out["abort_reason"] == "blocked" and out["abort_tick"] == 5


def test_no_motion_for_more_than_twelve_ticks_ends_the_loop_without_an_abort():
    poses = advance([0.0] * 30)
    judge = rig.SegmentJudge(seg(kind="rotate", axis=None), np.zeros(3), DOWN)
    ends = [judge.tick(p, r) for p, r in poses[1:14]]
    assert ends[-1] == "creep" and judge.crept and judge.abort == ""
    assert ends[:-1] == [""] * 12


def test_segment_result_outcomes():
    s = seg()
    done = rig.segment_result(
        s, np.zeros(3), DOWN, np.array([0.048, 0.001, 0]), DOWN, ""
    )
    assert done["outcome"] == "done" and done["along_axis_m"] == pytest.approx(0.048)
    assert done["lateral_m"] == pytest.approx(0.001)
    short = rig.segment_result(s, np.zeros(3), DOWN, np.array([0.03, 0, 0]), DOWN, "")
    assert short["outcome"] == "limited"
    push = rig.segment_result(
        seg(push=True), np.zeros(3), DOWN, np.array([0.05, 0, 0]), DOWN, ""
    )
    assert push["outcome"] == "done" and "touched nothing" in push["message"]
    turn = rig.segment_result(
        seg(kind="rotate", goal_p=np.zeros(3), goal_r=rz(30) @ DOWN, axis=None),
        np.zeros(3),
        DOWN,
        np.zeros(3),
        rz(30) @ DOWN,
        "",
    )
    assert turn["outcome"] == "done" and turn["turned_rad"] == pytest.approx(
        math.radians(30)
    )
    assert turn["along_axis_m"] == turn["moved_m"] == 0 and turn["lateral_m"] == 0


def test_grasp_state():
    assert rig.grasp_state("open", 0.001, False)["holding"] is None
    assert rig.grasp_state("hold", 0.04, False)["holding"] is None
    assert rig.grasp_state("close", 0.021, True)["holding"] is None  # still closing
    assert (
        rig.grasp_state("close", 0.075, False)["holding"] is None
    )  # close did not happen
    empty = rig.grasp_state("close", 0.003, False)
    assert empty["holding"] is False and empty["opening_mm"] == pytest.approx(3.0)
    assert rig.grasp_state("close", 0.002, True)["holding"] is False
    held = rig.grasp_state("close", 0.031, False)
    assert held["holding"] is True and 0 < held["closed_fraction"] < 1
    assert rig.opening_m([0.039, -0.039]) == pytest.approx(0.078)
    # closed_fraction is 0 at the closed end and 1 wide open (the original's field).
    assert rig.grasp_state("open", 0.0788, False)["closed_fraction"] == pytest.approx(
        1.0
    )
    assert rig.grasp_state("close", 0.0012, False)["closed_fraction"] == pytest.approx(
        0.0152, abs=1e-4
    )


def test_table_height_is_the_refined_mode_and_needs_a_hundred_points():
    rng = np.random.default_rng(0)
    z = np.concatenate(
        [-0.012 + rng.normal(0, 0.0005, 3000), rng.uniform(-0.3, 0.4, 1500)]
    )
    assert rig.table_height(z) == pytest.approx(-0.012, abs=0.001)
    assert rig.table_height(z[:99]) is None
    assert rig.table_height(np.full(150, np.nan)) is None


def test_the_table_top_is_the_static_geom_most_free_bodies_stand_on():
    table = {
        "name": "table",
        "body": "table_body",
        "type": rig.GEOM_BOX,
        "pos": [0, 0, 0.875],
        "xmat": np.eye(3),
        "size": [0.5, 0.6, 0.025],
    }
    floor = {
        "name": "floor",
        "type": rig.GEOM_PLANE,
        "pos": [0, 0, 0],
        "xmat": np.eye(3),
        "size": [3, 3, 0.125],
    }
    stove = {
        "name": "stove",
        "type": rig.GEOM_BOX,
        "pos": [-0.25, -0.13, 0.905],
        "xmat": np.eye(3),
        "size": [0.095, 0.095, 0.02],
    }
    mesh = {
        "name": "mesh",
        "type": 7,
        "pos": [0, 0, 2],
        "xmat": np.eye(3),
        "size": [9, 9, 9],
    }
    bodies = [[0.0, 0.2, 0.898], [0.05, 0.02, 0.909], [-0.19, 0.19, 0.899]]
    top = rig.table_top([floor, mesh, table, stove], bodies)
    # The floor carries every body too: the tie goes to the higher top.
    assert (
        top["name"] == "table"
        and top["body"] == "table_body"
        and top["top_z"] == pytest.approx(0.9)
        and top["count"] == 3
    )
    # A body 3 cm or more below a top does not stand on it.
    assert rig.table_top([table], [[0.0, 0.0, 0.86]]) is None
    tilted = dict(table, xmat=rz(90))
    _top, fp = rig.geom_top_and_footprint(
        rig.GEOM_BOX, [0, 0, 0], rz(90), [0.5, 0.6, 0.025]
    )
    assert fp == pytest.approx((-0.6, 0.6, -0.5, 0.5))
    assert rig.table_top([tilted], [[0.0, 0.55, 0.9]]) is None
    assert rig.table_top([tilted], [[0.55, 0.0, 0.9]])["count"] == 1


def quat_wxyz_matrix(q):
    w, x, y, z = q
    return rig.quat_xyzw_matrix([x, y, z, w])


def test_the_side_mount_rule_and_its_look_at():
    base = np.array([-0.66, 0.0, 0.912])
    agent = np.array([0.6586, 0.0, 1.6104])
    m = rig.side_mount(base, 0.9, agent)
    assert np.allclose(m["aim"], [-0.19, 0.0, 0.9])
    d = np.linalg.norm(agent - m["aim"])
    assert m["standoff"] == pytest.approx(d)
    off = m["eye"] - m["aim"]
    assert off[0] == pytest.approx(0) and off[1] > 0  # on the base's +Y side
    assert math.degrees(math.atan2(off[2], off[1])) == pytest.approx(35.0)
    assert np.linalg.norm(off) == pytest.approx(d)
    xmat = quat_wxyz_matrix(m["quat_wxyz"])
    k = rig.intrinsic(45.0, 768, 768)
    uv = rig.project(k, rig.cam2world(m["eye"], xmat), m["aim"])
    assert uv[0] == pytest.approx(384) and uv[1] == pytest.approx(384)
    # World up stays up in the picture: a point above the aim is above the centre.
    above = rig.project(k, rig.cam2world(m["eye"], xmat), m["aim"] + [0, 0, 0.1])
    assert above[1] < 384 and above[0] == pytest.approx(384)
    assert np.isclose(np.linalg.det(xmat), 1)


def test_intrinsic_projection_and_back_projection_round_trip():
    k = rig.intrinsic(45.0, 768, 768)
    assert k[0, 0] == pytest.approx(0.5 * 768 / math.tan(math.radians(22.5)))
    assert k[0, 2] == 384 and k[1, 2] == 384
    c2w = rig.cam2world(
        [0.1, 0.2, 1.5], np.eye(3)
    )  # MuJoCo camera looking down world -z
    assert np.allclose(c2w[:3, 2], [0, 0, -1])  # OpenCV z forward
    uv = rig.project(k, c2w, [0.1, 0.2, 0.9])
    assert uv == pytest.approx((384, 384, 0.6))
    # +x world is image right, +y world is image up (row decreases).
    right = rig.project(k, c2w, [0.2, 0.2, 0.9])
    up = rig.project(k, c2w, [0.1, 0.3, 0.9])
    assert right[0] > 384 and up[1] < 384
    pts, z = rig.back_project(np.full((768, 768), 0.6), k, c2w, stride=2)
    assert np.allclose(pts[:, 2], 0.9) and np.allclose(z, 0.6)
    assert rig.project(k, c2w, [0.1, 0.2, 1.6]) is None  # behind


def test_depth_is_quantised_like_the_16_bit_millimetre_image():
    d = rig.quantize_depth(np.array([0.2109, np.nan, 70.0, -1.0]))
    assert d.tolist() == pytest.approx([0.21, 0.0, 65.535, 0.0])


def wrist_points(depth_m, tool_z: float = 1.2):
    """A wrist camera at the tool point looking straight down (``depth_m``: a number for a flat
    table, or a depth image): the returns in the base frame and in the camera frame."""
    k = rig.intrinsic(75.0, 768, 768)
    c2w = rig.cam2world([0.0, 0.0, tool_z], np.eye(3))
    d = np.broadcast_to(np.asarray(depth_m, dtype=np.float64), (768, 768))
    cam, _z = rig.back_project(d, k, np.eye(4))
    return cam @ c2w[:3, :3].T + c2w[:3, 3], cam


def test_clearance_reads_the_cylinder_below_the_tool():
    pts, cam = wrist_points(0.30)
    out = rig.clearance(pts, cam, [0, 0, 1.2], DOWN)
    assert out["seen"] and out["mm"] == pytest.approx(300.0, abs=0.01)
    assert out["surface_z_m"] == pytest.approx(0.9) and out["jaws_down"]
    assert out["n_points"] >= rig.CLEARANCE_MIN_POINTS and not out["near_min_range"]
    near = rig.clearance(*wrist_points(0.25), [0, 0, 1.2], DOWN)
    assert near["seen"] and near["near_min_range"]
    # Inside the minimum range nothing returns: blind, a floor and not a surface.
    blind = rig.clearance(*wrist_points(0.15), [0, 0, 1.2], DOWN)
    assert not blind["seen"] and blind["at_range_floor"] and blind["mm"] is None
    assert blind["near_min_range"]
    floor = rig.clearance(*wrist_points(0.2115), [0, 0, 1.2], DOWN)
    assert not floor["seen"] and floor["at_range_floor"]
    # Points off the axis do not count.
    few = rig.clearance(pts + [0.5, 0, 0], cam, [0, 0, 1.2], DOWN)
    assert not few["seen"] and few["n_points"] == 0


def test_clearance_compares_depth_not_ray_length_and_counts_only_the_band_under_it():
    # An object top at 0.209 m depth under the tool, the table at 0.30 m around it: its returns
    # sit in the 2 mm band under the minimum range, so the table beside it is no reading.
    d = np.full((768, 768), 0.30)
    d[364:404, 364:404] = 0.209
    out = rig.clearance(*wrist_points(d), [0, 0, 1.2], DOWN)
    assert not out["seen"] and out["n_cut"] >= 10 and out["at_range_floor"]
    # The fingertips about 0.1 m from the lens are below the band: they neither count nor cut.
    d = np.full((768, 768), 0.30)
    d[364:404, 300:330] = 0.105
    fingers = rig.clearance(*wrist_points(d), [0, 0, 1.2], DOWN)
    assert fingers["seen"] and fingers["n_cut"] == 0
    # Near the minimum range is the mean return's distance from the lens (0.262 m here), not
    # the mean ray length (longer off the axis: 0.2635 m, past 1.25 x 0.21).
    pts, cam = wrist_points(0.262)
    out = rig.clearance(pts, cam, [0, 0, 1.2], DOWN)
    assert out["seen"] and out["near_min_range"]


def test_the_aim_spot_is_the_table_then_the_clearance_then_tracked_then_200_mm():
    p = [0.5, 0.0, 0.3]
    spot, depth, source = rig.aim_point(p, DOWN, 0.05)
    assert source == "table" and depth == pytest.approx(0.25)
    assert spot == pytest.approx([0.5, 0.0, 0.05])
    # Below the table's height + 20 mm the spot stays 20 mm under the tool.
    assert rig.aim_point([0.5, 0, 0.06], DOWN, 0.05)[1] == pytest.approx(0.02)
    # Within 45 deg of vertical the axis counts as straight down.
    tilt = rig.axis_angle_matrix([0, math.radians(30), 0]) @ DOWN
    assert rig.aim_point(p, tilt, 0.05)[0] == pytest.approx([0.5, 0.0, 0.05])
    # Tilted further: no table level; the clearance, then the tracked one, then 0.2 m along it.
    side = rig.axis_angle_matrix([0, math.radians(60), 0]) @ DOWN
    a = rig.approach_axis(side)
    spot, depth, source = rig.aim_point(p, side, 0.05, clearance_mm=120.0)
    assert (source, depth) == ("clearance", pytest.approx(0.12))
    assert spot == pytest.approx(np.asarray(p) + a * 0.12)
    assert rig.aim_point(p, side, 0.05, tracked_m=0.09)[1:] == (0.09, "tracked")
    assert rig.aim_point(p, side, None)[1:] == (0.2, "default")
    assert rig.aim_point(p, DOWN, None)[1:] == (0.2, "default")


def decode(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def test_the_model_pictures_carry_the_magenta_aim_marker():
    rgb = np.full((768, 768, 3), 90, dtype=np.uint8)
    imgs = rig.camera_images(rgb, (500.0, 200.0))
    marked = decode(imgs["marked768"])
    raw = decode(imgs["raw512"])
    assert marked.shape == (768, 768, 3) and raw.shape == (512, 512, 3)
    assert decode(imgs["marked512"]).shape == (512, 512, 3)
    ring = marked[200, 500 + 11]  # on the ring (radius ~10.8 px at 768)
    assert ring[0] > 200 and ring[2] > 200 and ring[1] < 80
    assert np.abs(raw.astype(int) - 90).max() < 6  # no marker on the raw picture
    unmarked = rig.camera_images(rgb, None)
    assert np.abs(decode(unmarked["marked768"]).astype(int) - 90).max() < 6
    assert decode(rig.mosaic([imgs["_marked"]] * 3)).shape == (256, 768, 3)


def test_rotations():
    r = rig.axis_angle_matrix([0.1, -0.2, 0.3])
    assert np.allclose(rig.rotvec(r), [0.1, -0.2, 0.3])
    assert np.allclose(
        rig.rotvec(rig.axis_angle_matrix([math.pi, 0, 0])), [math.pi, 0, 0]
    )
    q = rig.matrix_quat_xyzw(r)
    assert np.allclose(rig.quat_xyzw_matrix(q), r)
    assert np.allclose(quat_wxyz_matrix(rig.matrix_quat_wxyz(r)), r)
    assert rig.angle_between(np.eye(3), rz(20)) == pytest.approx(math.radians(20))
    assert rig.jaws_down(DOWN) and not rig.jaws_down(
        rig.axis_angle_matrix([0, math.radians(60), 0]) @ DOWN
    )


def test_caps_copy_the_original_table_and_follow_the_home_mode():
    caps = rig.libero_caps([0.05] * 3 + [0.5] * 3, 0.05, False, -0.012)
    assert caps["max_speed_m_s"] == pytest.approx(1.0)
    assert caps["max_rotation_speed_rad_s"] == pytest.approx(10.0)
    assert caps["reach_m"] == 0.78 and caps["gripper_open_span_m"] == 0.0788
    assert caps["tilt_limits_deg"]["pitch_up"] == 53.6 and caps["table_z_m"] == -0.012
    assert not caps["supports_home"] and "no home" in caps["notes"][-1]
    assert "after that no action is accepted" in caps["notes"][1]
    assert rig.libero_caps([0.05] * 6, 0.05, True, None)["supports_home"]


# ---------------------------------------------------------------------------
# The facade over a kinematic LIBERO worker stand-in
# ---------------------------------------------------------------------------

BASE_P = np.array([-0.66, 0.0, 0.912])
BASE_R = rz(90.0)  # turned, so base and world axes differ
TABLE_Z = 0.9


class MmArm:
    """A LiberoEnv stand-in with its worker: OSC deltas in the base frame (output_max 0.05 m /
    0.5 rad per step), a table at z = 0.9 the tool cannot go below, an optional sideways drift
    per unit of travel, fingers that move 8 mm per step toward +1 closed (on an object of
    ``obj`` width) / -1 open, and 0 holding; truncation after ``max_steps`` steps."""

    def __init__(
        self, max_steps: int = 10_000, obj: float = 0.0005, success_at: int = 0
    ):
        self.max_steps = max_steps
        self.obj = obj
        self.success_at = success_at
        self.workers = [self]
        self.calls: list[str] = []
        self._fresh()

    def _fresh(self):
        self.pos = np.array([-0.2, 0.0, 1.1])
        self.rot = DOWN.copy()
        self.width = 0.078
        self.steps = 0
        self.drift = 0.0
        self.floor = TABLE_Z + 0.002

    def _quat(self):
        return rig.matrix_quat_xyzw(self.rot)

    @property
    def current_raw_obs(self):
        return [
            {
                "robot0_eef_pos": self.pos.copy(),
                "robot0_eef_quat": self._quat(),
                "robot0_gripper_qpos": np.array([self.width / 2, -self.width / 2]),
            }
        ]

    def _obs(self):
        st = np.zeros((1, 8), dtype=np.float32)
        st[0, 6:8] = [self.width / 2, -self.width / 2]
        return {
            "main_images": np.zeros((1, 4, 4, 3), dtype=np.uint8),
            "wrist_images": np.zeros((1, 4, 4, 3), dtype=np.uint8),
            "states": st,
        }

    def _info(self):
        done = bool(self.success_at) and self.steps >= self.success_at
        return {"episode": {"success_once": np.array([done])}}

    def reset(self):
        self._fresh()
        return self._obs(), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(7)
        self.steps += 1
        d = BASE_R @ (a[:3] * 0.05)
        new = self.pos + d
        new[0] += self.drift * np.linalg.norm(d)
        new[2] = max(new[2], self.floor)
        self.pos = new
        self.rot = rig.axis_angle_matrix(BASE_R @ (a[3:6] * 0.5)) @ self.rot
        if a[6] > 0:
            self.width = max(self.obj, self.width - 0.008)
        elif a[6] < 0:
            self.width = min(0.078, self.width + 0.008)
        trunc = np.array([self.steps >= self.max_steps])
        return self._obs(), np.zeros(1), np.zeros(1, dtype=bool), trunc, self._info()

    task_descriptions = ["stack it"]

    @property
    def env(self):
        return self

    # -- the worker's env_call --

    def env_call(self, method, args=None, kwargs=None, target="robosuite"):
        self.calls.append(method)
        args, kwargs = list(args or []), dict(kwargs or {})
        return getattr(self, "_w_" + method)(*args, **kwargs)

    def _w_mm_static(self):
        return {
            "base": {
                "body": "robot0_base",
                "pos": BASE_P.tolist(),
                "xmat": BASE_R.tolist(),
            },
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "ref_frame": "base",
            "controller": "Fake",
            "control_freq": 20.0,
            "horizon": self.max_steps + 15,
        }

    def _w_mm_mount_side(self, camera, agent):
        return {
            "mounted": True,
            "camera": camera,
            "parent": "world",
            "pre": {"pos": [0, 1.3, 1.5]},
        }

    def _w_mm_frames(self, cameras, size, depth=()):
        out = {}
        for name in cameras:
            pos = self.pos if name == "wrist" else np.array([0.0, 0.0, 1.5])
            e = {
                "rgb": np.full((size, size, 3), 100, dtype=np.uint8),
                "fovy": 75.0 if name == "wrist" else 45.0,
                "cam_xpos": pos.copy(),
                "cam_xmat": np.eye(3),
            }
            if name in depth:
                e["depth"] = np.full((size, size), pos[2] - TABLE_Z, dtype=np.float32)
            out[name] = e
        return out

    def _w_mm_truth(self, state=False):
        return {
            "objects": {"cube_1": {"pos": [0.1, 0.0, 0.92], "quat_xyzw": [0, 0, 0, 1]}},
            "success_now": False,
            "sim_time_s": self.steps * 0.05,
        }

    def _w_mm_rays(self, origins, directions):
        return [
            {"dist": o[2] - TABLE_Z, "geom": "table_visual", "body": "table"}
            for o in origins
        ]

    def _w_mm_scene(self):
        return {
            "table_top": {
                "name": "table_collision",
                "body": "table",
                "top_z": TABLE_Z,
                "count": 3,
            }
        }


def mm_facade(**kw) -> LiberoEnvFacade:
    arm_kw = {k: kw.pop(k) for k in ("max_steps", "obj", "success_at") if k in kw}
    f = LiberoEnvFacade(MmArm(**arm_kw), meta={}, **kw)
    f.reset()
    return f


def base_pose(f):
    return f._mm_pose()


def test_reset_mounts_the_side_camera_settles_and_returns_the_settle_steps():
    f = LiberoEnvFacade(MmArm(), meta={}, side_camera=True, settle_ticks=4)
    obs, info = f._rpc["env.reset"](tool_call=True)
    assert info["settle_ticks"] == 4 and len(info["transitions"]) == 4
    assert all(t["action"][6] == -1 for t in info["transitions"])
    assert f._env.calls.count("mm_mount_side") == 1 and f._record is None
    clock = f._mm_clock
    _obs, info = f.reset()
    assert "transitions" not in info and f._mm_episode == 2 and f._mm_ticks == 4
    assert f._mm_clock > clock  # the frame clock never goes back
    # Without the MotorMind flags a reset is the plain env reset.
    plain = LiberoEnvFacade(MmArm(), meta={})
    _obs, info = plain.reset()
    assert info["settle_ticks"] == 0 and plain._env.calls == []


def test_observation_is_in_the_base_frame_with_calibration_images_and_clearance():
    f = mm_facade(side_camera=True)
    out = f._rpc["env.mm_observe"]()
    p_world = f._env.pos
    assert np.allclose(out["pose"]["position_m"], BASE_R.T @ (p_world - BASE_P))
    assert np.allclose(out["pose"]["rotation"], BASE_R.T @ DOWN)
    assert out["base_pose"]["body"] == "robot0_base"
    assert np.allclose(out["base_pose"]["rotation"], BASE_R)
    assert out["table_z_m"] == pytest.approx(TABLE_Z - BASE_P[2], abs=0.001)
    assert out["degraded"] == {
        "table_z": False,
        "side_mount": False,
        "side_mount_reason": None,
    }
    assert not out["caps"]["supports_home"] and out["home_mode"] == "refuse"
    assert set(out["cameras"]) == {"scene", "side", "wrist"}
    wrist = out["cameras"]["wrist"]
    assert wrist["depth_min_range_m"] == 0.21 and wrist["width"] == 768
    c2b = np.asarray(wrist["cam2base"])
    assert np.allclose(c2b[:3, :3] @ c2b[:3, :3].T, np.eye(3))
    assert np.linalg.det(c2b[:3, :3]) == pytest.approx(1.0)
    # The wrist camera sits on the tool looking down at the table 0.2 m below: the aim marker
    # (the table under the tool) is the image centre, the clearance is 200 mm.
    assert wrist["aim_px"] == pytest.approx([384, 384], abs=1e-6)
    assert out["clearance"]["seen"] is False  # 0.2 m is inside the 0.21 m minimum range
    assert out["aim"]["source"] == "table" and out["aim"]["depth_m"] == pytest.approx(
        0.2
    )
    assert out["config"] == {
        "side_camera": True,
        "settle_ticks": 0,
        "mm_home": "refuse",
        "max_episode_steps": None,
        "truth": None,
    }
    assert decode(out["mosaic_jpeg"]).shape == (256, 768, 3)
    assert out["gripper"]["commanded"] == "open" and out["grasp"]["holding"] is None
    nxt = f.mm_observe(images=False)
    assert nxt["obs_id"] == out["obs_id"] + 1 and "cameras" not in nxt
    two = mm_facade().mm_observe()
    assert (
        set(two["cameras"]) == {"scene", "wrist"} and not two["degraded"]["side_mount"]
    )


def translate(f, axis_base, mm: float, speed=0.05, push=False, **kw):
    p0, r0 = base_pose(f)
    a = np.asarray(axis_base, dtype=np.float64)
    goal = p0 + a * mm / 1000
    return f._rpc["env.mm_run_step"](
        seg={
            "kind": "translate",
            "goal_position_m": goal.tolist(),
            "goal_rotation": r0.tolist(),
            "axis": a.tolist(),
            "distance_m": mm / 1000,
            "speed": speed,
            "push": push,
        },
        tool_call=True,
        **kw,
    )


def test_a_translation_moves_along_the_base_axis_at_the_commanded_speed():
    f = mm_facade()
    w0 = f._env.pos.copy()
    out = translate(
        f, [0, 1, 0], 50, speed=0.025, trace=True
    )  # base +Y = MotorMind left
    assert out["outcome"] == "done" and out["abort_reason"] == ""
    assert out["along_axis_m"] == pytest.approx(0.05, abs=0.003)
    assert out["lateral_m"] < 0.001 and out["axis"] == [0, 1, 0]
    # The world displacement is R_b times the base one (base turned 90 deg: world -x).
    assert np.allclose(f._env.pos - w0, BASE_R @ [0, 0.05, 0], atol=0.003)
    per_tick = [c[0] for c in out["trace"]["commanded"]]
    assert max(per_tick) <= 0.025 * 0.05 + 1e-9
    assert out["ticks"] == len(out["transitions"]) == out["servo_ticks"]
    assert out["observation"]["obs_id"] == out["obs_id"]
    replay = rig.replay_segment(
        rig.Segment(
            "translate",
            np.asarray(out["trace"]["segment"]["goal_position_m"]),
            np.asarray(out["trace"]["segment"]["goal_rotation"]),
            0.025,
            np.array([0, 1.0, 0]),
            0.05,
        ),
        [
            (np.asarray(q["position_m"]), np.asarray(q["rotation"]))
            for q in out["trace"]["poses"]
        ],
        end=(
            np.asarray(out["trace"]["end"]["position_m"]),
            np.asarray(out["trace"]["end"]["rotation"]),
        ),
    )
    assert replay["outcome"] == out["outcome"]
    assert replay["along_axis_m"] == pytest.approx(out["along_axis_m"])


def test_a_descent_onto_the_table_is_blocked_and_a_push_reports_contact():
    f = mm_facade()
    out = translate(f, [0, 0, -1], 50)  # 1.1 -> the floor at 0.902 is 198 mm down
    assert out["outcome"] == "done"
    out = translate(f, [0, 0, -1], 200)
    assert (out["outcome"], out["abort_reason"]) == ("limited", "blocked")
    assert out["hold_ticks"] == 2 and out["lateral_m"] <= 0.005
    assert out["ticks"] == out["servo_ticks"] + 2
    # Starting on the table nothing ever advances (no best tick): creep ends it, not a stall.
    stuck = translate(f, [0, 0, -1], 30, push=True)
    assert (
        stuck["crept"] and stuck["abort_reason"] == "" and stuck["outcome"] == "limited"
    )
    translate(f, [0, 0, 1], 10)
    pushed = translate(f, [0, 0, -1], 40, push=True)
    assert pushed["outcome"] == "contact" and "axis" not in pushed


def test_a_translation_pushed_sideways_skids():
    f = mm_facade()
    f._env.drift = 1.5  # more sideways than the servo can win back per step
    out = translate(f, [1, 0, 0], 50)
    assert (out["outcome"], out["abort_reason"]) == ("limited", "skidded")
    assert out["lateral_limit_m"] == pytest.approx(0.015)


def test_a_rotation_turns_about_the_base_axis():
    f = mm_facade()
    p0, r0 = base_pose(f)
    goal = rig.axis_angle_matrix([0, 0, math.radians(30)]) @ r0
    out = f.mm_run_step(
        seg={
            "kind": "rotate",
            "goal_rotation": goal.tolist(),
            "speed": math.radians(60),
        }
    )
    assert out["outcome"] == "done" and out["turned_rad"] == pytest.approx(
        math.radians(30), abs=0.03
    )
    assert out["moved_m"] < 0.003


def test_the_gripper_closes_judges_the_grasp_and_holds_a_partial_opening():
    f = mm_facade()
    empty = f.mm_run_step(seg={"kind": "gripper", "state": "close"})
    assert empty["ticks"] == 32 and empty["grasp"]["holding"] is False
    assert empty["detail"]["opening_mm"] <= 3.0 and "nothing held" in empty["message"]
    f2 = mm_facade(obj=0.03)
    held = f2.mm_run_step(seg={"kind": "gripper", "state": "close"})
    assert held["grasp"]["holding"] is True and held["gripper"]["commanded"] == "close"
    part = f2.mm_run_step(seg={"kind": "gripper", "state": "open", "open_to_mm": 40})
    assert (
        40 <= part["detail"]["opening_mm"] <= 48 and part["gripper"]["channel"] == 0.0
    )
    assert part["gripper"]["commanded"] == "hold" and part["grasp"]["holding"] is None
    before = part["detail"]["opening_mm"]
    up = translate(f2, [0, 0, 1], 50)
    # HOLD carries over: the move steps with gripper channel 0 and the gap stays.
    assert all(t["action"][6] == 0.0 for t in up["transitions"])
    assert up["gripper"]["opening_mm"] == pytest.approx(before)
    assert f2.get_state()["gripper_cmd"] == 0.0
    opened = f2.mm_run_step(seg={"kind": "gripper", "state": "open"})
    assert opened["gripper"]["opening_mm"] == pytest.approx(78.0)


def test_wait_steps_seconds_over_the_control_period():
    f = mm_facade()
    # The original's count: at least one step, the seconds over the period truncated.
    assert f.mm_run_step(seg={"kind": "wait", "seconds": 0.5})["ticks"] == 10
    assert f.mm_run_step(seg={"kind": "wait", "seconds": 0.3})["ticks"] == 5
    assert f.mm_run_step(seg={"kind": "wait", "seconds": 0.02})["ticks"] == 1


def test_home_is_refused_by_default_and_resets_the_scene_as_a_variant():
    f = mm_facade(settle_ticks=3)
    translate(f, [1, 0, 0], 20)
    ep, ticks = f._mm_episode, f._mm_ticks
    out = f.mm_run_step(seg={"kind": "home"})
    assert out["outcome"] == "refused" and out["ticks"] == 0
    assert (f._mm_episode, f._mm_ticks) == (ep, ticks)
    assert out["observation"]["home"] == {"asked": 1, "refused": 1, "reset": 0}
    g = mm_facade(settle_ticks=3, mm_home="reset", obj=0.03)
    g.mm_run_step(seg={"kind": "gripper", "state": "close"})
    clock = g._mm_clock
    out = g._rpc["env.mm_run_step"](seg={"kind": "home"}, tool_call=True)
    assert out["outcome"] == "done" and out["episode_reset"] and out["episode_id"] == 2
    # The settle steps keep the command from before the reset; open comes after them.
    assert [t["action"][6] for t in out["transitions"]] == [1.0, 1.0, 1.0]
    assert out["gripper"]["commanded"] == "open" and out["ticks"] == 3
    assert g._mm_clock > clock and out["observation"]["caps"]["supports_home"]
    assert out["observation"]["home"]["reset"] == 1
    with pytest.raises(ValueError, match="mm-home"):
        LiberoEnvFacade(MmArm(), meta={}, mm_home="arm")


def test_after_the_episode_ends_motion_is_refused():
    f = mm_facade(max_steps=12)
    out = translate(f, [1, 0, 0], 100)
    assert out["episode_over"] and out["truncated"] and out["ticks"] == 12
    again = translate(f, [1, 0, 0], 10)
    assert (
        again["outcome"] == "refused" and again["ticks"] == 0 and again["episode_over"]
    )


def test_a_segment_runs_to_its_end_through_the_success_latch():
    f = mm_facade(success_at=5)
    out = translate(f, [1, 0, 0], 50)
    # Success latched at the 5th step; the segment still ran to its goal (the original reads the
    # task at the segment's end).
    assert out["outcome"] == "done" and out["ticks"] > 5
    assert len(out["transitions"]) == out["ticks"]
    assert [t["terminated"] for t in out["transitions"]].index(True) == 4
    assert out["terminated"] and out["episode_over"] and not out["truncated"]
    again = translate(f, [1, 0, 0], 10)
    assert again["outcome"] == "refused" and again["ticks"] == 0
    # The hold after an abort and a gripper's settle run through it too.
    g = mm_facade(success_at=3)
    assert g.mm_run_step(seg={"kind": "gripper", "state": "close"})["ticks"] == 32


def test_a_tool_call_whose_reply_was_lost_hands_its_steps_back():
    f = mm_facade()
    first = translate(f, [1, 0, 0], 10, delivered=0)
    n1 = len(first["transitions"])
    assert first["call_id"] == 1 and first["late_steps"] == 0 and n1 > 0
    # pi never got that reply (it still says 0): the next call starts with its steps.
    second = translate(f, [1, 0, 0], 10, delivered=0)
    assert second["late_steps"] == n1 and second["call_id"] == 2
    assert len(second["transitions"]) == n1 + second["ticks"]
    # Delivered: nothing comes back twice.
    obs = f.mm_observe(images=False, delivered=2)
    assert obs["transitions"] == [] and obs["call_id"] is None
    third = translate(f, [1, 0, 0], 10, delivered=2)
    assert third["late_steps"] == 0
    # Lost again, the observation returns them and keeps them until it is acknowledged.
    lost = f.mm_observe(images=False, delivered=2)
    assert len(lost["transitions"]) == third["ticks"] and lost["call_id"] == 4
    assert f.mm_observe(images=False, delivered=4)["transitions"] == []
    # A caller that does not take part (no `delivered`) leaves what is kept alone.
    kept = translate(f, [1, 0, 0], 10, delivered=4)
    f.mm_observe(images=False)
    translate(f, [1, 0, 0], 10)
    assert len(f.mm_observe(images=False, delivered=0)["transitions"]) == kept["ticks"]


def test_bad_segments_are_errors():
    f = mm_facade()
    with pytest.raises(ValueError, match="kind"):
        f.mm_run_step(seg={"kind": "jump"})
    with pytest.raises(ValueError, match="zero"):
        f.mm_run_step(
            seg={
                "kind": "translate",
                "goal_position_m": [0, 0, 0],
                "axis": [0, 0, 0],
                "speed": 0.05,
            }
        )
    with pytest.raises(ValueError, match="speed"):
        translate(f, [1, 0, 0], 10, speed=0.0)
    with pytest.raises(ValueError, match="open_to_mm"):
        f.mm_run_step(seg={"kind": "gripper", "state": "close", "open_to_mm": 30})


def test_truth_lines_are_keyed_by_obs_id(tmp_path):
    path = tmp_path / "mm_truth.jsonl"
    f = mm_facade(mm_truth=str(path))
    a = f.mm_observe(images=False)
    b = translate(f, [1, 0, 0], 10)
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert [x["obs_id"] for x in lines] == [a["obs_id"], b["obs_id"]]
    assert lines[1]["kind"] == "segment" and lines[1]["segment"]["outcome"] == "done"
    assert (
        "objects" in lines[0] and "objects" not in a
    )  # truth never enters an observation


def test_the_test_rpcs_exist_only_when_privileged():
    f = mm_facade()
    assert not [m for m in f._rpc if m.startswith("env.mm_debug_")]
    g = mm_facade(privileged=True)
    assert {
        "env.mm_debug_project",
        "env.mm_debug_stall",
        "env.mm_debug_table",
        "env.mm_debug_truth",
        "env.mm_debug_set_body",
    } <= set(g._rpc)
    t = g._rpc["env.mm_debug_table"]()
    assert t["truth"]["ray_z_base_m"] == pytest.approx(TABLE_Z - BASE_P[2])
    assert t["truth"]["geom_top_z_base_m"] == pytest.approx(TABLE_Z - BASE_P[2])
    failed = g._rpc["env.mm_debug_table"](fail=True)
    assert failed["table_z_m"] is None and failed["degraded"]
    obs = g.mm_observe(images=False)
    assert obs["table_z_m"] is None and obs["degraded"]["table_z"]
    g._rpc["env.mm_debug_stall"](0.2)
    t0 = time.monotonic()
    g.mm_run_step(seg={"kind": "wait", "seconds": 0})
    assert time.monotonic() - t0 >= 0.2
    truth = g._rpc["env.mm_debug_truth"]()
    assert truth["objects"]["cube_1"]["pos_base"] == pytest.approx(
        (BASE_R.T @ (np.array([0.1, 0.0, 0.92]) - BASE_P)).tolist()
    )
    for f_ in (f, g):
        f_._manifest_ready()  # the MotorMind methods are declared internal in libero.json


def test_a_missing_base_body_fails_the_observation():
    arm = MmArm()
    arm._w_mm_static = lambda: {**MmArm._w_mm_static(arm), "base": {"body": None}}
    f = LiberoEnvFacade(arm, meta={})
    f.reset()
    with pytest.raises(RuntimeError, match="base pose"):
        f.mm_observe()
