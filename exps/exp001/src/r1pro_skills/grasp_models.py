"""Grasp-pose models run on the wrist-camera view inside ``pick_object``.

After the hand hovers above the object, SAM3 segments the object in that arm's
wrist camera. A grasp model then proposes 6-DoF grasps for that segment:

- ``contact_graspnet``: the baseline's Contact-GraspNet service
  (``grasp_net_plan_fn``) on the wrist depth image and SAM3 mask.
- ``graspgenx``: GraspGen-X (NVlabs, galaxea_g1 = the R1Pro hand) on the
  wrist-camera object point cloud, through ``graspgenx_server.py``.
- ``geometric``: no model; ``pick_object`` keeps its top-down point-cloud grasp.

Both models output gripper poses with +Z the approach and +X the finger closing
direction. The R1Pro end-effector frame (measured in the simulator, see
BEHAVIOR.md v10-v12) has +Z the approach and +Y the closing direction,
with its origin at the fingertips, so a model pose maps to the end effector by
moving ``EEF_DEPTH`` along +Z and turning -90 degrees about Z.

Candidates are screened with IK only (no motion) before any is executed, so an
unreachable proposal costs no simulation steps.
"""
from __future__ import annotations

import json
import os
import urllib.request

import numpy as np

PLANNER_ENV = "EXP001_GRASP_PLANNER"
PLANNERS = ("geometric", "contact_graspnet", "graspgenx")
GRASPGENX_URL = os.environ.get("EXP001_GRASPGENX_URL", "http://127.0.0.1:8390")
# Distance from the model's gripper origin to the R1Pro end-effector origin
# along the approach axis. GraspGen-X galaxea_g1: palm (left_gripper_link) at
# z=0.0767, measured EEF 0.060 m beyond the palm. Contact-GraspNet (Panda):
# fingertips at 0.1034, the EEF sits 0.0033 m short of the fingertips.
EEF_DEPTH = {"graspgenx": 0.1367, "contact_graspnet": 0.100}
MAX_TILT_DEG = 70.0       # reject approaches more than this far from straight down
PREGRASP_BACKOFF = 0.10   # m back along the approach axis
SCREEN_LIMIT = 30         # candidates screened with IK per observation
EXECUTE_LIMIT = 2         # feasible candidates executed per observation


def planner_name(planner: str | None = None) -> str:
    name = planner or os.environ.get(PLANNER_ENV, "geometric")
    if name not in PLANNERS:
        raise ValueError(f"grasp planner must be one of {PLANNERS}, got {name!r}")
    return name


def model_to_eef(transform: np.ndarray, depth: float) -> tuple[np.ndarray, np.ndarray]:
    """World (XYZ, XYZW) end-effector pose for a model gripper pose."""
    from scipy.spatial.transform import Rotation
    transform = np.asarray(transform, dtype=float)
    rotation = transform[:3, :3]
    position = transform[:3, 3] + rotation[:, 2] * depth
    eef = Rotation.from_matrix(rotation) * Rotation.from_euler("z", -np.pi / 2)
    return position, eef.as_quat()


def plan_contact_graspnet(self, view) -> tuple[np.ndarray, np.ndarray]:
    from capx.integrations.r1pro import control as c
    depth = np.where(np.isfinite(view["depth"]) & (view["depth"] > 0), view["depth"], 0)
    mask = view["mask"]
    if np.count_nonzero(mask & (depth >= 0.2) & (depth <= 2.0)) < 6:
        print("[grasp-model] contact_graspnet: object outside the 0.2-2.0 m depth window", flush=True)
        return np.zeros((0, 4, 4)), np.zeros(0)
    samples, scores, _ = self.grasp_net_plan_fn(depth, view["intrinsics"], mask.astype(np.uint8), 1)
    samples, scores = np.asarray(samples, dtype=float), np.asarray(scores, dtype=float)
    if not len(samples):
        return np.zeros((0, 4, 4)), np.zeros(0)
    camera = np.asarray(view["camera"], dtype=float)
    world = np.stack([camera @ c.convert_T_cam_cv_to_cam_gl(sample) for sample in samples])
    return world, scores.reshape(-1)


def plan_graspgenx(self, view) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(view["points"], dtype=float)
    if len(points) > 4000:
        points = points[np.random.default_rng(0).choice(len(points), 4000, replace=False)]
    request = urllib.request.Request(
        f"{GRASPGENX_URL}/plan", json.dumps({"points": points.tolist(), "num_grasps": 200, "topk": 100}).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        reply = json.loads(response.read())
    return np.asarray(reply["grasps"], dtype=float).reshape(-1, 4, 4), np.asarray(reply["scores"], dtype=float)


def propose_grasps(self, view, planner: str, arm: int) -> list[tuple[tuple, tuple, float]]:
    """IK-feasible (pregrasp, grasp, score) candidates from the model, best first."""
    from r1pro_skills.ik import IKSolverTimeout
    plan = plan_graspgenx if planner == "graspgenx" else plan_contact_graspnet
    points = np.asarray(view["points"])
    print(f"[grasp-model] wrist segment: {len(points)} points, median {np.round(np.median(points, axis=0), 3)}, "
          f"extent {np.round(points.max(axis=0) - points.min(axis=0), 3)}", flush=True)
    debug_dir = os.environ.get("EXP001_GRASP_DEBUG_DIR")
    if debug_dir and "rgb" in view:
        import cv2
        os.makedirs(debug_dir, exist_ok=True)
        overlay = np.asarray(view["rgb"], dtype=np.uint8)[..., ::-1].copy()
        overlay[view["mask"]] = (0.5 * overlay[view["mask"]] + [0, 0, 127]).astype(np.uint8)
        index = len(os.listdir(debug_dir))
        cv2.imwrite(os.path.join(debug_dir, f"wrist_{index:02d}.png"), overlay)
    try:
        transforms, scores = plan(self, view)
    except OSError as error:   # service down or unreachable: keep the geometric grasp
        print(f"[grasp-model] {planner} unavailable ({error}); falling back to the top-down grasp", flush=True)
        return []
    order = np.argsort(-scores)
    print(f"[grasp-model] {planner}: {len(transforms)} grasps (best score "
          f"{scores.max() if len(scores) else float('nan'):.3f})", flush=True)
    robot, _, _ = self.get_robot_position()
    robot = np.asarray(robot, dtype=float)
    feasible, screened, tilted = [], 0, 0
    for index in order:
        approach = transforms[index][:3, 2]
        tilt = float(np.degrees(np.arccos(np.clip(-approach[2], -1, 1))))
        if tilt > MAX_TILT_DEG:
            tilted += 1
            continue
        position, quat = model_to_eef(transforms[index], EEF_DEPTH[planner])
        pregrasp = position - approach * PREGRASP_BACKOFF
        if np.linalg.norm(position[:2] - robot[:2]) > self._grasp_reach_limit:
            continue
        screened += 1
        residuals = []
        for target in (pregrasp, position + np.array([0.0, 0.0, -0.02])):
            try:
                self.solve_ik(target, quat[[3, 0, 1, 2]], arm=arm)
                residuals.append(self.last_ik_pos_residual)
            except IKSolverTimeout:
                residuals.append(float("inf"))
        ok = all(r is not None and np.isfinite(r) and r <= self._ik_residual_limit for r in residuals)
        print(f"[grasp-model]   #{index} score {scores[index]:.3f} tilt {tilt:.0f} deg "
              f"tip {np.round(position, 3)} IK residual {np.round(residuals, 3)}"
              + (" ok" if ok else ""), flush=True)
        if ok:
            feasible.append(((pregrasp, quat), (position, quat), float(scores[index])))
            if len(feasible) >= EXECUTE_LIMIT:
                break
        if screened >= SCREEN_LIMIT:
            break
    print(f"[grasp-model] {planner}: {len(feasible)} reachable of {screened} screened "
          f"({tilted} rejected as side/under approaches)", flush=True)
    return feasible
