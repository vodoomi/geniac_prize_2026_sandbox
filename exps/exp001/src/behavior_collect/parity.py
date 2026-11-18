"""Make in-process BEHAVIOR collection match the gateway evaluation.

At evaluation the submission only sees what the gateway sends (proprio
``joint_pos``/``cartesian_pos``, camera RGB/depth/intrinsics/pose) and drives the
robot with 25-dim raw actions through ``capx.submission.shims.r1pro``. The
in-process CaP-X low level, by contrast, can read simulator internals. This module
removes the internals that would otherwise leak into generated programs' feedback
and into the success labels:

* ``check_object_in_hand``: the simulator's assisted-grasp object
  (``_get_obj_in_hand``) is replaced by the shim's rule — the gripper was
  commanded closed and its fingers stay apart after settling.
* ``_navigate_to_pose``: cuRobo base planning against the scene's ground-truth
  collision meshes is replaced by the shim's straight-line holonomic servo (no map),
  using the same knobs as the competition ``env_configs/gateway/r1pro.yaml``.
* Scoring: the gateway ends a trial after ``trial_steplimit`` control cycles and
  counts it successful if the task was completed after any cycle within that
  limit. The same limit and the same "ever succeeded" latch are applied here, and
  stepping past the limit raises the gateway client's episode-over message, which
  ``capx.envs.trial`` already treats as truncation.

Ground truth is still read for scoring only (as the gateway does), never returned
to generated code.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

CAPX = Path(__file__).resolve().parents[4] / "third_party/geniac_prize_2026_baseline/geniac_prize_2026_code"
TASKPACK = CAPX / "taskpacks/original/pack.yaml"
GATEWAY_R1PRO = CAPX / "env_configs/gateway/r1pro.yaml"
# Gateway task ids whose BEHAVIOR activity has a different name.
ACTIVITY_TO_TASK = {"turning_on_radio": "picking_up_radio"}
_FINGERS = {0: (24, 25), 1: (26, 27)}


# Shim knobs this submission changes from the competition gateway config. The
# default base servo (0.15 m / 0.3 rad per 5-cycle RPC, no ramp) rocks the R1Pro
# 5-10 degrees on every start/stop of a translation even in the open, and 20-95
# degrees (a fall) next to furniture (BEHAVIOR.md, v10). 0.10 m / 0.2 rad keeps
# it under 2.5 degrees for ~20% more navigation cycles. The submission's r1pro.yaml carries the same
# values (behavior_collect/prepare_submission.py), so evaluation behaves the same.
KNOB_OVERRIDES = {"nav_max_lin": 0.10, "nav_max_ang": 0.2}


def submission_knob_values() -> dict:
    """The r1pro knob mapping written to the submission's gateway config."""
    config = yaml.safe_load(GATEWAY_R1PRO.read_text())
    return {**(config.get("r1pro") or {}), **KNOB_OVERRIDES}


def gateway_knobs():
    """The R1Pro shim knobs this submission runs with (see KNOB_OVERRIDES)."""
    from capx.submission.interfaces import R1ProKnobs
    return R1ProKnobs(**submission_knob_values())


def trial_steplimit(activity: str, pack_path: Path = TASKPACK) -> int:
    """``trial_steplimit`` of the public taskpack for a BEHAVIOR activity."""
    pack = yaml.safe_load(Path(pack_path).read_text())
    task_id = ACTIVITY_TO_TASK.get(activity, activity)
    for task in pack["tasks"]:
        if task["task_id"] == task_id and task["simulator"] == "behavior":
            return int(task.get("trial_steplimit", pack["limits"]["trial_steplimit"]))
    raise KeyError(f"BEHAVIOR task {task_id!r} is not in {pack_path}")


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def install_scoring(low_level_class) -> None:
    """Apply the gateway's step limit and success latch to the in-process sim."""
    from capx.submission.interfaces import EPISODE_OVER_MESSAGE

    original_reset = low_level_class.reset
    original_step = low_level_class.step
    original_completed = low_level_class.task_completed
    original_reward = low_level_class.compute_reward

    def reset(self, *args, **kwargs):
        self._exp001_success_cycle = None
        self._exp001_gripper_closed = {0: False, 1: False}
        # Read by CodeExecEnv.step, which reports truncation at this count.
        self.max_steps = trial_steplimit(self.task_name)
        return original_reset(self, *args, **kwargs)

    def step(self, action):
        if self._sim_step_count >= self.max_steps:
            raise RuntimeError(EPISODE_OVER_MESSAGE)
        from r1pro_skills.budget import check_block_budget
        check_block_budget(self)
        result = original_step(self, action)
        if getattr(self, "_exp001_success_cycle", None) is None and original_completed(self):
            self._exp001_success_cycle = self._sim_step_count
        return result

    def task_completed(self) -> bool:
        return getattr(self, "_exp001_success_cycle", None) is not None

    def at_default_arm(fn):
        # The trash/radio rewards read controller._get_obj_in_hand(), i.e. the
        # primitives' *current* arm. At evaluation the gateway drives raw actions
        # and never switches it, so only the robot's default (left) arm counts.
        # Score in-process with that same arm, whatever arm motion code selected.
        def scored(self):
            controller = getattr(self, "controller", None)
            selected = getattr(controller, "_selected_arm", None) if controller is not None else None
            if selected is not None:
                del controller._selected_arm
            try:
                return fn(self)
            finally:
                if selected is not None:
                    controller._selected_arm = selected
        return scored

    original_completed = at_default_arm(original_completed)
    original_reward = at_default_arm(original_reward)

    def compute_reward(self) -> float:
        return 1.0 if task_completed(self) else original_reward(self)

    low_level_class.reset = reset
    low_level_class.step = step
    low_level_class.task_completed = task_completed
    low_level_class.compute_reward = compute_reward


def install_grasp_check(low_level_class, knobs) -> None:
    """Replace the assisted-grasp query with the shim's proprio-only rule."""
    original_gripper = low_level_class._open_close_gripper

    def open_close_gripper(self, arm=0, max_steps=250, open=True):
        result = original_gripper(self, arm=arm, max_steps=max_steps, open=open)
        self._exp001_gripper_closed[int(arm)] = not open
        return result

    def check_object_in_hand(self, arm=0) -> bool:
        if arm not in (0, 1):
            raise ValueError("arm must be 0 (left) or 1 (right)")
        self._settle_robot()
        q = self.robot.get_joint_positions()
        i0, i1 = _FINGERS[arm]
        gap = float(q[i0] + q[i1])
        return bool(self._exp001_gripper_closed.get(arm, False) and gap > knobs.in_hand_gap_threshold)

    low_level_class._open_close_gripper = open_close_gripper
    low_level_class.check_object_in_hand = check_object_in_hand


# Stall detection for the straight-line servo. Without it a base pushing against
# furniture keeps commanding until NAV_MAX_ITERS (400 x 5 = 2000 cycles); with the
# public API's four fractional retries one blocked call could consume the whole
# trial budget and leave no turn for regeneration. A free base covers up to
# nav_max_lin (0.15 m) or nav_max_ang (0.3 rad) per iteration, so 10 iterations
# without 5 cm / 0.05 rad of improvement means it is blocked or oscillating.
# The submission shim must apply the same rule (see BEHAVIOR.md).
NAV_STALL_ITERS = 10
NAV_STALL_LIN = 0.05
NAV_STALL_ANG = 0.05


class NavigationProgress:
    """Detect a servo whose goal error stopped improving."""

    def __init__(self, window=NAV_STALL_ITERS, min_lin=NAV_STALL_LIN, min_ang=NAV_STALL_ANG):
        self.window, self.min_lin, self.min_ang = window, min_lin, min_ang
        self.checkpoint = None

    def stalled(self, iteration: int, lin_err: float, ang_err: float) -> bool:
        if self.checkpoint is None:
            self.checkpoint = (iteration, lin_err, ang_err)
            return False
        start, lin0, ang0 = self.checkpoint
        if lin0 - lin_err >= self.min_lin or ang0 - ang_err >= self.min_ang:
            self.checkpoint = (iteration, lin_err, ang_err)
            return False
        return iteration - start >= self.window


def clip_translation(dx: float, dy: float, max_lin: float) -> tuple[float, float]:
    """Scale (dx, dy) down so its length is at most max_lin (direction kept)."""
    length = math.hypot(dx, dy)
    if length <= max_lin or length <= 0.0:
        return dx, dy
    return dx * max_lin / length, dy * max_lin / length


def navigate_without_map(self, pose_2d, knobs, kinematics) -> bool:
    """``R1ProGatewayShim._navigate_to_pose`` on the in-process sim, plus stall detection."""
    import torch
    from capx.submission.shims import r1pro as shim

    goal_x, goal_y, goal_yaw = (float(v) for v in pose_2d)

    def pose():
        # Goals come from the API, i.e. the sim's world frame. The base's
        # virtual joints q[0:6] have a different origin, so errors use the world
        # pose and q is only used to hold the other joints still. The action's
        # base slots are base-frame deltas, which are frame independent.
        from scipy.spatial.transform import Rotation
        q = self.robot.get_joint_positions().detach().cpu().numpy().astype(np.float64)
        position, quat_xyzw = self.robot.get_position_orientation()
        position = position.detach().cpu().numpy().astype(np.float64)
        yaw = float(Rotation.from_quat(quat_xyzw.detach().cpu().numpy()).as_euler("xyz")[2])
        return q, float(position[0]), float(position[1]), yaw

    def converged():
        _, x, y, yaw = pose()
        return (math.hypot(goal_x - x, goal_y - y) < shim.NAV_LIN_TOL
                and abs(_wrap_to_pi(goal_yaw - yaw)) < shim.NAV_ANG_TOL)

    scale = 1.0 / shim.NAV_REPEAT
    monitor = NavigationProgress()
    for iteration in range(shim.NAV_MAX_ITERS):
        if converged():
            return True
        q, x, y, yaw = pose()
        lin_err = math.hypot(goal_x - x, goal_y - y)
        ang_err = abs(_wrap_to_pi(goal_yaw - yaw))
        if monitor.stalled(iteration, lin_err, ang_err):
            print(f"[navigate] no progress for {NAV_STALL_ITERS * shim.NAV_REPEAT} control cycles; "
                  f"stopped {lin_err:.2f} m / {ang_err:.2f} rad from the goal "
                  f"(path likely blocked by an obstacle)", flush=True)
            return False
        dx_world, dy_world = goal_x - x, goal_y - y
        cos_neg, sin_neg = math.cos(-yaw), math.sin(-yaw)
        dx_local = cos_neg * dx_world - sin_neg * dy_world
        dy_local = sin_neg * dx_world + cos_neg * dy_world
        action = kinematics.q28_to_action25(q, (float(q[0]), float(q[1]), float(q[5])))
        # Clip the translation as a vector, not per axis: per-axis clipping
        # drives diagonals sqrt(2) faster, which rocked the robot 7.8 degrees
        # vs 2.4 (BEHAVIOR.md, v10). The submission shim must
        # clip the same way (BEHAVIOR.md).
        dx_local, dy_local = clip_translation(dx_local, dy_local, knobs.nav_max_lin)
        action[0] = float(dx_local) * scale
        action[1] = float(dy_local) * scale
        action[2] = float(np.clip(_wrap_to_pi(goal_yaw - yaw), -knobs.nav_max_ang, knobs.nav_max_ang)) * scale
        command = torch.as_tensor(action, dtype=torch.float32)
        for _ in range(shim.NAV_REPEAT):
            self.step(command)
    return converged()


def install_motion(low_level_class, knobs, kinematics, step_chunk=None) -> None:
    """Joint servo, settle, gripper and torso primitives with the shim's cycle costs.

    The sim's own versions settle through OmniGibson's ``_settle_robot`` (at least
    50 and up to 550 cycles, repeated every few servo cycles) and run the gripper
    until *all* joints match. At evaluation the shim settles for
    ``knobs.settle_cycles`` (10) and servos in ``step_chunk`` blocks, so the same
    program consumed several times more of the step limit during collection than
    it would at evaluation. These ports follow ``capx.submission.shims.r1pro``.
    """
    import torch
    from capx.protocol.actions import MAX_REPEAT
    from capx.submission.interfaces import STEP_CHUNK_DEFAULT
    from capx.submission.shims import r1pro as shim

    chunk_size = int(step_chunk or STEP_CHUNK_DEFAULT)

    def q_now(self):
        return self.robot.get_joint_positions().detach().cpu().numpy().astype(np.float64)

    def hold(self, q28_cmd, cycles):
        q = q_now(self)
        command = torch.as_tensor(
            kinematics.q28_to_action25(q28_cmd, (float(q[0]), float(q[1]), float(q[5]))),
            dtype=torch.float32)
        for _ in range(int(cycles)):
            self.step(command)

    def chunks(cycles):
        cycles = int(cycles)
        while cycles > 0:
            chunk = min(chunk_size, cycles, MAX_REPEAT)
            yield chunk
            cycles -= chunk

    def clamp_towards(self, q28_cmd):
        limit = knobs.max_joint_delta_per_rpc
        if limit <= 0:
            return q28_cmd
        current = q_now(self)
        clamped = q28_cmd.copy()
        clamped[6:24] = current[6:24] + np.clip(q28_cmd[6:24] - current[6:24], -limit, limit)
        return clamped

    def as_q28(target):
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (28,) or not np.isfinite(target).all():
            raise ValueError("Joint targets must be 28 finite values")
        return target

    def settle_robot(self):
        hold(self, q_now(self), knobs.settle_cycles)

    def move_to_joint_positions(self, target_joint_positions, max_steps=20, settle_steps=10):
        target = as_q28(target_joint_positions)
        settle_period = max(1, int(settle_steps))
        since_settle = 0
        for chunk in chunks(max_steps):
            command = q_now(self)
            command[6:28] = target[6:28]
            hold(self, clamp_towards(self, command), chunk)
            since_settle += chunk
            if since_settle >= settle_period:
                settle_robot(self)
                since_settle = 0
            if np.allclose(q_now(self)[6:28], target[6:28], atol=shim.SERVO_TOL):
                return True
        return bool(np.allclose(q_now(self)[6:28], target[6:28], atol=shim.SERVO_TOL))

    def execute_motion_plan(self, q_traj):
        if isinstance(q_traj, torch.Tensor):
            q_traj = q_traj.detach().cpu().numpy()
        trajectory = np.asarray(q_traj, dtype=np.float64)
        if trajectory.ndim == 1:
            trajectory = trajectory[None, :]
        for row in trajectory:
            command = q_now(self)
            command[6:28] = as_q28(row)[6:28]
            hold(self, clamp_towards(self, command), min(chunk_size, MAX_REPEAT))

    def open_close_gripper(self, arm=0, max_steps=100, open=True):
        names = (("left_gripper_finger_joint1", "left_gripper_finger_joint2") if arm == 0
                 else ("right_gripper_finger_joint1", "right_gripper_finger_joint2"))
        indices = [shim._JOINT_Q28_INDEX[name] for name in names]
        targets = [kinematics.joint_limits(name)[1 if open else 0] for name in names]
        for chunk in chunks(max_steps):
            command = q_now(self)
            for index, value in zip(indices, targets):
                command[index] = value
            hold(self, command, chunk)
            q = q_now(self)
            if all(abs(q[i] - v) < shim.SERVO_TOL for i, v in zip(indices, targets)):
                break

    def update_torso(self, torso_joint_name, angle, max_steps=50):
        index = shim._TORSO_Q28_INDEX[torso_joint_name]
        lower, upper = kinematics.joint_limits(torso_joint_name)
        value = float(angle.item()) if isinstance(angle, torch.Tensor) else float(angle)
        target = q_now(self)
        target[index] = float(np.clip(value, lower, upper))
        return move_to_joint_positions(self, target, max_steps=max_steps, settle_steps=max_steps + 1)

    low_level_class._settle_robot = settle_robot
    low_level_class._move_to_joint_positions = move_to_joint_positions
    low_level_class._execute_motion_plan = execute_motion_plan
    low_level_class._open_close_gripper = open_close_gripper
    low_level_class.update_torso = update_torso


def install(low_level_class):
    """Install all gateway-parity overrides; returns the map-free navigator."""
    from capx.submission.shims.r1pro_kinematics import R1ProKinematics
    knobs = gateway_knobs()
    kinematics = R1ProKinematics()
    install_scoring(low_level_class)
    install_motion(low_level_class, knobs, kinematics)
    install_grasp_check(low_level_class, knobs)

    def navigate(self, pose_2d):
        return navigate_without_map(self, pose_2d, knobs, kinematics)

    return navigate
