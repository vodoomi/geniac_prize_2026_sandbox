"""Collection-side camera, navigation and grasp fixes for the public R1Pro API."""
from __future__ import annotations

import numpy as np


class NavigationStalledError(RuntimeError):
    """Repeated navigation failed without changing the robot pose."""


def install_base_planner(generator_class, base_embodiment):
    original = generator_class.plan_batch

    def plan_batch(self, start_state, goal_pose, plan_config, link_poses=None, emb_sel="default"):
        if emb_sel == base_embodiment:
            plan_config = plan_config.clone()
            # Upstream explicitly disables geometric graph search. A single
            # trajectory seed then often cannot route around furniture.
            # cuRobo's installed batch planner supports graph search with one
            # graph/trajectory seed per query.
            plan_config.enable_graph = True
            plan_config.enable_graph_attempt = 1
            plan_config.num_graph_seeds = 1
            plan_config.num_trajopt_seeds = 1
            plan_config.max_attempts = min(plan_config.max_attempts, 5)
            plan_config.timeout = min(plan_config.timeout, 10.0)
        result, success, paths = original(self, start_state, goal_pose, plan_config,
                                          link_poses=link_poses, emb_sel=emb_sel)
        if emb_sel == base_embodiment and not bool(success.any()):
            print(f"[base-planner] no path: {result.status}", flush=True)
        return result, success, paths

    generator_class.plan_batch = plan_batch


def get_camera_intrinsics(self) -> np.ndarray:
    """Return the ego camera calibration used by RGB/depth perception.

    The wrist cameras are independent streams; querying their lazy annotators
    can fail even while the ego camera is valid. Never substitute calibration
    from another camera or invent a matrix when the ego stream is invalid.
    """
    cameras = [sensor for name, sensor in self._env.robot.sensors.items()
               if name.endswith(":zed_link:Camera:0")]
    if len(cameras) != 1:
        raise RuntimeError(f"Expected one ego camera, found {len(cameras)}")
    import omnigibson as og

    for attempt in range(3):
        try:
            matrix = cameras[0].intrinsic_matrix.detach().cpu().numpy()
        except AssertionError as error:
            if "is degenerate" not in str(error):
                raise
        else:
            if (matrix.shape == (3, 3) and np.isfinite(matrix).all()
                    and matrix[0, 0] > 0 and matrix[1, 1] > 0
                    and np.allclose(matrix[2], [0, 0, 1])):
                return matrix
        if attempt < 2:
            # Refresh rendering only; do not change the physics state.
            for _ in range(4):
                og.sim.render()
    raise RuntimeError("Ego camera calibration is invalid after 3 render refresh attempts")


def install_navigation_guard(api_class):
    original = api_class.navigate_to_pose

    def navigate_to_pose(self, pose_2d) -> bool:
        """Navigate to world (x, y, yaw); False means the goal was not reached.

        The base drives in a straight line toward the goal without a map or
        obstacle avoidance; furniture in the way blocks it. Choose goals with a
        clear path, or go via intermediate waypoints. A blocked base stops after
        about 50 control cycles without progress and the call returns False.
        To move toward a perceived object, prefer approach_object(object_name).
        On failure choose a different approach position. Partial movement is
        possible but not guaranteed. Repeated calls without progress raise
        NavigationStalledError so the current program can be regenerated.
        A None goal (no usable perceived geometry) returns False without motion.
        Before moving, a bent torso is straightened and an outstretched hand is
        pulled in; before a large turn the base first moves clear of furniture
        seen nearby (an in-place turn then happens at that clear spot).
        """
        if pose_2d is None:
            return False
        goal = np.asarray(pose_2d, dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("Navigation goal must be three finite values: x, y, yaw")
        epoch = getattr(self._env, "_exp001_reset_id", 0)
        if getattr(self, "_navigation_epoch", None) != epoch:
            self._navigation_epoch = epoch
            self._navigation_stalls = 0
        from r1pro_skills.navigation import prepare_base_motion
        position, _, _ = self.get_robot_position()
        in_place = np.linalg.norm(goal[:2] - np.asarray(position[:2], dtype=float)) < 0.05
        prepare_base_motion(self, float(goal[2]))
        position, _, yaw = self.get_robot_position()
        if in_place:
            goal = np.array([position[0], position[1], goal[2]], dtype=float)
        before = np.array([*position[:2], float(yaw)])
        success = bool(original(self, goal))
        position, _, yaw = self.get_robot_position()
        after = np.array([*position[:2], float(yaw)])
        turn = abs((after[2] - before[2] + np.pi) % (2 * np.pi) - np.pi)
        if success or np.linalg.norm(after[:2] - before[:2]) > 0.02 or turn > 0.08:
            self._navigation_stalls = 0
        else:
            self._navigation_stalls += 1
            if self._navigation_stalls >= 3:
                raise NavigationStalledError(
                    "Navigation made no progress in 3 consecutive calls. "
                    "Stop repeating this approach; regenerate the program with "
                    "a different waypoint or observe the scene again."
                )
        return success

    api_class.navigate_to_pose = navigate_to_pose


def _move_hand_with_ik(self, target, orientation, arm, label):
    from r1pro_skills.ik import IKSolverTimeout
    try:
        joints = self.solve_ik(target, orientation[[3, 0, 1, 2]], arm=arm)
    except IKSolverTimeout as error:
        print(f"[grasp] {label}: {error}; no motion executed", flush=True)
        return False
    residual = self.last_ik_pos_residual
    if residual is None or not np.isfinite(residual) or residual > self._ik_residual_limit:
        print(f"[grasp] {label}: unreachable IK target (residual={residual})")
        return False
    if not np.isfinite(joints).all():
        raise RuntimeError("IK returned non-finite joints")
    self.move_to_joint_positions(joints, max_steps=200, settle_steps=5)
    actual, _ = self.get_current_eef_pose(arm=arm)
    error = float(np.linalg.norm(actual - target))
    if not np.isfinite(actual).all() or not np.isfinite(error) or error > 0.04:
        print(f"[grasp] {label}: motion stopped {error:.3f} m from target")
        return False
    return True


def lift_arm(self, arm=0) -> bool:
    """Raise the hand 15 cm in world Z, preserving its current orientation.

    Returns True when the measured hand rose at least 10 cm (a held object
    makes the arm sag a few cm short of the target). False means IK or motion
    failed; do not assume the object was lifted.
    The torso is not automatically reset after the lift.
    """
    if arm not in (0, 1):
        raise ValueError("arm must be 0 (left) or 1 (right)")
    position, quat = self.get_current_eef_pose(arm=arm)
    position = np.asarray(position, dtype=float)
    if _move_hand_with_ik(self, position + np.array([0, 0, LIFT_HEIGHT]), quat, arm, "lift"):
        return True
    # v10 radio 0: the radio was held and raised, the hand stopped 4.2 cm short
    # of the target, and the grasp was dropped as a failure.
    after, _ = self.get_current_eef_pose(arm=arm)
    after = np.asarray(after, dtype=float)
    rise, drift = after[2] - position[2], float(np.linalg.norm(after[:2] - position[:2]))
    if np.isfinite(after).all() and rise >= LIFT_MIN_RISE and drift <= 0.05:
        print(f"[grasp] lift: hand rose {rise:.3f} m; accepting the lift", flush=True)
        return True
    return False


LIFT_HEIGHT = 0.15
LIFT_MIN_RISE = 0.10


def move_hand(self, target_pose, arm=0) -> bool:
    """Move a hand to world (XYZ, XYZW), accepting lists or numpy arrays.

    Returns whether planning/execution succeeded. This matches the public
    API's existing obstacle policy; choose a reachable pose on False.
    """
    import torch
    if arm not in (0, 1):
        raise ValueError("arm must be 0 (left) or 1 (right)")
    position, quat = (np.asarray(value, dtype=float) for value in target_pose)
    if (position.shape != (3,) or quat.shape != (4,) or not np.isfinite(position).all()
            or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-6):
        raise ValueError("Hand pose must contain finite XYZ and nonzero XYZW")
    quat = quat / np.linalg.norm(quat)
    result = self._env._move_hand((torch.from_numpy(position), torch.from_numpy(quat)),
                                  arm=arm, ignore_all_obstacles=True, skip_obstacle_update=True)
    self._env._settle_robot()
    return bool(result)


def grasp_object(self, pregrasp_pose, grasp_pose, object_name: str, arm=0) -> bool:
    """Approach an observed grasp via its pregrasp, close, and verify holding.

    Poses contain world position and XYZW quaternion. Returns False if out of
    reach, an IK/motion stage fails, or the gripper remains empty. Reobserve
    and sample a different grasp on failure. True means an object is held
    after a verified 15 cm upward hand motion; use perception to verify its
    identity. The lifted posture is preserved; reset the torso explicitly
    only when the subsequent task requires it.

    Also accepts the TWO PAIRED CANDIDATE LISTS returned by sample_grasp_pose:
        pres, grasps = sample_grasp_pose("blue can of soda")
        held = grasp_object(pres, grasps, "blue can of soda", arm=0)
    Candidates are tried in order until verified holding. Each individual pose
    is (XYZ position, XYZW quaternion); the candidate lists are not XYZ arrays.
    """
    if arm not in (0, 1):
        raise ValueError("arm must be 0 (left) or 1 (right)")
    if pregrasp_pose is None or grasp_pose is None:
        return False

    def is_pose(value):
        try:
            return (len(value) == 2 and np.asarray(value[0]).shape == (3,)
                    and np.asarray(value[1]).shape == (4,))
        except (TypeError, ValueError):
            return False

    if not (is_pose(pregrasp_pose) and is_pose(grasp_pose)):
        # Preserve correspondence; never flatten mixed position/quaternion
        # arrays or guess which candidate the caller intended.
        if (len(pregrasp_pose) != len(grasp_pose)
                or not all(is_pose(pre) and is_pose(grasp)
                           for pre, grasp in zip(pregrasp_pose, grasp_pose))):
            raise ValueError("Expected paired (XYZ, XYZW) poses or equally sized candidate lists")
        for pre, grasp in zip(pregrasp_pose, grasp_pose):
            if grasp_object(self, pre, grasp, object_name, arm=arm):
                return True
        return False

    def as_pose(pose):
        position, quat = (np.asarray(value, dtype=float) for value in pose)
        if (position.shape != (3,) or quat.shape != (4,)
                or not np.isfinite(position).all() or not np.isfinite(quat).all()
                or np.linalg.norm(quat) < 1e-6):
            raise ValueError("Grasp pose needs finite XYZ and a nonzero XYZW quaternion")
        return position.copy(), quat / np.linalg.norm(quat)

    preposition, prequat = as_pose(pregrasp_pose)
    position, quat = as_pose(grasp_pose)
    target = position + np.array([0, 0, -0.02])
    steps = max(1, int(np.ceil(np.linalg.norm(target - preposition) / 0.08)))
    if steps > 16:
        raise ValueError("Pregrasp is too far from the grasp; resample the grasp poses")
    robot_pos, _, _ = self.get_robot_position()
    if np.linalg.norm(position[:2] - robot_pos[:2]) > self._grasp_reach_limit:
        print(f"[grasp] {object_name}: outside arm reach; navigate closer")
        return False

    def move(target, orientation):
        return _move_hand_with_ik(self, target, orientation, arm, object_name)

    self.open_gripper(arm=arm)
    print(f"[grasp] {object_name}: arm={arm}, approaching pregrasp {preposition}", flush=True)
    if not move(preposition, prequat):
        return False
    # Preserve the existing 2 cm grasp insertion, but approach from above
    # instead of sending the full-body IK directly from rest to the floor.
    for fraction in np.linspace(0, 1, steps + 1)[1:]:
        if not move(preposition + fraction * (target - preposition), quat):
            return False
    print(f"[grasp] {object_name}: closing at grasp target", flush=True)
    self.close_gripper(arm=arm)
    if not self.check_object_in_hand(arm=arm):
        print(f"[grasp] {object_name}: gripper is empty; resample before retrying")
        return False
    # The legacy lift_arm rotates one shoulder joint by a fixed angle. It
    # can lower the hand when the torso is bent. Lift in world Z instead and
    # verify the measured hand pose before reporting a successful pickup.
    if not self.lift_arm(arm=arm):
        return False
    return bool(self.check_object_in_hand(arm=arm))


def install(api_class, low_level_class):
    api_class.get_camera_intrinsics = get_camera_intrinsics
    api_class.grasp_object = grasp_object
    api_class.lift_arm = lift_arm
    api_class.move_hand = move_hand
    install_navigation_guard(api_class)
    original_reset = low_level_class.reset

    def reset(self, *args, **kwargs):
        from behavior_collect.supervisor import report_phase
        report_phase("reset")
        self._exp001_reset_id = getattr(self, "_exp001_reset_id", 0) + 1
        self._exp001_navigation_stalls = 0
        result = original_reset(self, *args, **kwargs)
        report_phase("running")
        return result

    low_level_class.reset = reset
    # Search helpers call the low-level method directly, bypassing the public
    # navigate_to_pose wrapper. Bound these failures as well; allow all four
    # fractional waypoints per public call before counting three whole calls.
    original_navigation = low_level_class._navigate_to_pose

    def low_level_navigation(self, pose_2d):
        before, before_quat = self.robot.get_position_orientation()
        before = before.clone()
        before_quat = before_quat.clone()
        success = original_navigation(self, pose_2d)
        after, after_quat = self.robot.get_position_orientation()
        moved = float((after[:2] - before[:2]).norm()) > 0.02
        turned = min(float((after_quat - before_quat).norm()),
                     float((after_quat + before_quat).norm())) > 0.04
        if success or moved or turned:
            self._exp001_navigation_stalls = 0
        else:
            self._exp001_navigation_stalls = getattr(self, "_exp001_navigation_stalls", 0) + 1
            if self._exp001_navigation_stalls >= 12:
                raise NavigationStalledError(
                    "Navigation/search made no progress in 12 planning calls; "
                    "regenerate with a different approach instead of repeating the search."
                )
        return success

    low_level_class._navigate_to_pose = low_level_navigation
