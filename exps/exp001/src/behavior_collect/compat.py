"""Adapt CaP-X's older control calls to the pinned OmniGibson 3.9.2 API.

Installed only by the exp001 bootstrap, after Kit starts. Physics, observations,
task rewards, and trial handling remain those of the competition runtime.
"""
import math
import os
from pathlib import Path
from xml.etree import ElementTree


def prepare_ik_urdf():
    """Keep competition IK kinematics, resolving its meshes against OG assets."""
    exp = Path(__file__).resolve().parents[2]
    capx = exp.parents[1] / "third_party/geniac_prize_2026_baseline/geniac_prize_2026_code"
    source = capx / "capx/third_party/b1k/assets/r1pro_ik.urdf"
    data = Path(os.environ.get("OMNIGIBSON_DATA_PATH", exp / "runtime/data"))
    asset_urdf_dir = data / "omnigibson-robot-assets/models/r1pro/urdf"
    tree = ElementTree.parse(source)
    for mesh in tree.iter("mesh"):
        filename = (asset_urdf_dir / mesh.attrib["filename"]).resolve()
        if not filename.is_file():
            raise FileNotFoundError(f"R1Pro IK collision mesh missing: {filename}")
        mesh.set("filename", str(filename))
    target = exp / "runtime/r1pro_ik.urdf"
    contents = ElementTree.tostring(tree.getroot(), encoding="unicode")
    if not target.exists() or target.read_text() != contents:
        target.write_text(contents)
    return target


RELAXED_TORSO_YAW_WEIGHT = 5.0   # rest-cost weight of torso_joint4 in the relaxed IK retry
RELAX_IK_ABOVE = 0.01            # m; strict IK residual above which the relaxed retry runs


def install(low_level_class):
    import torch
    import omnigibson.lazy as lazy
    from omnigibson.action_primitives import starter_semantic_action_primitives as sap
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection as Embodiment
    from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
    from capx.envs.simulators import r1pro_b1k
    from capx.integrations.r1pro.control import R1ProControlApi

    if getattr(low_level_class, "_exp001_compat", False):
        return

    from capx.integrations.r1pro import control
    context_loader = control.get_pyroki_context
    ik_urdf = prepare_ik_urdf()

    def get_pyroki_context(robot_urdf_or_name="panda_description", **kwargs):
        if Path(robot_urdf_or_name).name == "r1pro_ik.urdf":
            robot_urdf_or_name = str(ik_urdf)
        return context_loader(robot_urdf_or_name, **kwargs)

    control.get_pyroki_context = get_pyroki_context
    from r1pro_skills.ik import BoundedIK
    bounded_ik = BoundedIK(ik_urdf)
    original_api_init = R1ProControlApi.__init__

    def initialize_api(self, *args, **kwargs):
        original_api_init(self, *args, **kwargs)
        self._pks = bounded_ik
        # The relaxed IK fallback (all rest weights 1.0) let hand moves twist the
        # torso yaw (torso_joint4) by up to 1.6 rad, swinging both arms through
        # the radio and the table (BEHAVIOR.md, v10). With a
        # yaw weight of 5 the same targets solve to ~3 mm with <= 0.1 rad of
        # twist (0.3 rad for an object 0.2 rad to the right). The bends that
        # reach low objects (torso joints 1-3) stay relaxed.
        yaw = self._robot.joints.actuated_names.index("torso_joint4")
        self._rest_cost_weights_relaxed = self._rest_cost_weights_relaxed.at[yaw].set(RELAXED_TORSO_YAW_WEIGHT)

    R1ProControlApi.__init__ = initialize_api
    original_solve_ik = R1ProControlApi.solve_ik

    def solve_ik(self, *args, **kwargs):
        # Upstream retries with the relaxed weights only when the strict solve
        # misses by more than _ik_residual_limit (5 cm). Strict solves 0.5 m
        # ahead at table height miss by 3.6-5 cm, were accepted, and the hand
        # then stopped ~4 cm short of every pregrasp (v10 radio 0/9). Retry
        # (keeping the better solution, as upstream does) above 1 cm instead.
        limit = self._ik_residual_limit
        self._ik_residual_limit = min(limit, RELAX_IK_ABOVE)
        try:
            return original_solve_ik(self, *args, **kwargs)
        finally:
            self._ik_residual_limit = limit

    solve_ik.__doc__ = original_solve_ik.__doc__
    R1ProControlApi.solve_ik = solve_ik

    class CompatiblePrimitives(sap.StarterSemanticActionPrimitives):
        @property
        def arm(self):
            return getattr(self, "_selected_arm", self.robot.default_arm)

        def overwrite_arm(self, arm):
            if isinstance(arm, int):
                if arm < 0 or arm >= len(self.robot.arm_names):
                    raise ValueError(f"Invalid arm index: {arm}")
                arm = self.robot.arm_names[arm]
            if arm not in self.robot.arm_names:
                raise ValueError(f"Invalid arm name: {arm}")
            self._selected_arm = arm

        def _move_hand(
            self, target_pose, stop_on_contact=False, stop_on_ag=False,
            motion_constraint=None, low_precision=False, lock_auxiliary_arm=False,
            ignore_objects=None, attached_obj_scale=None, ignore_all_obstacles=False,
            ik_only=False, ik_world_collision_check=True, skip_obstacle_update=False,
        ):
            reference = self.robot.get_joint_positions()
            pos, quat = (torch.as_tensor(v, dtype=reference.dtype, device=reference.device)
                         for v in target_pose)
            yield from self._settle_robot()
            targets = {self.robot.eef_link_names[self.arm]: (pos, quat)}
            if lock_auxiliary_arm:
                for arm in self.robot.arm_names:
                    if arm != self.arm:
                        targets[self.robot.eef_link_names[arm]] = self.robot.get_eef_pose(arm=arm)
            generator = self._motion_generator
            planner = generator.mg[Embodiment.ARM]
            # Honor the existing public move_hand API's explicit obstacle bypass;
            # navigation and ordinary planned moves still use scene collisions.
            if ignore_all_obstacles:
                planner.update_world(lazy.curobo.geom.types.WorldConfig())
            elif not skip_obstacle_update:
                generator.update_obstacles(ignore_objects=ignore_objects)
            try:
                # The held object comes from assisted-grasp internals that a
                # gateway submission cannot read; plan without attaching it.
                attached = None
                successes, paths = generator.compute_trajectories(
                    target_pos={k: v[0].unsqueeze(0).repeat(generator.batch_size, 1) for k, v in targets.items()},
                    target_quat={k: v[1].unsqueeze(0).repeat(generator.batch_size, 1) for k, v in targets.items()},
                    is_local=False,
                    max_attempts=math.ceil(sap.m.MAX_PLANNING_ATTEMPTS / generator.batch_size),
                    timeout=60.0, ik_fail_return=sap.m.MAX_IK_FAILURES_BEFORE_RETURN,
                    enable_finetune_trajopt=True, finetune_attempts=1,
                    success_ratio=1.0 / generator.batch_size,
                    attached_obj=attached, attached_obj_scale=attached_obj_scale,
                    motion_constraint=motion_constraint, skip_obstacle_update=True,
                    ik_only=ik_only, ik_world_collision_check=ik_world_collision_check,
                    emb_sel=Embodiment.ARM,
                )
                indices = torch.where(successes)[0].cpu()
                if not len(indices):
                    raise ActionPrimitiveError(
                        ActionPrimitiveError.Reason.PLANNING_ERROR,
                        "No accessible arm motion to the requested pose",
                    )
                trajectory = generator.path_to_joint_trajectory(
                    paths[int(indices[0])], get_full_js=True, emb_sel=Embodiment.ARM,
                ).cpu().float().reshape(-1, reference.numel())
                if ik_only:
                    trajectory = torch.cat([reference.cpu().unsqueeze(0), trajectory])
                if len(trajectory) > 1:
                    trajectory = generator.add_linearly_interpolated_waypoints(trajectory, max_inter_dist=0.01)
                yield from self._execute_motion_plan(
                    trajectory, stop_on_contact=stop_on_contact,
                    stop_on_ag=stop_on_ag, low_precision=low_precision,
                )
            finally:
                if ignore_all_obstacles:
                    generator.update_obstacles()

        def _move_hand_direct_ik(self, target_pose, **kwargs):
            # The configured robot uses JointControllers, whereas 3.9's direct
            # IK helper requires InverseKinematicsControllers. Solve using the
            # same ARM kinematics and execute the resulting joint targets.
            yield from self._move_hand(target_pose, ik_only=True, **kwargs)

    r1pro_b1k.StarterSemanticActionPrimitives = CompatiblePrimitives

    def execute_hand(self, target_pose, arm=0, **kwargs):
        self.controller.overwrite_arm(arm)
        try:
            for action in self.controller._move_hand(target_pose, **kwargs):
                self.step(action)
        except ActionPrimitiveError as error:
            print(f"Hand target is unreachable: {error}", flush=True)
            return False
        return True

    def execute_hand_direct(self, target_pose, arm=0):
        return execute_hand(self, target_pose, arm=arm, ik_only=True)

    low_level_class._move_hand = execute_hand
    low_level_class._move_hand_direct_ik = execute_hand_direct

    def cast_joints(method):
        def wrapped(self, targets, *args, **kwargs):
            reference = self.robot.get_joint_positions()
            targets = torch.as_tensor(targets, dtype=reference.dtype, device=reference.device)
            if targets.shape != reference.shape or not bool(torch.isfinite(targets).all()):
                raise ValueError("Joint targets must be finite and match the robot joint count")
            return method(self, targets, *args, **kwargs)
        return wrapped

    low_level_class._move_to_joint_positions = cast_joints(low_level_class._move_to_joint_positions)
    low_level_class._execute_motion_plan = cast_joints(low_level_class._execute_motion_plan)

    # The function has always returned FIVE values. Its three-value annotation
    # led generated programs to unpack only three and fail during perception.
    R1ProControlApi.get_object_pose.__annotations__["return"] = tuple[
        object, object, object, object, object
    ]
    R1ProControlApi.get_object_pose.__doc__ += (
        "\nAlways unpack all FIVE return values, including when return_bbox_extent=False:\n"
        "position, quaternion_wxyz, bbox_extent, o3d_points, obb = get_object_pose(object_name)\n"
        "Use result[0] if only the position is needed.\n"
    )
    from r1pro_skills.control import install as install_control_fixes
    # Gateway parity: no ground-truth scene map for base motion, proprio-only
    # grasp check, the taskpack step limit and the gateway's success latch.
    # The cuRobo BASE planner (and its graph-search patch) is no longer used.
    from behavior_collect.parity import install as install_parity
    low_level_class._navigate_to_pose = install_parity(low_level_class)
    install_control_fixes(R1ProControlApi, low_level_class)
    from r1pro_skills.perception import install as install_perception
    install_perception(R1ProControlApi)
    from r1pro_skills.navigation import install as install_navigation
    install_navigation(R1ProControlApi)
    from r1pro_skills.grasp import install as install_grasp
    install_grasp(R1ProControlApi)
    from capx.envs.tasks.base import CodeExecutionEnvBase
    from r1pro_skills.budget import install as install_budget
    install_budget(CodeExecutionEnvBase, R1ProControlApi)
    R1ProControlApi.grasp_object.__doc__ += (
        "\nPrefer pick_object(object_name), which re-observes the object with the wrist\n"
        "camera and aligns the gripper before grasping.\n")
    # These APIs return booleans/tuples despite upstream's None/array
    # annotations. combined_doc exposes signatures directly to code generation.
    import numpy as np
    for name in ("check_object_in_hand", "find_object_base_rotate", "find_object_torso_rotate",
                 "move_to_joint_positions", "detect_object_sam3"):
        getattr(R1ProControlApi, name).__annotations__["return"] = bool
    R1ProControlApi.get_robot_position.__annotations__["return"] = tuple[np.ndarray, np.ndarray, np.ndarray]
    R1ProControlApi.get_robot_relative_eef_pose.__annotations__["return"] = tuple[object, object]
    low_level_class._exp001_compat = True
    print("[behavior-compat] arm selection, motion arguments, joint dtypes, ego calibration, "
          "navigation guard and stall stop, approach_object, pick_object, step budget, staged grasps and gateway parity enabled", flush=True)
