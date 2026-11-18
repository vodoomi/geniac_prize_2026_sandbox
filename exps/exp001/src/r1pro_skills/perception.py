"""Perception contracts: absent objects are normal results, invalid services are errors."""
from __future__ import annotations

import numpy as np


def valid_object_mask(results, depth, threshold):
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth, got {depth.shape}")
    if not results:
        return None
    scores = np.asarray([r["score"] for r in results], dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("SAM3 returned non-finite confidence")
    best = results[int(scores.argmax())]
    if scores.max() < threshold:
        return None
    mask = np.asarray(best["mask"], dtype=bool)
    if mask.shape != depth.shape:
        raise ValueError("SAM3 mask and depth resolution do not match")
    mask = mask & np.isfinite(depth) & (depth > 0)
    return mask if int(mask.sum()) >= 6 else None


def candidate_masks(results, depth, threshold):
    """[(score, mask)] of every SAM3 detection above the threshold, best first."""
    depth = np.asarray(depth)
    out = []
    for r in sorted(results or [], key=lambda r: -float(r["score"])):
        score = float(r["score"])
        if not np.isfinite(score):
            raise ValueError("SAM3 returned non-finite confidence")
        if score < threshold:
            break
        mask = np.asarray(r["mask"], dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError("SAM3 mask and depth resolution do not match")
        mask = mask & np.isfinite(depth) & (depth > 0)
        if int(mask.sum()) >= 6:
            out.append((score, mask))
    return out


# Max distance (m) from the expected position for a detection to count as the tracked object.
NEAR_EXPECTED = 0.35


def _select_near(self, results, depth, intrinsics, camera, expected):
    """Mask of the above-threshold detection nearest `expected` (within NEAR_EXPECTED), or None."""
    from capx.integrations.r1pro import control as c
    best = None
    for score, mask in candidate_masks(results, depth, self.sam3_score_threshold):
        points = np.asarray(c.backproject_depth(mask, depth, intrinsics, camera))
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 6:
            continue
        distance = float(np.linalg.norm(np.median(points, axis=0) - expected))
        if distance <= NEAR_EXPECTED and (best is None or distance < best[0]):
            best = (distance, score, mask)
    if best is None:
        return None, None
    return best[2], best[1]


def erode_mask(mask, min_pixels=50):
    """Drop the mask's 1-pixel rim: its depth often mixes object and background."""
    import cv2
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return eroded if int(eroded.sum()) >= min_pixels else mask


def drop_stray_points(points):
    """Remove the trail of background points behind/in front of an object.

    Mixed-depth pixels along the silhouette back-project along camera rays far
    from the object (radio instance 0: a 1.5 m trail to the floor that moved the
    OBB centre 0.6 m off the radio). Statistical outlier removal keeps such a
    trail because it is locally dense, so keep points near the median instead.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 20:
        return points
    distance = np.linalg.norm(points - np.median(points, axis=0), axis=1)
    radius = max(0.05, 2.0 * float(np.percentile(distance, 75)))
    return points[distance <= radius]


def observe_object(self, object_name):
    from capx.integrations.r1pro import control as c
    obs = self._env.get_observation()
    rgb, depth = c.obs_get_rgb_depth(obs)
    depth = depth.cpu().numpy()
    rgb = rgb.cpu().numpy()[..., :3]
    results = self.sam3_seg_fn(rgb, text_prompt=object_name)
    intrinsics = self.get_camera_intrinsics()
    camera = self.get_camera_pose()
    expected = getattr(self, "_exp001_expected", None)
    mask, score = None, None
    if expected is not None:
        # Several objects can match the prompt: prefer the one being tracked
        # even when another detection has a higher confidence.
        mask, score = _select_near(self, results, depth, intrinsics, camera, np.asarray(expected, dtype=float))
    if mask is None:
        mask = valid_object_mask(results, depth, self.sam3_score_threshold)
        score = None if mask is None else float(max(r["score"] for r in results))
    # Confidence of the detection behind the last returned pose (used to tell
    # the named object from other objects that also match the prompt).
    self._exp001_last_score = score
    self._exp001_last_view = (rgb, depth, results)
    if mask is None:
        return None
    mask = erode_mask(mask)
    # backproject_depth uses selected pixels; do not feed it invalid depth.
    points = c.backproject_depth(mask, depth, intrinsics, camera)
    points = np.asarray(points)
    points = points[np.isfinite(points).all(axis=1)]
    points = drop_stray_points(points)
    if len(points) < 6 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 3:
        return None
    cloud = c.o3d.geometry.PointCloud()
    cloud.points = c.o3d.utility.Vector3dVector(points)
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=min(20, len(points) - 1), std_ratio=2.0)
    points = np.asarray(cloud.points)
    if len(points) < 6 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 3:
        return None
    obb = cloud.get_oriented_bounding_box()
    if not np.isfinite(obb.center).all() or not np.isfinite(obb.R).all():
        raise RuntimeError("Point-cloud geometry returned a non-finite pose")
    return depth, mask, intrinsics, camera, points, obb


def get_object_pose(self, object_name: str, return_bbox_extent: bool = False) -> tuple:
    """Return FIVE values: position, WXYZ quaternion, extent, points, OBB.

    If no object has enough valid depth, return (None, None, None, None, None).
    Reobserve from another camera pose on that result. Service/program errors
    propagate and must not be interpreted as an absent object.
    """
    from scipy.spatial.transform import Rotation
    observation = observe_object(self, object_name)
    if observation is None:
        return None, None, None, None, None
    _, _, _, _, points, obb = observation
    quaternion = Rotation.from_matrix(obb.R).as_quat()[[3, 0, 1, 2]]
    return np.asarray(obb.center), quaternion, np.asarray(obb.extent) if return_bbox_extent else None, points, obb


def sample_grasp_pose(self, object_name: str) -> tuple:
    """Return paired world (position, XYZW quaternion) lists, or (None, None).

    An absent object or insufficient valid depth is a normal failure. Empty
    GraspNet candidates fall back to the geometric grasp from the observed OBB.
    """
    from capx.integrations.r1pro import control as c
    observation = observe_object(self, object_name)
    if observation is None:
        return None, None
    depth, mask, intrinsics, camera, _, obb = observation
    simple_pre, simple_grasp = self.sample_grasp_pose_simple(object_name, obb)
    safe_depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0)
    # GraspNet's default depth window is 0.2-2.0 m. A detected object
    # outside it leaves an empty local region and triggers mean-of-empty
    # warnings/repeated inference. The geometric fallback remains valid.
    if np.count_nonzero(mask & (safe_depth >= .2) & (safe_depth <= 2.0)) < 6:
        return [simple_pre], [simple_grasp]
    samples, scores, _ = self.grasp_net_plan_fn(safe_depth, intrinsics, mask.astype(np.uint8), 1)
    if not len(samples):
        return [simple_pre], [simple_grasp]
    samples, scores = np.asarray(samples), np.asarray(scores)
    if samples.shape != (len(scores), 4, 4) or not np.isfinite(samples).all() or not np.isfinite(scores).all():
        raise ValueError("GraspNet returned invalid grasp matrices or scores")
    grasp_camera = c.vtf.SE3.from_matrix(samples[int(scores.argmax())]) @ c.vtf.SE3.from_translation(np.array([0, 0, 0.05]))
    grasp_world = c.vtf.SE3.from_matrix(camera) @ c.vtf.SE3.from_matrix(c.convert_T_cam_cv_to_cam_gl(grasp_camera.as_matrix()))
    position = grasp_world.wxyz_xyz[-3:].copy()
    # Preserve the upstream candidate's top-down orientation and approach.
    from scipy.spatial.transform import Rotation
    quat = grasp_world.wxyz_xyz[:4][[1, 2, 3, 0]]
    approach = Rotation.from_quat(quat).apply([0, 0, -1.])
    topdown = np.array([1., 0, 0, 0])
    return [simple_pre, (position - approach * 0.1, topdown)], [simple_grasp, (position, topdown)]


def get_navigation_pose(self, P_table, P_object) -> tuple[float, float, float] | None:
    """Return an approach (x,y,yaw) from perceived point clouds, or None.

    Missing/empty/collinear geometry cannot define a table edge. Reobserve on
    None; navigate_to_pose(None) returns False and sends no motion command.
    """
    if P_table is None or P_object is None:
        return None
    table, obj = np.asarray(P_table, dtype=float), np.asarray(P_object, dtype=float)
    if table.size == 0 or obj.size == 0:
        return None
    if table.ndim != 2 or obj.ndim != 2 or table.shape[1] != 3 or obj.shape[1] != 3:
        raise ValueError("Navigation requires Nx3 world point clouds")
    table, obj = table[np.isfinite(table).all(axis=1)], obj[np.isfinite(obj).all(axis=1)]
    if len(table) < 3 or not len(obj) or np.linalg.matrix_rank(table[:, :2] - table[:, :2].mean(axis=0)) < 2:
        return None
    from scipy.spatial import QhullError
    from capx.integrations.r1pro import control as c
    try:
        goal = c.get_navigation_pose(table, obj)
    except QhullError:
        return None
    if not np.isfinite(goal).all():
        return None
    return tuple(float(value) for value in goal)


def install(api_class):
    original_pose = api_class.get_object_pose
    original_sample = api_class.sample_grasp_pose

    # get_api_prompt inspects class methods, so use class wrappers with accurate
    # return contracts as well as the instance bindings used by generated code.
    def pose(self, object_name: str, return_bbox_extent: bool = False) -> tuple:
        return get_object_pose(self, object_name, return_bbox_extent) if self.use_sam3 else original_pose(self, object_name, return_bbox_extent)

    def sample(self, object_name: str) -> tuple:
        return sample_grasp_pose(self, object_name) if self.use_sam3 else original_sample(self, object_name)

    pose.__doc__ = get_object_pose.__doc__
    pose.__annotations__["return"] = tuple[np.ndarray | None, np.ndarray | None,
                                            np.ndarray | None, np.ndarray | None, object | None]
    sample.__doc__ = sample_grasp_pose.__doc__
    sample.__annotations__["return"] = tuple[list[tuple[np.ndarray, np.ndarray]] | None,
                                              list[tuple[np.ndarray, np.ndarray]] | None]
    api_class.get_object_pose = pose
    api_class.sample_grasp_pose = sample
    api_class.get_navigation_pose = get_navigation_pose
    def get_sam3_mask(self, object_name: str) -> int:
        """Return detected foreground pixel count, or zero when not detected."""
        from capx.integrations.r1pro import control as c
        rgb, depth = c.obs_get_rgb_depth(self._env.get_observation())
        mask = valid_object_mask(self.sam3_seg_fn(rgb.cpu().numpy()[..., :3], text_prompt=object_name),
                                 depth.cpu().numpy(), self.sam3_score_threshold)
        return int(mask.sum()) if mask is not None else 0
    api_class.get_sam3_mask = get_sam3_mask
