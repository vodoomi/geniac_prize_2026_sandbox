"""High-level, perception-driven base approach for generated BEHAVIOR programs.

Qwen repeatedly computed approach goals itself (e.g. passing a single floor
point as ``P_table`` to ``get_navigation_pose``) and sent one long straight-line
move that ran into furniture. ``approach_object`` moves in short legs and
re-observes the object with SAM3 after each leg, so the goal follows what the
camera sees. Before each leg it checks the corridor the base would sweep against
the head camera's depth points and detours around anything in the way. For an
object above the floor it finds the supporting surface in the depth points and
drives to the point outside that surface's edge nearest the object. It uses only
the public perception APIs and the low-level base servo, i.e. information
available at evaluation (no map, no ground truth).
"""
from __future__ import annotations

import math

import numpy as np

REACH_TOLERANCE = 0.1   # m beyond the standoff that still counts as reached
FACE_TOLERANCE = 0.15   # rad; turn in place to face the object above this
SIDESTEP = 0.5          # m; lateral detour when no obstacle extent is visible
SIDESTEP_MARGIN = 0.10  # m beyond the obstacle's side edge plus the corridor half width
MAX_SIDESTEP = 2.0      # m; wider obstacles are left to look-aside / regeneration
SIDESTEP_LOOKAHEAD = 1.2  # m ahead of the base that must be clear after the sidestep
MIN_PROGRESS = 0.1      # m; a leg (or sidestep) moving less is "blocked"

# R1Pro base: x -0.392..0.235 m (front), y +-0.337 m. The corridor is the swept
# width plus a margin; points nearer than SELF_RADIUS are the robot's own body.
CORRIDOR_HALF_WIDTH = 0.40   # base (z <= BASE_TOP): +-0.337 m plus margin
UPPER_HALF_WIDTH = 0.30      # torso and tucked arms above the base
BASE_TOP = 0.45
FRONT_CLEARANCE = 0.30
MIN_PARTIAL = 0.15           # m; shortest partial leg worth driving
LOOK_ASIDE = (60, -60)       # degrees; turn to look for a way when the view is blocked
ELEVATED_REACH = 0.65        # m; an object on furniture this close can be grasped
SELF_RADIUS = 0.42
OBSTACLE_Z = (0.08, 1.60)   # floor and ceiling are not obstacles
DETOUR_HEADINGS = (20, -20, 35, -35)   # degrees; kept inside the camera's view
DETOUR_LENGTHS = (1.0, 0.6)
ELEVATED_Z = 0.25           # object centres above this rest on furniture
EDGE_BUFFER = 0.32          # base centre outside the support edge (front is 0.235)
NEAR_LEG = 2.0              # m; legs get shorter inside this distance
LOST_DETOUR_DEG = 35       # side approach after losing sight of the object
ELEVATED_STANDOFF = 0.55    # m; fallback standoff for furniture without a found top
# A resting object does not move while the robot approaches it. A detection
# farther than this from the tracked position is a different object that SAM3
# also matched (seen in radio instance 0 once the real radio left the view).
JUMP_TOLERANCE = 0.35
SWITCH_MARGIN = 0.2   # SAM3 confidence gain that justifies switching to a far detection


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _robot_xy_yaw(api):
    position, _, yaw = api.get_robot_position()
    return np.asarray(position, dtype=float)[:2], float(np.asarray(yaw))


def _observe(api, object_name):
    """World (position, points) of the object, or (None, None) when not seen."""
    position, _, _, points, _ = api.get_object_pose(object_name)
    if position is None:
        return None, None
    return np.asarray(position, dtype=float), (None if points is None else np.asarray(points, dtype=float))


def _score(api):
    score = getattr(api, "_exp001_last_score", None)
    return 0.0 if score is None else float(score)


def _observe_tracked(api, object_name, expected, track=None):
    """_observe, but a detection far from the tracked position counts as not seen
    unless SAM3 is clearly more confident in it than in the tracked one (then
    the track switches: the first match may have been a different object)."""
    api._exp001_expected = None if expected is None else np.asarray(expected, dtype=float)
    try:
        seen, points = _observe(api, object_name)
    finally:
        api._exp001_expected = None
    score = _score(api)
    if seen is None and expected is not None:
        _save_lost_view(api, object_name, expected)
    if seen is not None and expected is not None:
        jump = float(np.linalg.norm(seen - np.asarray(expected, dtype=float)))
        tracked_score = track["score"] if track is not None else 0.0
        if jump > JUMP_TOLERANCE:
            if score >= tracked_score + SWITCH_MARGIN:
                print(f"[approach] '{object_name}' now matched at {np.round(seen, 2)} with confidence "
                      f"{score:.2f} (tracked {np.round(expected, 2)} had {tracked_score:.2f}); switching", flush=True)
            else:
                print(f"[approach] '{object_name}' detected {jump:.2f} m from where it was tracked "
                      f"({np.round(seen, 2)} conf {score:.2f} vs {np.round(expected, 2)} conf {tracked_score:.2f}); "
                      "ignoring it as a different object", flush=True)
                _save_lost_view(api, object_name, expected)
                return None, None
    if seen is not None and track is not None:
        track["score"] = score
    return seen, points


def _save_lost_view(api, object_name, expected) -> None:
    """Debug only (EXP001_APPROACH_DEBUG_DIR): head image and every SAM3 candidate when the object is lost."""
    import os
    directory = os.environ.get("EXP001_APPROACH_DEBUG_DIR")
    view = getattr(api, "_exp001_last_view", None)
    if not directory or view is None:
        return
    import cv2
    rgb, _, results = view
    os.makedirs(directory, exist_ok=True)
    image = np.asarray(rgb, dtype=np.uint8)[..., ::-1].copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    for rank, r in enumerate(sorted(results or [], key=lambda r: -float(r["score"]))[:5]):
        mask = np.asarray(r["mask"], dtype=bool)
        image[mask] = (0.5 * image[mask] + 0.5 * np.array(colors[rank])).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if len(xs):
            cv2.putText(image, f"{float(r['score']):.2f}", (int(xs.mean()), int(ys.mean())), 0, 0.5,
                        (255, 255, 255), 1)
    index = len(os.listdir(directory))
    path = os.path.join(directory, f"lost_{index:02d}.png")
    cv2.imwrite(path, image)
    xy, yaw = _robot_xy_yaw(api)
    print(f"[approach] lost '{object_name}' (expected {np.round(expected, 2)}, base {np.round(xy, 2)} "
          f"yaw {yaw:.2f}); view saved to {path}", flush=True)


def _drive(api, goal) -> bool:
    goal = tuple(float(v) for v in goal)
    start, _ = _robot_xy_yaw(api)
    prepare_base_motion(api, goal[2])
    if np.linalg.norm(np.asarray(goal[:2]) - start) < 0.05:
        # An in-place turn stays in place even if the base first backed off to turn.
        here, _ = _robot_xy_yaw(api)
        goal = (float(here[0]), float(here[1]), goal[2])
    print(f"[approach] moving base to ({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f})", flush=True)
    return bool(api._env._navigate_to_pose(goal))


def scene_points(api, stride: int = 4) -> np.ndarray:
    """World XYZ of the head camera's valid depth pixels (every `stride` pixel)."""
    from capx.integrations.r1pro import control as c
    if not hasattr(api._env, "get_observation"):
        return np.zeros((0, 3))   # an API without cameras (unit-test fakes)
    _, depth = c.obs_get_rgb_depth(api._env.get_observation())
    depth = depth.detach().cpu().numpy() if hasattr(depth, "detach") else np.asarray(depth)
    depth = np.asarray(depth, dtype=float)
    if depth.ndim == 3:
        depth = depth[..., 0]
    mask = np.zeros(depth.shape, dtype=bool)
    mask[::stride, ::stride] = True
    mask &= np.isfinite(depth) & (depth > 0.1) & (depth < 6.0)
    if not mask.any():
        return np.zeros((0, 3))
    points = np.asarray(c.backproject_depth(mask, depth, api.get_camera_intrinsics(), api.get_camera_pose()))
    return points[np.isfinite(points).all(axis=1)]


def obstacle_xy(points, robot_xy, ignore=None) -> np.ndarray:
    """XYZ of obstacle points: robot-height, not the robot itself, not `ignore`d."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    keep = (points[:, 2] > OBSTACLE_Z[0]) & (points[:, 2] < OBSTACLE_Z[1])
    keep &= np.linalg.norm(points[:, :2] - robot_xy, axis=1) > SELF_RADIUS
    if ignore is not None:
        keep &= ~ignore(points)
    return points[keep]


def clear_distance(obstacles, start, goal) -> float:
    """How far along start->goal the base corridor is free (len(goal-start) if all)."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    length = float(np.linalg.norm(goal - start))
    if length < 1e-6 or len(obstacles) == 0:
        return length
    obstacles = np.asarray(obstacles, dtype=float)
    width = np.full(len(obstacles), CORRIDOR_HALF_WIDTH)
    if obstacles.shape[1] == 3:
        width[obstacles[:, 2] > BASE_TOP] = UPPER_HALF_WIDTH
    direction = (goal - start) / length
    relative = obstacles[:, :2] - start
    along = relative @ direction
    across = np.abs(relative @ np.array([-direction[1], direction[0]]))
    hits = along[(across < width) & (along > 0) & (along < length + FRONT_CLEARANCE)]
    if len(hits) == 0:
        return length
    return float(max(0.0, min(length, hits.min() - FRONT_CLEARANCE)))


def support_surface(points, object_position, object_points=None):
    """XY points of the horizontal surface the object rests on, or None."""
    found = support_layer(points, object_position, object_points)
    return None if found is None else found[0]


def support_layer(points, object_position, object_points=None):
    """(XY points, height) of the horizontal surface the object rests on, or None.

    The height is the most populated 2 cm height bin of the scene points around
    the object, below the object's top. The object's own lowest points are not
    used: on glass or glossy tops SAM3 also segments the object's mirror image,
    whose depth lies below the surface.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    obj = np.asarray(object_position, dtype=float)
    top = obj[2]
    if object_points is not None and len(object_points):
        top = float(np.percentile(np.asarray(object_points)[:, 2], 95))
    near = np.linalg.norm(points[:, :2] - obj[:2], axis=1) < 0.35
    below = near & (points[:, 2] < top - 0.03) & (points[:, 2] > 0.1)
    if below.sum() < 5:
        return None
    bins = np.floor(points[below, 2] / 0.02).astype(int)
    values, counts = np.unique(bins, return_counts=True)
    level = points[below, 2][bins == values[int(np.argmax(counts))]]
    height = float(np.median(level))
    layer = points[np.abs(points[:, 2] - height) < 0.025]
    layer = layer[np.linalg.norm(layer[:, :2] - obj[:2], axis=1) < 2.5]
    if len(layer) < 10:
        return None
    # Keep the 5 cm grid cells connected to the cells around the object.
    cells = {tuple(c) for c in np.floor(layer[:, :2] / 0.05).astype(int)}
    seed = tuple(np.floor(obj[:2] / 0.05).astype(int))
    frontier = [c for c in cells if abs(c[0] - seed[0]) <= 4 and abs(c[1] - seed[1]) <= 4]
    connected = set(frontier)
    while frontier:
        x, y = frontier.pop()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbour = (x + dx, y + dy)
                if neighbour in cells and neighbour not in connected:
                    connected.add(neighbour)
                    frontier.append(neighbour)
    if len(connected) < 6:
        return None
    keys = np.floor(layer[:, :2] / 0.05).astype(int)
    member = np.array([tuple(k) in connected for k in keys])
    return layer[member, :2], height


def sam3_support(api, support_name, object_position, object_points=None):
    """(XY points, height) of the SAM3-segmented support (e.g. "table") under the object, or None."""
    _, support_points = _observe(api, support_name)
    if support_points is None or len(support_points) < 3:
        return None
    support_points = np.asarray(support_points, dtype=float)
    obj = np.asarray(object_position, dtype=float)
    if np.min(np.linalg.norm(support_points[:, :2] - obj[:2], axis=1)) > 1.0:
        return None   # a different piece of furniture
    found = support_layer(support_points, object_position, object_points)
    if found is None:
        # Too few points for the layer search: the segment itself is the top.
        top = np.percentile(np.asarray(object_points)[:, 2], 95) if object_points is not None else obj[2]
        below = support_points[support_points[:, 2] < top - 0.03]
        if len(below) >= 3:
            found = below[:, :2], float(np.median(below[:, 2]))
    if found is not None:
        print(f"[approach] support '{support_name}' found by SAM3 at height {found[1]:.2f} m", flush=True)
    return found


def convex_hull(xy) -> np.ndarray:
    """Counter-clockwise hull vertices of 2-D points."""
    from scipy.spatial import ConvexHull
    xy = np.asarray(xy, dtype=float)
    return xy[ConvexHull(xy).vertices]


def inside_convex(hull, xy, margin=0.0) -> np.ndarray:
    """Whether each XY point lies inside the CCW convex polygon grown by `margin`."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    inside = np.ones(len(xy), dtype=bool)
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 1e-9:
            continue
        cross = (edge[0] * (xy[:, 1] - a[1]) - edge[1] * (xy[:, 0] - a[0])) / length
        inside &= cross >= -margin
    return inside


def edge_goal(surface_xy, object_xy, buffer=EDGE_BUFFER):
    """(x, y, yaw) outside the surface edge nearest the object, facing it, or None."""
    from scipy.spatial import ConvexHull, QhullError
    surface_xy = np.asarray(surface_xy, dtype=float)
    try:
        hull = surface_xy[ConvexHull(surface_xy).vertices]
    except (QhullError, ValueError):
        return None
    centre = hull.mean(axis=0)
    obj = np.asarray(object_xy, dtype=float)
    best = None
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 1e-6:
            continue
        t = float(np.clip((obj - a) @ edge / length ** 2, 0, 1))
        point = a + t * edge
        distance = float(np.linalg.norm(obj - point))
        if best is None or distance < best[0]:
            best = (distance, point, edge / length)
    if best is None:
        return None
    _, point, unit = best
    normal = np.array([-unit[1], unit[0]])
    if normal @ (centre - point) > 0:
        normal = -normal
    base = point + normal * buffer
    yaw = math.atan2(obj[1] - base[1], obj[0] - base[0])
    return float(base[0]), float(base[1]), yaw


AROUND_CLEARANCE = 0.55   # m from the support's edge while driving around it


def around_support(surface_xy, start, goal, clearance=AROUND_CLEARANCE):
    """Waypoints around the support's convex hull from start to goal (the short
    way round), or None when the straight segment does not cross the support.

    v9/v10 radio 6: the radio lies near the FAR edge of a 0.8 x 1.7 m coffee
    table, so its edge goal is on the other side; the local +-35 degree
    detours never got there and the base pushed into the table.
    """
    from shapely.geometry import LineString, MultiPoint, Point
    surface_xy = np.asarray(surface_xy, dtype=float)
    if len(surface_xy) < 3:
        return None
    hull = MultiPoint([tuple(p) for p in surface_xy]).convex_hull
    if hull.geom_type != "Polygon":
        return None
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    if not LineString([tuple(start), tuple(goal)]).intersects(hull.buffer(-0.05)):
        return None
    ring = hull.buffer(clearance, join_style="mitre", mitre_limit=2.0).exterior
    total = ring.length
    s_at, g_at = ring.project(Point(*start)), ring.project(Point(*goal))
    coords = np.asarray(ring.coords)[:-1]
    at = np.array([ring.project(Point(*c)) for c in coords])
    forward = (g_at - s_at) % total
    if forward <= total - forward:
        between = [(a - s_at) % total for a in at]
        order = [i for i in np.argsort(between) if 0 < between[i] < forward]
    else:
        between = [(s_at - a) % total for a in at]
        order = [i for i in np.argsort(between) if 0 < between[i] < total - forward]
    route = [coords[i] for i in order] + [goal]
    # Drop corners that can be cut: keep only the farthest waypoint reachable in
    # a straight line that stays clear of the support (by most of the clearance).
    keep_out = hull.buffer(clearance - 0.05)
    pruned, here = [], start
    index = 0
    while index < len(route):
        far = index
        for j in range(len(route) - 1, index - 1, -1):
            if not LineString([tuple(here), tuple(route[j])]).intersects(keep_out):
                far = j
                break
        pruned.append(np.asarray(route[far], dtype=float))
        here, index = route[far], far + 1
    return pruned


EDGE_REACH = 0.80   # m base centre to object: the left arm reaches this far (IK, torso bend)


def edge_goals(surface_xy, object_xy, buffer=EDGE_BUFFER) -> list:
    """(object distance, (x, y, yaw)) outside every hull edge of the support,
    at the edge point nearest the object, facing the object; nearest first."""
    from scipy.spatial import ConvexHull, QhullError
    surface_xy = np.asarray(surface_xy, dtype=float)
    try:
        hull = surface_xy[ConvexHull(surface_xy).vertices]
    except (QhullError, ValueError):
        return []
    centre = hull.mean(axis=0)
    obj = np.asarray(object_xy, dtype=float)
    goals = []
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 0.05:
            continue
        point = a + float(np.clip((obj - a) @ edge / length ** 2, 0, 1)) * edge
        normal = np.array([-edge[1], edge[0]]) / length
        if normal @ (centre - point) > 0:
            normal = -normal
        base = point + normal * buffer
        goals.append((float(np.linalg.norm(obj - base)),
                      (float(base[0]), float(base[1]), math.atan2(obj[1] - base[1], obj[0] - base[0]))))
    return sorted(goals, key=lambda g: g[0])


def choose_edge_goal(surface_xy, object_xy, robot_xy, obstacles, previous=None):
    """The edge goal to drive to: among goals within EDGE_REACH of the object
    whose base rectangle touches no obstacle, the one with the shortest drive
    (around the support if needed); else the nearest clear goal; else the
    nearest goal. v10 radio 4: the edge nearest the radio left only a 0.74 m
    gap to a sofa for the 0.67 m wide base."""
    goals = edge_goals(surface_xy, object_xy)
    if not goals:
        return None
    points = np.asarray(obstacles, dtype=float)
    if len(points) and points.shape[1] == 3:
        points = points[points[:, 2] < TURN_Z]
    clear = [g for g in goals if not len(points) or not footprint_hits(points, g[1]).any()]
    reachable = [g for g in clear if g[0] <= EDGE_REACH]

    def drive_length(goal):
        route = around_support(surface_xy, robot_xy, goal[:2])
        if route is None:
            return float(np.linalg.norm(np.asarray(goal[:2]) - robot_xy))
        path = [np.asarray(robot_xy)] + [np.asarray(w) for w in route]
        return float(sum(np.linalg.norm(b - a) for a, b in zip(path, path[1:])))
    if previous is not None:
        # Keep the goal chosen on the previous leg while it is still valid, so
        # the base does not alternate between two sides of the table.
        for g in reachable:
            if np.linalg.norm(np.asarray(g[1][:2]) - np.asarray(previous[:2])) < 0.15:
                return g[1]
    if reachable:
        return min(reachable, key=lambda g: drive_length(g[1]))[1]
    return (clear or goals)[0][1]


def sidestep_shifts(obstacles, start, direction, lookahead=SIDESTEP_LOOKAHEAD):
    """Lateral shift per side (+1 = left of `direction`) that clears the obstacle ahead.

    The obstacle is every point in the stretch the base would drive next
    (0..lookahead + FRONT_CLEARANCE ahead). For each side the shift is the
    smallest one that moves the base corridor past that obstacle's edge, plus
    SIDESTEP_MARGIN, so a wide table needs a long sidestep and a chair leg a
    short one. A side whose obstacle continues past MAX_SIDESTEP (or past the
    camera's view) gets None. With nothing ahead, both sides get SIDESTEP.
    """
    obstacles = np.asarray(obstacles, dtype=float).reshape(-1, obstacles.shape[1] if len(obstacles) else 3)
    direction = np.asarray(direction, dtype=float) / max(np.linalg.norm(direction), 1e-9)
    normal = np.array([-direction[1], direction[0]])
    if len(obstacles) == 0:
        return {1.0: SIDESTEP, -1.0: SIDESTEP}
    relative = obstacles[:, :2] - np.asarray(start, dtype=float)
    along, across = relative @ direction, relative @ normal
    width = np.full(len(obstacles), CORRIDOR_HALF_WIDTH)
    if obstacles.shape[1] == 3:
        width[obstacles[:, 2] > BASE_TOP] = UPPER_HALF_WIDTH
    ahead = (along > -0.3) & (along < lookahead + FRONT_CLEARANCE)
    if not ahead.any():
        return {1.0: SIDESTEP, -1.0: SIDESTEP}
    across, width = across[ahead], width[ahead]
    shifts = {}
    for sign in (1.0, -1.0):
        found = None
        for shift in np.arange(0.0, MAX_SIDESTEP + 1e-9, 0.05):
            if not np.any(np.abs(across - sign * shift) < width):
                found = float(shift)
                break
        if found is not None and found > 0:
            found = min(MAX_SIDESTEP, found + SIDESTEP_MARGIN)
        shifts[sign] = found
    return shifts


BASE_FRONT, BASE_REAR, BASE_HALF_WIDTH = 0.235, 0.392, 0.337
FOOTPRINT_MARGIN = 0.05


def footprint_hits(points, pose, margin=FOOTPRINT_MARGIN) -> np.ndarray:
    """Which XY(Z) points lie inside the base's rectangle at pose (x, y, yaw), grown by margin."""
    points = np.asarray(points, dtype=float).reshape(-1, np.asarray(points).shape[-1] if len(points) else 3)
    if not len(points):
        return np.zeros(0, dtype=bool)
    x, y, yaw = pose
    rel = points[:, :2] - np.array([x, y])
    c, s = math.cos(yaw), math.sin(yaw)
    forward = rel @ np.array([c, s])
    left = rel @ np.array([-s, c])
    return ((forward < BASE_FRONT + margin) & (forward > -BASE_REAR - margin)
            & (np.abs(left) < BASE_HALF_WIDTH + margin))


def swept_clear(obstacles, start, goal) -> bool:
    """Whether the base rectangle stays free of obstacle points (below TURN_Z) on
    the way from pose `start` to pose `goal` (start itself excluded: points the
    base already overlaps were dropped as its own body)."""
    points = np.asarray(obstacles, dtype=float)
    if len(points) and points.shape[1] == 3:
        points = points[points[:, 2] < TURN_Z]
    if not len(points):
        return True
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    turn = _wrap(goal[2] - start[2])
    count = max(2, int(math.ceil(max(np.linalg.norm(goal[:2] - start[:2]) / 0.05, abs(turn) / 0.1))) + 1)
    for t in np.linspace(0.0, 1.0, count)[1:]:
        pose = (*(start[:2] + t * (goal[:2] - start[:2])), start[2] + t * turn)
        if footprint_hits(points, pose).any():
            return False
    return True


def _plan_leg(api, goal_xy, facing, obstacles, look_at=None):
    """Goal for this leg: straight if clear, else a visible detour or the clear part.

    Waypoints short of `goal_xy` face `look_at` (the object) so it stays in view.
    """
    xy, _ = _robot_xy_yaw(api)

    def toward(point):
        if look_at is None:
            return facing
        return math.atan2(look_at[1] - point[1], look_at[0] - point[0])
    goal_xy = np.asarray(goal_xy, dtype=float)
    length = float(np.linalg.norm(goal_xy - xy))
    free = clear_distance(obstacles, xy, goal_xy)
    import os
    if os.environ.get("EXP001_APPROACH_DEBUG") and free < length - 1e-6 and len(obstacles):
        direction = (goal_xy - xy) / max(length, 1e-6)
        rel = np.asarray(obstacles)[:, :2] - xy
        along, across = rel @ direction, np.abs(rel @ np.array([-direction[1], direction[0]]))
        hit = (along > 0) & (across < CORRIDOR_HALF_WIDTH) & (along < length + FRONT_CLEARANCE)
        nearest = np.asarray(obstacles)[hit][np.argsort(along[hit])[:5]]
        print(f"[approach-debug] leg {np.round(xy, 2)} -> {np.round(goal_xy, 2)} free {free:.2f}/{length:.2f}; "
              f"nearest corridor points {np.round(nearest, 2).tolist()}", flush=True)
    if free >= length - 1e-6:
        return (*goal_xy, facing), "clear"
    # The corridor test keeps FRONT_CLEARANCE ahead in the direction of travel,
    # which rejects a short sideways/diagonal last leg to a table edge although
    # the base (facing the object) fits. Accept a leg the base rectangle can
    # actually sweep without touching anything.
    _, yaw_now = _robot_xy_yaw(api)
    if length <= NEAR_LEG and swept_clear(obstacles, (*xy, yaw_now), (*goal_xy, facing)):
        return (*goal_xy, facing), "clear"
    direction = (goal_xy - xy) / max(length, 1e-6)
    base_angle = math.atan2(direction[1], direction[0])
    best = None
    for degrees in DETOUR_HEADINGS:
        angle = base_angle + math.radians(degrees)
        for step in DETOUR_LENGTHS:
            waypoint = xy + step * np.array([math.cos(angle), math.sin(angle)])
            if clear_distance(obstacles, xy, waypoint) < step - 1e-6:
                continue
            remaining = float(np.linalg.norm(goal_xy - waypoint))
            if best is None or remaining < best[0]:
                best = (remaining, waypoint)
            break
    if best is not None and best[0] < length - 0.1:
        return (*best[1], toward(best[1])), "detour"
    if free >= MIN_PARTIAL:
        point = xy + free * direction
        return (*point, toward(point)), "partial"
    return None, "blocked"


def approach_object(self, object_name: str, standoff: float = 0.7,
                    support_name: str | None = None, max_step: float = 1.5,
                    max_legs: int = 8) -> dict:
    """Move the base next to an object so it can be grasped, avoiding obstacles.

    Repeats: observe the object with SAM3 -> drive one leg toward it, facing it
    -> observe again. Before every leg the head camera's depth image is checked
    for obstacles in the corridor the base would sweep; if something is in the
    way the base takes a detour (up to 35 degrees to the side) or stops short of
    it. Legs shorten as the object gets close. If the object is not visible at
    the start it first searches with find_object_base_rotate /
    find_object_torso_rotate; if it is lost after a leg, the base returns to the
    previous position and approaches from 35 degrees to the side.
    - Object on the floor: stops `standoff` m (base centre to object) away.
    - Object on furniture (table, counter, shelf...): the supporting surface is
      found automatically in the depth image and the base stops just outside
      the edge of that surface nearest to the object, facing the object.
    Obstacles seen earlier in the trial are remembered, so furniture that left
    the view still counts. Before each move a bent torso is straightened, and
    before a large turn the base first moves clear of nearby furniture. If the
    object is not in view at the end, "REACHED" still holds its last tracked
    position ("visible": False); pick_object grasps there with the wrist camera.

    Args:
        object_name: Text prompt for the object, e.g. "blue can of soda".
        standoff: Final distance to a floor object in metres (default 0.7).
        support_name: Optional text prompt of the surface (e.g. "table") used
            when the surface cannot be found automatically.
        max_step: Longest single straight-line leg in metres.
        max_legs: Maximum number of observe-and-move legs.
    Returns:
        dict with keys
          "status": "REACHED" | "NOT_FOUND" | "LOST_SIGHT" | "BLOCKED" | "MAX_LEGS",
          "distance": horizontal base-to-object distance in m (None if never seen),
          "object_position": last observed world position (None if never seen),
          "visible": whether the object is in view at return,
          "at_edge": True if it stopped at the edge of a supporting surface.
        Only "REACHED" means grasping can start (pick_object).
        On "BLOCKED" try another direction with navigate_to_pose; on
        "LOST_SIGHT" search again before retrying.

    Example:
        result = approach_object("blue can of soda")
        if result["status"] == "REACHED":
            picked = pick_object("blue can of soda")
    """
    if standoff <= 0 or max_step <= 0 or max_legs < 1:
        raise ValueError("standoff and max_step must be positive and max_legs >= 1")
    at_edge = False

    def report(status, position, visible):
        distance = None
        if position is not None:
            xy, _ = _robot_xy_yaw(self)
            distance = float(np.linalg.norm(position[:2] - xy))
        result = {"status": status, "distance": distance,
                  "object_position": None if position is None else position.copy(),
                  "visible": bool(visible), "at_edge": at_edge}
        print(f"[approach] {object_name}: {status}"
              + (f", {distance:.2f} m away" if distance is not None else "")
              + (" at the support edge" if at_edge else ""), flush=True)
        return result

    position, points = _observe(self, object_name)
    if position is None:
        for search in (self.find_object_base_rotate, self.find_object_torso_rotate):
            if search(object_name):
                position, points = _observe(self, object_name)
                if position is not None:
                    break
    if position is None:
        return report("NOT_FOUND", None, False)
    track = {"score": _score(self)}
    print(f"[approach] '{object_name}' at {np.round(position, 2)} (confidence {track['score']:.2f})", flush=True)

    sidesteps_left = 2
    looks_left = 2
    chosen_edge = None
    lost = 0   # times the object was lost after a leg; later legs approach from the side
    for _ in range(max_legs):
        xy, yaw = _robot_xy_yaw(self)
        offset = position[:2] - xy
        distance = float(np.linalg.norm(offset))
        if distance < 1e-6:
            return report("REACHED", position, True)
        direction = offset / distance
        facing = math.atan2(direction[1], direction[0])
        if position[2] > ELEVATED_Z and distance <= ELEVATED_REACH:
            # Already within arm reach of an object on furniture.
            if abs(_wrap(facing - yaw)) > FACE_TOLERANCE:
                _drive(self, (xy[0], xy[1], facing))
            break
        # The current view plus everything seen earlier in this trial: close to
        # the table its edge drops out of the head camera's view, and a partial
        # view of the top put the "edge" inside the table (v17 radio 1).
        cloud = scene_points(self)
        remembered = remember_obstacles(self, cloud)
        if len(remembered):
            cloud = np.vstack([cloud, remembered])
        object_radius = 0.25
        surface_height = None

        def is_target(p, centre=position):
            near = np.linalg.norm(p[:, :2] - centre[:2], axis=1) < object_radius
            if surface_height is not None:
                near &= p[:, 2] > surface_height + 0.02   # the object, not the furniture under it
            return near

        # Objects resting on furniture: go to the nearest edge of the surface.
        target_goal, edge = None, None
        if points is not None and len(points) and np.percentile(points[:, 2], 95) > ELEVATED_Z:
            found = support_layer(cloud, position, points)
            if found is None:
                found = sam3_support(self, support_name or "table", position, points)
            if found is not None:
                surface, surface_height = found
                kept = points[points[:, 2] > surface_height + 0.01]
                if len(kept) >= 6 and len(kept) < len(points):
                    # Points below the top are the object's mirror image.
                    position = np.median(kept, axis=0)
                    points = kept
                    offset = position[:2] - xy
                    distance = float(np.linalg.norm(offset))
                    direction = offset / max(distance, 1e-6)
                    facing = math.atan2(direction[1], direction[0])
                if position[2] > ELEVATED_Z and distance <= ELEVATED_REACH:
                    if abs(_wrap(facing - yaw)) > FACE_TOLERANCE:
                        _drive(self, (xy[0], xy[1], facing))
                    break
                if distance - EDGE_BUFFER <= max_step + 1.0:
                    edge = choose_edge_goal(surface, position[:2], xy, obstacle_xy(cloud, xy, is_target),
                                            previous=chosen_edge)
                    chosen_edge = edge
        if edge is not None:
            # The whole top stays an obstacle. The goal is EDGE_BUFFER (0.32 m)
            # outside the edge and legs stop FRONT_CLEARANCE (0.30 m) short of
            # obstacle points, so a correct goal is still reachable, while a goal
            # wrongly placed inside the top (partial view) now stops at the edge
            # instead of driving into the table (v9 radio 6 pushed it 0.27 m).
            ignore = is_target
            to_edge = np.asarray(edge[:2]) - xy
            if np.linalg.norm(to_edge) <= REACH_TOLERANCE:
                at_edge = True
                if abs(_wrap(edge[2] - yaw)) > FACE_TOLERANCE:
                    _drive(self, (xy[0], xy[1], edge[2]))
                break
            route = around_support(surface, xy, edge[:2])
            if route is not None and len(route) > 1:
                # The goal is across the support: drive around it, one hull
                # corner per leg, still facing the object to keep it in view.
                waypoint = next((w for w in route if np.linalg.norm(w - xy) > REACH_TOLERANCE), route[-1])
                step = np.asarray(waypoint) - xy
                if np.linalg.norm(step) > max_step:
                    waypoint = xy + max_step * step / np.linalg.norm(step)
                print(f"[approach] the edge nearest the object is across the support; going around "
                      f"via {np.round(waypoint, 2)} ({len(route) - 1} corner(s) left)", flush=True)
                target_goal = np.asarray(waypoint, dtype=float)
                target_facing = math.atan2(position[1] - target_goal[1], position[0] - target_goal[0])
            elif np.linalg.norm(to_edge) > max_step:
                target_goal = xy + max_step * to_edge / np.linalg.norm(to_edge)
                target_facing = math.atan2(position[1] - target_goal[1], position[0] - target_goal[0])
            else:
                target_goal, target_facing = np.asarray(edge[:2]), edge[2]
        else:
            ignore = is_target
            if position[2] > ELEVATED_Z:
                # On furniture whose top could not be found: stay within arm reach.
                standoff = min(standoff, ELEVATED_STANDOFF)
            if distance <= standoff + REACH_TOLERANCE:
                if abs(_wrap(facing - yaw)) > FACE_TOLERANCE:
                    _drive(self, (xy[0], xy[1], facing))
                break
            remaining = distance - standoff
            step = min(max_step, remaining)
            if distance < NEAR_LEG and remaining > 0.4:
                step = min(step, max(0.4, 0.6 * remaining))
            heading = direction
            if lost:
                angle = math.radians(LOST_DETOUR_DEG) * (1 if lost % 2 else -1)
                heading = np.array([[math.cos(angle), -math.sin(angle)],
                                    [math.sin(angle), math.cos(angle)]]) @ direction
                step = min(step, 1.0)
            target_goal, target_facing = xy + step * heading, facing

        obstacles = obstacle_xy(cloud, xy, ignore)
        goal, how = _plan_leg(self, target_goal, target_facing, obstacles, look_at=position[:2])
        if goal is None and looks_left > 0:
            # Nothing free in view: turn to look sideways for a way around.
            looks_left -= 1
            for degrees in LOOK_ASIDE:
                angle = facing + math.radians(degrees)
                _drive(self, (xy[0], xy[1], angle))
                side_obstacles = obstacle_xy(scene_points(self), xy, ignore)
                side = xy + 0.8 * np.array([math.cos(angle), math.sin(angle)])
                free = clear_distance(side_obstacles, xy, side)
                if free >= 0.4:
                    print(f"[approach] path blocked ahead; going around via {degrees:+d} degrees", flush=True)
                    goal, how = (*(xy + free * (side - xy) / 0.8), angle), "look-aside"
                    break
        if goal is None:
            print("[approach] path toward the object is blocked in the camera view", flush=True)
        else:
            if how != "clear":
                print(f"[approach] obstacle ahead; taking a {how} leg", flush=True)
            reached = _drive(self, goal)
            if how == "look-aside":
                _face(self, position[:2])
        moved = float(np.linalg.norm(_robot_xy_yaw(self)[0] - xy))
        if goal is None or (not reached and moved < MIN_PROGRESS):
            # Blocked at the start of the leg: step sideways past the obstacle's
            # visible edge (shorter side first), once each way.
            normal = np.array([-direction[1], direction[0]])
            escaped = False
            start = _robot_xy_yaw(self)[0]
            shifts = sidestep_shifts(obstacles, start, direction)
            order = sorted((sign for sign in shifts if shifts[sign]), key=lambda sign: shifts[sign])
            if not order:
                print("[approach] the obstacle ahead is wider than a sidestep on both sides", flush=True)
            for sign in order:
                if sidesteps_left <= 0:
                    break
                sidesteps_left -= 1
                start = _robot_xy_yaw(self)[0]
                shift = shifts[sign]
                side = start + sign * shift * normal
                if clear_distance(obstacles, start, side) < shift - 1e-6:
                    continue
                print(f"[approach] stepping {shift:.2f} m to the {'left' if sign > 0 else 'right'} "
                      "past the obstacle's edge", flush=True)
                _drive(self, (*side, facing))
                if np.linalg.norm(_robot_xy_yaw(self)[0] - start) >= MIN_PROGRESS:
                    escaped = True
                    break
            if not escaped:
                seen, _ = _observe_tracked(self, object_name, position, track)
                if seen is not None and seen[2] > ELEVATED_Z and \
                        np.linalg.norm(seen[:2] - _robot_xy_yaw(self)[0]) <= ELEVATED_REACH:
                    position = seen
                    break
                return report("BLOCKED", position if seen is None else seen, seen is not None)

        seen, seen_points = _observe_tracked(self, object_name, position, track)
        if seen is None and self.find_object_torso_rotate(object_name):
            seen, seen_points = _observe_tracked(self, object_name, position, track)
        if seen is None and lost < 2:
            # Hidden behind something: go back to where it was visible and
            # approach from the side next time.
            print("[approach] lost sight; returning to the previous position", flush=True)
            _drive(self, (xy[0], xy[1], facing))
            seen, seen_points = _observe_tracked(self, object_name, position, track)
            lost += 1
        if seen is None:
            return report("LOST_SIGHT", position, False)
        position, points = seen, seen_points
    else:
        return report("MAX_LEGS", position, True)

    seen, _ = _observe_tracked(self, object_name, position, track)
    # No torso search here: right at the furniture a forward bend hit the table
    # (v9 radio 1/3/11), and pick_object grasps at this tracked position with
    # the wrist camera even when the head camera no longer sees the object.
    final = position if seen is None else seen
    _trial_memory(self)[TRACKED_PREFIX + object_name] = np.asarray(final, dtype=float).copy()
    return report("REACHED", final, seen is not None)


def _trial_memory(api) -> dict:
    """Object positions seen in THIS trial only (rulebook 8.4: nothing crosses trials).

    Invalidated when the env is reset (reset id changes) or its step counter
    goes backwards, which also happens on the gateway shim between trials.
    """
    env = api._env
    key = (id(env), getattr(env, "_exp001_reset_id", None))
    count = int(getattr(env, "_sim_step_count", 0))
    memory = getattr(api, "_exp001_seen", None)
    if memory is None or memory["key"] != key or count < memory["count"]:
        memory = {"key": key, "count": count, "objects": {}}
        api._exp001_seen = memory
    memory["count"] = count
    return memory["objects"]


TRACKED_PREFIX = "approach::"


def tracked_position(api, object_name):
    """Where approach_object last confirmed the object in this trial, or None."""
    position = _trial_memory(api).get(TRACKED_PREFIX + object_name)
    return None if position is None else position.copy()


def last_seen_position(api, object_name):
    position = _trial_memory(api).get(object_name)
    return None if position is None else position.copy()


def _face(api, target_xy) -> None:
    xy, yaw = _robot_xy_yaw(api)
    offset = np.asarray(target_xy, dtype=float) - xy
    if np.linalg.norm(offset) < 1e-6:
        return
    facing = math.atan2(offset[1], offset[0])
    if abs(_wrap(facing - yaw)) > FACE_TOLERANCE:
        prepare_base_motion(api, facing)
        xy, _ = _robot_xy_yaw(api)
        api._env._navigate_to_pose((xy[0], xy[1], facing))


# --- Safe base motion: trial-local obstacle memory, posture, turning room ------
# In radio trials the robot tipped over (up to 95 degrees) and shoved the glass
# coffee table because (BEHAVIOR.md, v10):
#  * it turned in place next to the table: the base's rear corners sweep a
#    0.52 m radius, but the head camera no longer sees furniture beside/behind it;
#  * it drove with the torso still bent or twisted by an earlier search / IK
#    move, or with a hand still out over the table;
#  * torso searches bent forward from a twisted torso, right at the table edge.
# Everything here uses only the head camera's depth, proprioception and the
# hands' poses, i.e. what the gateway gives a submission.
MEMORY_KEY = "__obstacle_points__"
MEMORY_VOXEL = 0.04
ARM_CLEARANCE = 0.15   # m around the segment torso->hand: the robot's own arm
TURN_Z = 1.2           # base, tucked arms and lower torso sweep below this
TURN_CHECK = 0.35      # rad; smaller turns sweep almost nothing
MAX_ESCAPE = 0.45
TORSO_TILTED = 0.25    # rad; any torso joint beyond this is straightened before driving
HAND_OUT = 0.60        # m from the base centre: a hand this far out is pulled in first
TORSO_ROOM = 0.60      # m ahead of the base centre kept free before bending forward


def _segment_distance(points, a, b):
    ab = b - a
    t = np.clip((points - a) @ ab / max(float(ab @ ab), 1e-9), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def _joint_positions(api):
    """Proprioceptive joint positions (28), or None for an API without them (test fakes)."""
    if not hasattr(api, "get_current_joint_positions"):
        return None
    return np.asarray(api.get_current_joint_positions(), dtype=float)


def _hand_positions(api) -> list:
    if not hasattr(api, "get_current_eef_pose"):
        return []
    return [np.asarray(api.get_current_eef_pose(arm=arm)[0], dtype=float) for arm in (0, 1)]


def remember_obstacles(api, cloud=None) -> np.ndarray:
    """Add the head camera's obstacle points to this trial's memory; return all
    remembered points (world XYZ). Points of the robot itself (base radius, the
    arms) are not stored."""
    if cloud is None:
        cloud = scene_points(api)
    xy, _ = _robot_xy_yaw(api)
    points = obstacle_xy(cloud, xy)
    torso = np.array([xy[0], xy[1], 1.1])
    for hand in _hand_positions(api):
        if len(points):
            points = points[_segment_distance(points, torso, hand) > ARM_CLEARANCE]
    store = _trial_memory(api).setdefault(MEMORY_KEY, {})
    if len(points):
        keys = np.floor(points / MEMORY_VOXEL).astype(int)
        _, first = np.unique(keys, axis=0, return_index=True)
        for i in first:
            store.setdefault(tuple(keys[i]), points[i])
    return remembered_obstacles(api)


def remembered_obstacles(api) -> np.ndarray:
    store = _trial_memory(api).get(MEMORY_KEY) or {}
    return np.array(list(store.values())) if store else np.zeros((0, 3))


def tidy_posture(api) -> None:
    """Straighten a bent/twisted torso and pull in a hand that is out, before the base moves.

    Driving with the torso bent (left by find_object_torso_rotate or a hand IK
    move) or a hand over the table rocked the robot and swept the table.
    Holding something is kept: the hand is only raised, never opened.
    """
    env = api._env
    q = _joint_positions(api)
    hands = _hand_positions(api)
    if q is None or not hands:
        return
    xy, _ = _robot_xy_yaw(api)
    hand_out = any(np.linalg.norm(h[:2] - xy) > HAND_OUT for h in hands)
    torso_bent = bool(np.any(np.abs(q[6:10]) > TORSO_TILTED))
    if not hand_out and not torso_bent:
        return
    print(f"[base] straightening the torso{' and pulling in the hands' if hand_out else ''} "
          "before moving the base", flush=True)
    if hand_out:
        # Raise the outstretched hand first so pulling it in clears the furniture.
        from r1pro_skills.control import _move_hand_with_ik
        for arm, hand in enumerate(hands):
            if np.linalg.norm(hand[:2] - xy) > HAND_OUT:
                _, quat = api.get_current_eef_pose(arm=arm)
                _move_hand_with_ik(api, hand + np.array([0.0, 0.0, 0.12]), np.asarray(quat, dtype=float),
                                   arm, "raise before moving")
    target = np.asarray(api.get_current_joint_positions(), dtype=float).copy()
    target[6:10] = 0.0
    if hand_out:
        # Arm joints are interleaved (left, right) from index 10; all zero is the
        # tucked reset pose (reset_robot_joints).
        for arm in (0, 1):
            if np.linalg.norm(hands[arm][:2] - xy) > HAND_OUT:
                target[10 + arm:24:2] = 0.0
    env._move_to_joint_positions(target, max_steps=150, settle_steps=10)


def make_turn_room(api, goal_yaw) -> None:
    """Before turning more than TURN_CHECK in place, translate (keeping the yaw)
    until the base rectangle can sweep the turn without touching any
    remembered obstacle point."""
    xy, yaw = _robot_xy_yaw(api)
    if abs(_wrap(goal_yaw - yaw)) < TURN_CHECK:
        return
    points = remember_obstacles(api)
    points = points[(points[:, 2] < TURN_Z)] if len(points) else points
    if not len(points) or swept_clear(points, (*xy, yaw), (*xy, goal_yaw)):
        return
    near = float(np.min(np.linalg.norm(points[:, :2] - xy, axis=1)))
    best = None
    for angle in np.linspace(0, 2 * math.pi, 24, endpoint=False):
        direction = np.array([math.cos(angle), math.sin(angle)])
        for step in np.arange(0.05, MAX_ESCAPE + 1e-9, 0.05):
            spot = xy + step * direction
            if not swept_clear(points, (*xy, yaw), (*spot, yaw)):
                break
            if swept_clear(points, (*spot, yaw), (*spot, goal_yaw)):
                if best is None or step < best[0]:
                    best = (step, spot)
                break
    if best is None:
        print(f"[base] obstacle {near:.2f} m from the base centre and no room to back off; "
              "turning anyway", flush=True)
        return
    step, goal = best
    print(f"[base] the turn would sweep an obstacle {near:.2f} m from the base centre; "
          f"moving {step:.2f} m clear before turning", flush=True)
    api._env._navigate_to_pose((float(goal[0]), float(goal[1]), float(yaw)))


def prepare_base_motion(api, goal_yaw) -> None:
    """Tidy the posture, then make room for a large turn."""
    tidy_posture(api)
    make_turn_room(api, goal_yaw)


def make_torso_room(api) -> None:
    """Back off (straight back) until TORSO_ROOM ahead of the base is free of
    remembered obstacles, before bending the torso forward."""
    xy, yaw = _robot_xy_yaw(api)
    points = remember_obstacles(api)
    if not len(points):
        return
    points = points[points[:, 2] < TURN_Z]
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    relative = points[:, :2] - xy
    along = relative @ forward
    across = np.abs(relative @ np.array([-forward[1], forward[0]]))
    ahead = along[(along > 0) & (across < CORRIDOR_HALF_WIDTH + 0.05)]
    if not len(ahead) or ahead.min() >= TORSO_ROOM:
        return
    back = min(MAX_ESCAPE, TORSO_ROOM - float(ahead.min()))
    behind = along[(along < 0) & (across < CORRIDOR_HALF_WIDTH)]
    if len(behind) and -behind.max() < 0.392 + back + 0.05:
        back = max(0.0, -behind.max() - 0.392 - 0.05)
    if back < 0.05:
        return
    goal = xy - back * forward
    print(f"[torso] obstacle {ahead.min():.2f} m ahead; backing off {back:.2f} m before bending forward",
          flush=True)
    api._env._navigate_to_pose((float(goal[0]), float(goal[1]), float(yaw)))


# Torso search samples as (fraction of the full tilt). The upstream helper visits
# 6 poses down and 6 up, moving each torso joint separately (24 servo moves);
# this visits 4 down and 2 up with the same extremes.
TORSO_DOWN = (0.25, 0.5, 0.75, 1.0)
TORSO_UP = (0.5, 1.0)
TORSO_RANGE = (1.5, 2.5)   # joint1 back, joint2 forward at full downward tilt
TORSO_MAX_STEPS = 30


def find_object_torso_rotate(self, object_name: str) -> bool:
    """Tilt the torso to move the head camera down, then up, until the object is seen.

    Checks the current view first, then 4 downward and 2 upward tilts. The torso
    is first untwisted, and the base backs off if furniture is right in front,
    so the bend does not hit it. The torso stays at the tilt where the object
    was found (the next base motion straightens it). Returns whether the
    object was detected.
    """
    if self.detect_object_sam3(object_name):
        return True
    # Bend from an upright, untwisted torso (an IK move may have left torso
    # joints 3/4 twisted, so "forward" pointed sideways) and with room ahead:
    # bending at the table edge shoved the table (v9 radio 1/3/11).
    q = _joint_positions(self)
    if q is not None and np.any(np.abs(q[8:10]) > TORSO_TILTED):
        target = q.copy()
        target[8:10] = 0.0
        self._env._move_to_joint_positions(target, max_steps=100, settle_steps=10)
        if self.detect_object_sam3(object_name):
            return True
    make_torso_room(self)
    joints = self._env.controller.robot.joints
    start1 = float(joints["torso_joint1"].get_state()[0][0])
    start2 = float(joints["torso_joint2"].get_state()[0][0])

    def tilt(fraction):
        self._env.update_torso("torso_joint1", start1 - fraction * TORSO_RANGE[0], TORSO_MAX_STEPS)
        self._env.update_torso("torso_joint2", start2 + fraction * TORSO_RANGE[1], TORSO_MAX_STEPS)
        return self.detect_object_sam3(object_name)

    for fraction in TORSO_DOWN:
        if tilt(fraction):
            return True
    for fraction in TORSO_UP:
        if tilt(-fraction):
            return True
    tilt(0.0)
    return False


def install_search(api_class):
    """Trial-local object memory, memory-first base search, cheaper torso search."""
    original_pose = api_class.get_object_pose
    original_base_search = api_class.find_object_base_rotate

    def get_object_pose(self, object_name: str, return_bbox_extent: bool = False) -> tuple:
        result = original_pose(self, object_name, return_bbox_extent)
        if result[0] is not None:
            _trial_memory(self)[object_name] = np.asarray(result[0], dtype=float).copy()
        return result

    def find_object_base_rotate(self, object_name: str) -> bool:
        known = last_seen_position(self, object_name)
        if known is not None:
            _face(self, known[:2])
            if self.detect_object_sam3(object_name):
                print(f"[search] {object_name}: found again at its last seen position", flush=True)
                return True
        return bool(original_base_search(self, object_name))

    get_object_pose.__doc__ = original_pose.__doc__
    get_object_pose.__annotations__ = dict(getattr(original_pose, "__annotations__", {}))
    find_object_base_rotate.__doc__ = (
        "Rotate the robot base until the object is found in the current field of view.\n"
        "If the object was already located earlier in this trial (get_object_pose), the\n"
        "base first turns toward that position, which avoids a full rotation.\n"
        "Args:\n    object_name: Name of the object to find.\n"
        "Returns:\n    success: Whether the object was found.")
    api_class.get_object_pose = get_object_pose
    api_class.find_object_base_rotate = find_object_base_rotate
    api_class.find_object_torso_rotate = find_object_torso_rotate


def install(api_class):
    """Expose approach_object to generated code (and to the API prompt)."""
    install_search(api_class)
    api_class.approach_object = approach_object
    original_functions = api_class.functions

    def functions(self):
        fns = dict(original_functions(self))
        fns["approach_object"] = self.approach_object
        return fns

    api_class.functions = functions
