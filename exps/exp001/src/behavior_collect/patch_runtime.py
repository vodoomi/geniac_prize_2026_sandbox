"""Apply exp001 compatibility changes to the pinned simulator tree.

CaP-X records its own multi-turn artifacts. Upstream's moving LeRobot branch
now requires Python 3.12, while Isaac Sim 5.1 requires Python 3.11. Extend the
upstream Blackwell cuRobo embodiment workaround to B300; physics, observations
and scoring are unchanged.
"""
if __package__ in (None, ""):
    # Run as a file: put src/ (not this package directory) first on sys.path so
    # sibling modules are imported through the package and cannot shadow others.
    import sys as _sys
    from pathlib import Path as _Path
    _here = _Path(__file__).resolve().parent
    _sys.path[:] = [p for p in _sys.path if _Path(p or ".").resolve() != _here]
    _sys.path.insert(0, str(_here.parent))

from pathlib import Path


def patch_motion_gen(text):
    """Keep IK validity independent from mutable trajectory/graph results.

    MotionGenResult stores the tensor by reference. A failed graph search clears
    result.success in-place; without a clone this also clears ik_result.success,
    and the subsequent goal assignment indexes zero rows with nonempty goals.
    """
    for method in ("_plan_from_solve_state", "_plan_from_solve_state_batch"):
        start = text.index(f"    def {method}(")
        end = text.find("\n    def ", start + 1)
        if end == -1:
            end = len(text)
        block = text[start:end]
        mask = "ik_result.success" if method.endswith("_batch") else "ik_result.success.view(-1)[0:1]"
        old = f"result = MotionGenResult(\n            {mask},"
        new = f"result = MotionGenResult(\n            {mask}.clone(),"
        if old in block:
            block = block.replace(old, new, 1)
        assert new in block, f"Unexpected cuRobo result construction in {method}"
        # Chained boolean indexing writes a temporary tensor, not the result.
        block = block.replace(
            "result.success[ik_success][graph_result.success] = True",
            "result.success[:] = False\n"
            "                    result.success[torch.nonzero(ik_success).view(-1)[graph_result.success]] = True",
        )
        text = text[:start] + block + text[end:]
    return text


def main():
    upstream = Path(__file__).resolve().parents[2] / "runtime/BEHAVIOR-1K"
    setup = upstream / "OmniGibson/setup.py"
    text = setup.read_text()
    line = '        "lerobot[dataset] @ git+https://github.com/wensi-ai/lerobot@release/b1k",\n'
    if line in text:
        setup.write_text(text.replace(line, ""))
    wrapper = upstream / "OmniGibson/omnigibson/envs/__init__.py"
    text = wrapper.read_text()
    for line in ["from omnigibson.envs.lerobot_data_wrapper import LeRobotDataWrapper, LeRobotPlaybackWrapper\n",
                 '    "LeRobotDataWrapper",\n', '    "LeRobotPlaybackWrapper",\n']:
        text = text.replace(line, "")
    wrapper.write_text(text)
    print("[setup] Disabled unused LeRobot recording wrapper; CaP-X recording is unchanged")
    curobo = upstream / "OmniGibson/omnigibson/action_primitives/curobo.py"
    text = curobo.read_text()
    original = 'if th.cuda.get_device_capability(device) == (12, 0):'
    patched = 'if th.cuda.get_device_capability(device) in ((10, 0), (10, 3), (12, 0)):'
    if original in text:
        text = text.replace(original, patched)
    assert patched in text, "Unexpected cuRobo workaround source"
    text = text.replace('with cuda architecture 12.0 GPU:', 'with Blackwell GPU:')
    # All MotionGen embodiments share one world collision checker. DEFAULT is
    # absent on B300, so update it through an available embodiment instead.
    original = 'self.mg[CuRoboEmbodimentSelection.DEFAULT].update_world(world)'
    patched = 'next(iter(self.mg.values())).update_world(world)'
    text = text.replace(original, patched)
    assert patched in text, "Unexpected cuRobo obstacle-update source"
    # Geometric graph search changes the batch seed shape in cuRobo. BASE's
    # optimizer otherwise replays a CUDA graph captured with the old shape
    # (observed: batch 12 vs 3). Use eager execution for BASE only; ARM keeps
    # its existing CUDA graph acceleration and collision checking is unchanged.
    marker = '            if motion_cfg_kwargs is not None:\n'
    eager_base = ('            if emb_sel == CuRoboEmbodimentSelection.BASE:\n'
                  '                motion_kwargs["use_cuda_graph"] = False\n')
    if eager_base not in text:
        assert marker in text, "Unexpected cuRobo MotionGen construction source"
        text = text.replace(marker, eager_base + marker, 1)
    curobo.write_text(text)
    print("[setup] B300 cuRobo: BASE/ARM planners and shared obstacle updates enabled")
    print("[setup] BASE planner uses eager execution for geometric graph search")
    motion_gen = upstream.parent / "venv/lib/python3.11/site-packages/curobo/wrap/reacher/motion_gen.py"
    original = motion_gen.read_text()
    patched = patch_motion_gen(original)
    backup = upstream.parent / "motion_gen.py.before_reliability_fix"
    if not backup.exists():
        backup.write_text(original)
    if patched != original:
        motion_gen.write_text(patched)
    import difflib
    (upstream.parent / "curobo_reliability.patch").write_text("".join(difflib.unified_diff(
        backup.read_text().splitlines(True), patched.splitlines(True),
        fromfile="curobo/motion_gen.py.original", tofile="curobo/motion_gen.py")))
    print("[setup] cuRobo preserves IK masks when graph planning fails")


if __name__ == "__main__":
    main()
