"""Use baseline's planner with the explicit exp001 BEHAVIOR instance policy.

Only this process changes the seed guard; the baseline checkout is untouched.
All other preflight checks, config rewrites, model servers and outputs are reused.
"""
from __future__ import annotations

if __package__ in (None, ""):
    # Run as a file: put src/ (not this package directory) first on sys.path so
    # sibling modules are imported through the package and cannot shadow others.
    import sys as _sys
    from pathlib import Path as _Path
    _here = _Path(__file__).resolve().parent
    _sys.path[:] = [p for p in _sys.path if _Path(p or ".").resolve() != _here]
    _sys.path.insert(0, str(_here.parent))

import argparse
import hashlib
import json
import shlex
from pathlib import Path

import yaml
from geniac_baseline.collect import plan
from geniac_baseline.collect.preflight import PreflightError

EXP = Path(__file__).resolve().parents[2]
ROOT = EXP.parents[1]


def check_behavior_instances(mode, task, seeds, n_layouts=None):
    split = yaml.safe_load((ROOT / "evaluation/split.yaml").read_text())
    if mode != "collect" or task not in split["development_task_ids"]["behavior"] or n_layouts is not None:
        raise PreflightError(f"{task}: exp001 BEHAVIOR development collection only")
    allowed = set(split["initial_state_policy"]["behavior"]["development_instance_ids"])
    ids = list(seeds)
    if not ids or not set(ids) <= allowed:
        raise PreflightError(f"{task}: only development instance IDs {sorted(allowed)} are allowed, got {ids}")


def validated_spec(path: Path):
    spec = plan.load_spec(path)
    if spec.num_workers != 1:
        raise PreflightError("BEHAVIOR's Kit bootstrap currently supports num_workers: 1 only")
    seen = set()
    for task in spec.tasks:
        cfg = yaml.safe_load(task.config.read_text())
        low = cfg["env"]["cfg"]["low_level"]
        if low.get("_target_") != "capx.envs.simulators.r1pro_b1k.R1ProBehaviourLowLevel":
            raise PreflightError(f"{task.task}: not the BEHAVIOR simulator")
        controller = spec.capx_root / "capx/envs/simulators/configs" / low.get("controller_cfg", "r1pro_primitives.yaml")
        activity = low.get("activity_name", yaml.safe_load(controller.read_text())["task"]["activity_name"])
        expected = "turning_on_radio" if task.task == "picking_up_radio" else task.task
        if activity != expected or cfg["env"]["cfg"].get("privileged") or cfg.get("use_oracle_code"):
            raise PreflightError(f"{task.task}: activity/observation/oracle mismatch")
        for segment in task.seeds:
            start, end = plan.parse_segment(segment)
            check_behavior_instances(spec.mode, task.task, range(start, end + 1), task.layouts)
            for instance in range(start, end + 1):
                key = (task.task, instance)
                if key in seen:
                    raise PreflightError(f"Duplicate task/instance: {key}")
                seen.add(key)
    plan.check_seeds = check_behavior_instances
    plan.preflight(spec)
    return spec


def collection_digest(path: Path) -> str:
    spec = validated_spec(path)
    # Everything that changes what a trial does: the collection harness, the
    # API extensions the generated code calls, and the launch scripts.
    files = [path, ROOT / "evaluation/split.yaml"]
    for package in ("behavior_collect", "r1pro_skills"):
        files.extend(sorted((EXP / "src" / package).glob("*.py")))
    files.extend([EXP / "scripts/behavior_env.sh", EXP / "scripts/run_behavior.sh"])
    for task in spec.tasks:
        files.append(task.config)
        low = yaml.safe_load(task.config.read_text())["env"]["cfg"]["low_level"]
        files.append(spec.capx_root / "capx/envs/simulators/configs" / low["controller_cfg"])
    digest = hashlib.sha256()
    for f in files:
        digest.update(str(f.resolve()).encode() + b"\0" + f.read_bytes() + b"\0")
    return digest.hexdigest()


# Each lane runs its segments one after another with its own SAM3/GraspNet
# sidecars; all lanes share the run's vLLM servers. Lane L moves the perception
# ports by LANE_PORT_STRIDE * L on top of the spec's port_offset (8313-8317,
# 8413-8417, ...), clear of the vLLM ports (8310-8311, 8322).
LANE_PORT_STRIDE = 100
MAX_LANES = 4
LANE_STAGGER_S = 120
PERCEPTION_SERVICES = ("SAM2", "SAM3", "GRASPNET", "PYROKI", "OWLVIT")


def lane_port_offset(spec, lane: int) -> int:
    return spec.port_offset + LANE_PORT_STRIDE * lane


def sidecar_ports(spec, lanes: int) -> list[int]:
    """Perception ports the run binds (checked free before starting)."""
    from geniac_baseline.collect import settings as S
    return sorted(p + lane_port_offset(spec, lane) for lane in range(lanes) for p in S.PERCEPTION_PORTS)


def _segment_blocks(lines: list[str]) -> tuple[int, int, list[list[str]]]:
    """Locate plan.write_plan's sequential segment blocks (echo, launch, RC=, echo)."""
    start = lines.index("declare -A RC") + 1
    end = lines.index('echo "[run] done; trial folders under $OUT/seg_*/outputs/"') - 1
    body = lines[start:end]
    if len(body) % 4:
        raise RuntimeError("Unexpected run.sh layout from geniac_baseline.collect.plan")
    return start, end, [body[i:i + 4] for i in range(0, len(body), 4)]


def parallelize(spec, script: Path, lanes: int) -> None:
    """Rewrite run.sh so segments run in ``lanes`` parallel lanes sharing the vLLM servers."""
    from geniac_baseline.collect import settings as S
    from geniac_baseline.collect.run_config import write_run_config

    lines = script.read_text().splitlines()
    start, end, blocks = _segment_blocks(lines)
    segments = [(t, *plan.parse_segment(s)) for t in spec.tasks for s in t.seeds]
    if len(blocks) != len(segments):
        raise RuntimeError("Segment count differs between spec and run.sh")
    # Longest segments first onto the least-loaded lane.
    order = sorted(range(len(segments)), key=lambda i: segments[i][2] - segments[i][1], reverse=True)
    load, assignment = [0] * lanes, {}
    for i in order:
        lane = min(range(lanes), key=lambda k: load[k])
        assignment[i] = lane
        load[lane] += segments[i][2] - segments[i][1] + 1
    lane_lines: list[list[str]] = [[] for _ in range(lanes)]
    for i, (block, (task, a, b)) in enumerate(zip(blocks, segments)):
        lane = assignment[i]
        offset = lane_port_offset(spec, lane)
        seg = spec.output_dir / f"seg_{task.task}_{a}"
        # The run config's api_servers carry the sidecar ports; rewrite for this lane.
        write_run_config(task.config, seg / f"{task.task}.yaml", start=a, end=b,
                         port_offset=offset, num_workers=spec.num_workers)
        env = " ".join(f"{name}_SERVICE_URL=http://127.0.0.1:{port + offset}"
                       for name, port in zip(PERCEPTION_SERVICES, S.PERCEPTION_PORTS, strict=True))
        echo, launch, _, _ = block
        key = f"{task.task}:{a}-{b}"
        rc = shlex.quote(str(seg / ".rc"))
        lane_lines[lane] += [
            f"  {echo[:-1]} lane={lane}'",
            f"  {env} {launch}",
            f"  rc=$?; echo \"$rc\" > {rc}; echo \"[run] {key} lane={lane} rc=$rc\"",
        ]
    parallel = ["LANE_PIDS=()"]
    for lane, body in enumerate(lane_lines):
        # Own scratch per lane; staggered first Kit start so lanes do not build
        # the shared shader/asset caches at the same moment.
        head = [f'  export TMPDIR="${{TMPDIR:-/tmp}}/lane{lane}"; mkdir -p "$TMPDIR"',
                f"  sleep {LANE_STAGGER_S * lane}"]
        parallel += ["(", *head, *body, ") &", "LANE_PIDS+=($!)", f"echo \"[run] lane {lane} pid=$!\""]
    parallel += ['wait "${LANE_PIDS[@]}"']
    lines[start:end] = parallel
    text = "\n".join(lines) + "\n"
    # Stop lane subshells and their supervisors too (workers follow their supervisor).
    old_cleanup = next(line for line in lines if line.startswith("cleanup() {"))
    text = text.replace(old_cleanup, (
        'cleanup() { for p in "${LANE_PIDS[@]:-}"; do [ -n "$p" ] && { pkill -TERM -P "$p" 2>/dev/null;'
        ' kill "$p" 2>/dev/null; }; done; ' + old_cleanup[len("cleanup() { "):]), 1)
    # Per-segment RC from the lanes' files (the associative array is lane-local).
    text = text.replace(
        'for k in "${!RC[@]}"; do [ "${RC[$k]}" = 0 ] ||\n'
        '  { echo "[run] FAILED segment $k rc=${RC[$k]}"; FAILED=1; }; done',
        'for f in "$OUT"/seg_*/.rc; do [ -f "$f" ] && [ "$(< "$f")" = 0 ] ||\n'
        '  { echo "[run] FAILED segment $f rc=$(cat "$f" 2>/dev/null)"; FAILED=1; }; done\n'
        f'[ "$(ls "$OUT"/seg_*/.rc 2>/dev/null | wc -l)" = {len(segments)} ] || '
        '{ echo "[run] FAILED: missing segment rc files"; FAILED=1; }')
    # One Qwen/Molmo pair now serves every lane's concurrent ensemble calls.
    text = "\n".join(
        line.replace("--max-num-seqs 8 ", f"--max-num-seqs {8 * lanes} ") if line.startswith('"$SERVE_PY"') else line
        for line in text.splitlines()) + "\n"
    script.write_text(text)


def write_behavior_plan(spec, lanes: int = 1):
    if not 1 <= lanes <= MAX_LANES:
        raise PreflightError(f"lanes must be 1..{MAX_LANES}")
    script = plan.write_plan(spec)
    original = shlex.quote(str(spec.capx_root / spec.launch_script))
    bootstrap = EXP / "src/behavior_collect/launch.py"
    text = script.read_text()
    assert text.count(original) == sum(len(t.seeds) for t in spec.tasks)
    script.write_text(text.replace(original, shlex.quote(str(bootstrap))))
    if lanes > 1:
        parallelize(spec, script, lanes)
    metadata_path = spec.output_dir / "plan.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["behavior_bootstrap"] = str(bootstrap)
    metadata["behavior_lanes"] = lanes
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return script


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--check", action="store_true", help="validate without writing a plan")
    p.add_argument("--lanes", type=int, default=1, help="segments run in this many parallel simulator lanes")
    a = p.parse_args()
    spec = validated_spec(a.spec)
    print("[plan] valid" if a.check else f"[plan] wrote {write_behavior_plan(spec, a.lanes)}")
