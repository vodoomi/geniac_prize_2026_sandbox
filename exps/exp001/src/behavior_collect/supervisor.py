"""Supervise fresh BEHAVIOR trial processes with deadlines independent of Python/GIL."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import yaml


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".new")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    pending.replace(path)


def report_phase(phase):
    path = os.environ.get("EXP001_BEHAVIOR_STATUS")
    if path:
        atomic_json(path, {"instance": int(os.environ["EXP001_BEHAVIOR_INSTANCE"]),
                           "phase": phase, "since": time.monotonic(),
                           "at": datetime.now().astimezone().isoformat(), "pid": os.getpid()})


def final_dirs(output, instance):
    # Match baseline's final-save contract without importing the simulator/torch.
    import re
    return [p for p in output.glob(f"trial_{instance:02d}_*")
            if (p / ".trial-complete.json").is_file()
            and (p / "summary.txt").is_file() and re.search(
                r"\n  Stdout: .*?\n  Stderr: .*?\n  Reward:",
                (p / "summary.txt").read_text(errors="replace"), re.S)]


def commit_trial(folder, infrastructure=False):
    """Commit after the whole artifact writer returns, including videos/images."""
    files = ["summary.txt", "infrastructure_failure.json"] if infrastructure else ["summary.txt", "code.py", "all_responses.json"]
    if not infrastructure:
        json.loads((folder / "all_responses.json").read_text())
    hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in files}
    atomic_json(folder / ".trial-complete.json", {"files": hashes, "infrastructure": infrastructure,
                "at": datetime.now().astimezone().isoformat()})


def terminate_group(proc, grace_s=5.0):
    # Each worker has its own session: terminate only it and its sidecars/IK.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_s
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    # Even if the leader exited, a hung grandchild may still own GPU memory/ports.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=5)
    wait_group_exit(proc.pid)


def wait_group_exit(pgid, timeout_s=180.0):
    """Wait until every process of the worker group is gone.

    A SIGKILLed CUDA sidecar (SAM3, GraspNet) can take tens of seconds to
    release GPU memory, and its listening socket keeps accepting connections
    meanwhile. The next trial would then see the port "already running", skip
    starting its own server, and lose perception once the old one vanishes.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return True
        time.sleep(0.5)
    print(f"[supervisor] process group {pgid} still alive after {timeout_s:.0f}s", flush=True)
    return False


def monitor(command, env, status_path, log_path, limits, poll_s=1.0):
    start = time.monotonic()
    phase = {"phase": "startup", "since": start}
    next_report = start
    with Path(log_path).open("ab", buffering=0) as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, close_fds=True)
        try:
            while proc.poll() is None:
                if Path(status_path).exists():
                    phase = json.loads(Path(status_path).read_text())
                elapsed = time.monotonic() - phase["since"]
                limit = limits[phase["phase"]]
                if time.monotonic() >= next_report:
                    print(f"[supervisor] pid={proc.pid} phase={phase['phase']} "
                          f"elapsed={elapsed:.0f}s limit={limit:g}s", flush=True)
                    next_report = time.monotonic() + 60
                if elapsed > limit:
                    try:
                        os.kill(proc.pid, signal.SIGUSR1)  # faulthandler stack in worker log
                    except ProcessLookupError:
                        pass
                    time.sleep(min(0.5, poll_s))
                    return {"ok": False, "reason": "hard_deadline", "phase": phase["phase"],
                            "elapsed_s": round(elapsed, 2), "limit_s": limit}
                time.sleep(poll_s)
            return {"ok": proc.returncode == 0, "reason": "exit", "rc": proc.returncode}
        finally:
            terminate_group(proc)


def record_infrastructure_failure(output, instance, incident):
    """Save an explicit operational failure, never a fabricated training example."""
    folder = output / f"trial_{instance:02d}_sandboxrc_-1_reward_0.000_taskcompleted_0"
    folder.mkdir(parents=True, exist_ok=True)
    reason = json.dumps(incident, ensure_ascii=False)
    (folder / "summary.txt").write_text(
        "Infrastructure failure; task outcome unavailable. See infrastructure_failure.json.\n"
        "Environment response:\n  Sandbox failed: -1\n  Stdout: \n"
        f"  Stderr: {reason}\n  Reward: 0\n  Task Completed: False\n"
        "  Terminated: False, Truncated: True\n")
    # No all_responses.json or code.py: baseline's training classifier excludes it.
    atomic_json(folder / "infrastructure_failure.json", incident)
    commit_trial(folder, infrastructure=True)


def supervise(argv):
    def terminate(_signum, _frame):
        raise SystemExit(143)
    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-trials", type=int)
    parser.add_argument("--num-workers", type=int)
    args, _ = parser.parse_known_args(argv)
    cfg = yaml.safe_load(Path(args.config_path).read_text())
    if (args.num_workers or cfg.get("num_workers", 1)) != 1:
        raise ValueError("BEHAVIOR supervision requires one worker per assigned GPU")
    end = args.total_trials if args.total_trials is not None else cfg.get("trials", 10)
    begin = cfg.get("resume_idx")
    begin = 1 if begin is None else begin
    if not 0 <= begin <= end <= 19:
        raise ValueError("Only development instances 0-19 are allowed")
    raw_output = Path(args.output_dir).resolve()
    output = raw_output.parent / args.model.replace("/", "_") / raw_output.name
    output.mkdir(parents=True, exist_ok=True)
    audit = output / "_supervision"
    audit.mkdir(exist_ok=True)
    limits = {"startup": 600., "setup": 600., "reset": 600.,
              "running": float(os.environ.get("CAPX_TRIAL_TIMEOUT_S", "2400")) + 120.,
              "shutdown": 60.}
    incidents = []
    command = [sys.executable, str(Path(__file__).with_name("launch.py")), *argv]
    for instance in range(begin, end + 1):
        existing = final_dirs(output, instance)
        if len(existing) > 1:
            raise RuntimeError(f"Multiple final saves for instance {instance}: {existing}")
        if existing:
            print(f"[supervisor] instance={instance} already finalized", flush=True)
            continue
        for attempt in (1, 2):
            state = audit / f"trial_{instance:02d}_attempt_{attempt}.json"
            if state.exists():
                raise FileExistsError(f"Refusing to overwrite supervision state: {state}")
            env = os.environ.copy()
            env.update(EXP001_BEHAVIOR_INSTANCE=str(instance), EXP001_BEHAVIOR_STATUS=str(state),
                       EXP001_BEHAVIOR_SUPERVISOR_PID=str(os.getpid()))
            print(f"[supervisor] instance={instance} attempt={attempt} starting", flush=True)
            result = monitor(command, env, state, state.with_suffix(".log"), limits)
            finals = final_dirs(output, instance)
            if len(finals) > 1:
                raise RuntimeError(f"Multiple final saves after worker exit: {finals}")
            # A final save can precede a stuck simulator shutdown. Preserve that
            # outcome, record the infrastructure incident, and start a fresh Kit.
            if not result["ok"] or not finals:
                incident = {"instance": instance, "attempt": attempt, "at": datetime.now().astimezone().isoformat(),
                            "result": result, "final_saved": bool(finals), "log": str(state.with_suffix('.log'))}
                incidents.append(incident)
                atomic_json(audit / "incidents.json", incidents)
                print(f"[supervisor] incident: {json.dumps(incident)}", flush=True)
            if finals:
                print(f"[supervisor] instance={instance} finalized: {finals[0].name}", flush=True)
                break
            # Preserve intermediate programs/images before a clean retry. They
            # stay outside discover_trials' search paths and cannot be counted twice.
            quarantine = audit / f"trial_{instance:02d}_attempt_{attempt}_partial"
            for folder in output.glob(f"trial_{instance:02d}_*"):
                quarantine.mkdir(exist_ok=True)
                shutil.move(str(folder), str(quarantine / folder.name))
            if attempt == 2:
                record_infrastructure_failure(output, instance, incidents[-1])
    atomic_json(audit / "complete.json", {"instances": list(range(begin, end + 1)),
                "infrastructure_incidents": len(incidents), "at": datetime.now().astimezone().isoformat()})
    print(f"[supervisor] segment completed; infrastructure incidents={len(incidents)}", flush=True)


def bind_worker_to_supervisor():
    """Stop this worker's whole process group when the supervisor disappears.

    Workers run in their own session so the supervisor can kill them as a
    group. If the supervisor itself is killed (SIGKILL, lost terminal), the
    worker would otherwise keep its sidecars, GPU memory and ports with no
    deadline. The kernel delivers SIGTERM on parent death; the handler then
    kills the group, including SAM3/GraspNet/IK children.
    """
    import ctypes

    def stop_group(_signum=None, _frame=None):
        os.killpg(os.getpgid(0), signal.SIGKILL)

    signal.signal(signal.SIGTERM, stop_group)
    libc = ctypes.CDLL(None, use_errno=True)
    PR_SET_PDEATHSIG = 1
    if libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    expected = os.environ.get("EXP001_BEHAVIOR_SUPERVISOR_PID")
    if expected and os.getppid() != int(expected):
        stop_group()
    return stop_group


def install_worker_runtime():
    """Install a single-instance selection and observable phase boundaries."""
    import faulthandler
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    report_phase("startup")
    from capx.envs import runner
    instance = int(os.environ["EXP001_BEHAVIOR_INSTANCE"])
    runner.MAX_TRIAL_RETRIES = 1
    original_parallel = runner.run_parallel_with_setup
    original_setup = runner._worker_setup
    original_run = runner._run_single_trial_worker
    original_headless = runner._run_headless_trials
    original_timeout_summary = runner._build_timeout_summary
    original_with_timeout = runner._run_single_trial_with_timeout
    current_env = []

    def with_timeout(env, *args, **kwargs):
        # The upstream timeout summary has no env handle; keep it so the frames
        # recorded before the deadline can still be written as videos.
        current_env[:] = [env]
        try:
            return original_with_timeout(env, *args, **kwargs)
        finally:
            current_env.clear()

    def parallel(trial_ids, **kwargs):
        if instance not in trial_ids:
            raise ValueError(f"Instance {instance} not requested by this segment")
        return original_parallel([instance], **kwargs)

    def setup(**kwargs):
        report_phase("setup")
        return original_setup(**kwargs)

    def run(*args, **kwargs):
        report_phase("reset")
        result = original_run(*args, **kwargs)
        if result.code_path:
            commit_trial(Path(result.code_path).parent)
        report_phase("shutdown")
        return result

    def headless(args, env_factory, config, start_time):
        config = dict(config)
        # Upstream treats total_trials=0 as an empty job. Instance zero is an
        # actual development layout; the parallel selector below keeps only it.
        if instance == 0 and config["total_trials"] == 0:
            config["total_trials"] = 1
        return original_headless(args, env_factory, config, start_time)

    def timeout_summary(trial, timeout_seconds, pa, config, exc):
        # The upstream partial state initializes num_code_blocks=0 and only
        # updates it on normal completion. Match normal summary semantics and
        # explicitly record that the deadline truncated the trial.
        pa = dict(pa)
        pa["num_code_blocks"] = len(pa.get("code_blocks", []))
        pa["truncated"] = True
        summary = original_timeout_summary(trial, timeout_seconds, pa, config, exc)
        if current_env and summary.code_path:
            save_timeout_videos(current_env[0], config, Path(summary.code_path).parent)
        return summary

    runner.run_parallel_with_setup = parallel
    runner._worker_setup = setup
    runner._run_single_trial_worker = run
    runner._run_headless_trials = headless
    runner._build_timeout_summary = timeout_summary
    runner._run_single_trial_with_timeout = with_timeout


def save_timeout_videos(env, config, trial_dir):
    """Write the frames recorded before a trial deadline, named like normal trials."""
    if not config.get("record_video") or not hasattr(env, "get_video_frames"):
        return
    from capx.utils.video_utils import _write_video
    try:
        frames = env.get_video_frames(clear=True)
        if isinstance(frames, dict):
            for key, stream in frames.items():
                if stream:
                    _write_video(stream, str(trial_dir), suffix=f"combined_{key}")
        elif frames:
            _write_video(frames, str(trial_dir), suffix="combined")
    except Exception as exc:  # a video failure must not lose the trial record
        print(f"[supervisor] could not save timeout videos: {exc!r}", flush=True)


if __name__ == "__main__":
    supervise(sys.argv[1:])
