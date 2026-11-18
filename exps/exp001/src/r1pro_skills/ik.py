"""Run native/JAX IK in a persistent, bounded subprocess, outside Kit's threads."""
from __future__ import annotations

if __package__ in (None, ""):
    # Run as a file: put src/ (not this package directory) first on sys.path so
    # sibling modules are imported through the package and cannot shadow others.
    import sys as _sys
    from pathlib import Path as _Path
    _here = _Path(__file__).resolve().parent
    _sys.path[:] = [p for p in _sys.path if _Path(p or ".").resolve() != _here]
    _sys.path.insert(0, str(_here.parent))

import atexit
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

import numpy as np


class IKSolverTimeout(RuntimeError):
    """The solver was terminated without sending any joint commands."""


class BoundedIK:
    def __init__(self, urdf, timeout_s=90.0):
        self.urdf = str(urdf)
        self.timeout_s = timeout_s
        self.proc = None
        self.buffer = b""
        atexit.register(self.close)

    def close(self):
        proc, self.proc = self.proc, None
        self.buffer = b""
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        proc.stdin.close()
        proc.stdout.close()

    def solve_ik_rest(self, *, robot, target_link_name, target_position,
                      target_wxyz, rest_cost_weights, initial_q):
        # Shapes/dtypes stay fixed across calls, avoiding repeated JIT variants.
        payload = {"target_link_name": target_link_name}
        for name, value, shape in (
            ("target_position", target_position, (3,)),
            ("target_wxyz", target_wxyz, (4,)),
            ("rest_cost_weights", rest_cost_weights, (18,)),
            ("initial_q", initial_q, (18,)),
        ):
            value = np.asarray(value, dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid IK {name}: expected finite shape {shape}")
            payload[name] = value.tolist()
        if target_link_name not in ("left_gripper_link", "right_gripper_link"):
            raise ValueError(f"Invalid R1Pro target link: {target_link_name}")
        if self.proc is None or self.proc.poll() is not None:
            self.close()
            env = os.environ.copy()
            env.update(JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            # CPU IK uses no GPU and does not share Kit/PyTorch's native runtime.
            self.proc = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker", self.urdf],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
                close_fds=True,
            )
        deadline = time.monotonic() + self.timeout_s
        try:
            self.proc.stdin.write((json.dumps(payload, allow_nan=False) + "\n").encode())
            self.proc.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(self.proc.stdout, selectors.EVENT_READ)
                while b"\n" not in self.buffer:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise IKSolverTimeout(f"IK exceeded {self.timeout_s:g}s; solver stopped")
                    chunk = os.read(self.proc.stdout.fileno(), 65536)
                    if not chunk:
                        raise RuntimeError("IK worker exited without a response")
                    self.buffer += chunk
            line, self.buffer = self.buffer.split(b"\n", 1)
            response = json.loads(line)
            if "error" in response:
                raise RuntimeError(f"IK worker: {response['error']}")
            joints = np.asarray(response["joints"], dtype=np.float64)
            if joints.shape != (18,) or not np.isfinite(joints).all():
                raise RuntimeError("IK worker returned invalid joints")
            return joints
        except BaseException:
            self.close()
            raise


def worker(urdf):
    # Keep the protocol separate from libraries' initialization/debug prints.
    protocol = sys.stdout
    sys.stdout = sys.stderr
    from capx.integrations.motion.pyroki_context import get_pyroki_context
    from capx.integrations.motion.pyroki_snippets import solve_ik_rest
    context = get_pyroki_context(urdf, target_link_name="left_gripper_link")
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            for key in ("target_position", "target_wxyz", "rest_cost_weights", "initial_q"):
                payload[key] = np.asarray(payload[key], dtype=np.float32)
            joints = solve_ik_rest(robot=context.robot, **payload)
            response = {"joints": np.asarray(joints).tolist()}
        except Exception as error:
            response = {"error": f"{type(error).__name__}: {error}"}
        protocol.write(json.dumps(response, allow_nan=False) + "\n")
        protocol.flush()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("usage: ik.py --worker URDF")
    worker(sys.argv[2])
