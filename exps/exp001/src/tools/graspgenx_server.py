"""GraspGen-X grasp planning as a local HTTP service (runs in the GraspGenX venv).

POST /plan  {"points": [[x, y, z], ...] world-frame object points (metres, +Z up),
             "num_grasps": int, "topk": int}
  -> {"grasps": [K x 4 x 4] world-frame gripper poses, "scores": [K]}

Gripper frame (galaxea_g1, the R1Pro hand): +Z approach, +X finger closing,
fingertips at z = FINGERTIP_DEPTH. GET /health -> {"ok": true, "gripper": ...}.
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "runtime/GraspGenX"


def build(gripper):
    from graspgenx import get_checkpoints_version_dir
    from graspgenx.grasp_server import GraspGenXSampler
    from graspgenx.utils.checkpoint_io import load_model_cfg
    root = get_checkpoints_version_dir()
    cfg = load_model_cfg(str(root / "gen"), str(root / "dis"), None, None)
    assets = ROOT / "ext/gripper_descriptions/gripper_descriptions/assets"
    return GraspGenXSampler(cfg, gripper, assets_dir=str(assets), diffusion_grasps=200,
                            obb_grasps=10 * 5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8390)
    p.add_argument("--gripper", default="galaxea_g1")
    p.add_argument("--planner", default="graspmoe", choices=["graspmoe", "diffusion"])
    a = p.parse_args()
    os.environ.setdefault("GRASPGENX_CHECKPOINT_DIR", str(ROOT / "ext/graspgenx_checkpoints"))
    from graspgenx.samplers import run_planner_on_object
    sampler = build(a.gripper)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"ok": True, "gripper": a.gripper, "planner": a.planner})

        def do_POST(self):
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                points = np.asarray(request["points"], dtype=np.float32)
                if points.ndim != 2 or points.shape[1] != 3 or len(points) < 10 or not np.isfinite(points).all():
                    return self._send(400, {"error": "points must be a finite Nx3 array with N >= 10"})
                center = points.mean(axis=0)
                with lock:
                    grasps, scores, tags, _ = run_planner_on_object(
                        points - center, sampler, planner=a.planner, grasp_threshold=-1.0,
                        num_grasps=int(request.get("num_grasps", 200)),
                        topk_num_grasps=int(request.get("topk", 100)))
                grasps = np.asarray(grasps, dtype=float).copy()
                grasps[:, :3, 3] += center
                order = np.argsort(-np.asarray(scores))
                self._send(200, {"grasps": grasps[order].tolist(),
                                 "scores": np.asarray(scores, dtype=float)[order].tolist(),
                                 "branch": [tags[i] for i in order]})
            except Exception as exc:  # report to the client; the service keeps running
                self._send(500, {"error": repr(exc)})

        def log_message(self, *args):
            pass

    print(f"[graspgenx] serving {a.gripper} ({a.planner}) on 127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
