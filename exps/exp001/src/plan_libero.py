"""Plan exp001 LIBERO collection with baseline's planner, checked against the evaluation.

Refuses a spec unless every task config
* uses the submission's prompt template (baseline ``submission/gateway/franka_libero.yaml``)
  with the taskpack instruction already in the goal line, so every turn's prompt is
  what the gateway path produces (the in-process loop only fills
  ``{libero_environment_goal}`` into the first prompt);
* has the same multi-turn prompt and APIs as that submission config;
* has the simulator horizon equal to the taskpack's ``trial_steplimit``.
The generated ``run.sh`` launches ``launch_libero.py`` (success latch) instead of
CaP-X's ``launch.py``.
"""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import yaml
from geniac_baseline.collect import plan
from geniac_baseline.collect.preflight import PreflightError

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
BASELINE = ROOT / "third_party/geniac_prize_2026_baseline"
SUBMISSION_CONFIG = BASELINE / "submission/gateway/franka_libero.yaml"
TASKPACK = ROOT / "third_party/geniac_prize_2026_code/taskpacks/original/pack.yaml"
PLACEHOLDER = "{task_instruction}"
BOOTSTRAP = EXP / "src/launch_libero.py"


def taskpack_tasks() -> dict[str, dict]:
    pack = yaml.safe_load(TASKPACK.read_text())
    default = pack["limits"]["trial_steplimit"]
    return {t["task_id"]: {**t, "trial_steplimit": t.get("trial_steplimit", default)} for t in pack["tasks"]}


def check_task_config(task: str, cfg: dict, tasks: dict, submission: dict) -> None:
    if task not in tasks or tasks[task]["simulator"] != "libero":
        raise PreflightError(f"{task}: not a LIBERO task of the taskpack")
    env = cfg["env"]["cfg"]
    expected_prompt = submission["prompt"].replace(PLACEHOLDER, tasks[task]["instruction"])
    if env["prompt"] != expected_prompt:
        raise PreflightError(f"{task}: prompt differs from the submission template with the taskpack instruction")
    if env.get("multi_turn_prompt") != submission["multi_turn_prompt"] or env["apis"] != submission["apis"]:
        raise PreflightError(f"{task}: multi-turn prompt or APIs differ from the submission config")
    if env["low_level"].get("max_steps") != tasks[task]["trial_steplimit"]:
        raise PreflightError(f"{task}: max_steps must equal trial_steplimit {tasks[task]['trial_steplimit']}")
    if env.get("privileged") or env["low_level"].get("privileged") or cfg.get("use_oracle_code"):
        raise PreflightError(f"{task}: privileged observation or oracle code")


def validated_spec(path: Path):
    spec = plan.load_spec(path)
    src = str(EXP / "src")
    if not spec.pythonpath or src not in spec.pythonpath.split(":"):
        raise PreflightError(f"pythonpath must include {src} (launch_libero.py imports libero_parity)")
    tasks = taskpack_tasks()
    submission = yaml.safe_load(SUBMISSION_CONFIG.read_text())
    for task in spec.tasks:
        check_task_config(task.task, yaml.safe_load(task.config.read_text()), tasks, submission)
    plan.preflight(spec)
    return spec


def write_libero_plan(spec) -> Path:
    script = plan.write_plan(spec)
    original = shlex.quote(str(spec.capx_root / spec.launch_script))
    text = script.read_text()
    if text.count(original) != sum(len(t.seeds) for t in spec.tasks):
        raise RuntimeError("Unexpected run.sh layout; refusing to rewrite the launch command")
    script.write_text(text.replace(original, shlex.quote(str(BOOTSTRAP))))
    metadata_path = spec.output_dir / "plan.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["libero_bootstrap"] = str(BOOTSTRAP)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return script


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--check", action="store_true", help="validate without writing a plan")
    a = p.parse_args()
    spec = validated_spec(a.spec)
    print("[plan] valid" if a.check else f"[plan] wrote {write_libero_plan(spec)}")
