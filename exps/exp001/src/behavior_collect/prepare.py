"""Generate BEHAVIOR collection configs from exp001's development split."""
from __future__ import annotations

if __package__ in (None, ""):
    # Run as a file: put src/ (not this package directory) first on sys.path so
    # sibling modules are imported through the package and cannot shadow others.
    import sys as _sys
    from pathlib import Path as _Path
    _here = _Path(__file__).resolve().parent
    _sys.path[:] = [p for p in _sys.path if _Path(p or ".").resolve() != _here]
    _sys.path.insert(0, str(_here.parent))

import copy
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[2]
ROOT = EXP.parents[1]
BASELINE = ROOT / "third_party/geniac_prize_2026_baseline"
CAPX = BASELINE / "geniac_prize_2026_code"
# Collection-only step cap (the gateway manifest keeps 8000 at evaluation).
# Successful trials usually finish within 5000 steps; capping saves time on
# failures. Trials cut here are labelled failures even if they would have
# succeeded by 8000 steps.
DATA_VERSION = "v10"   # collection output suffix; bump when the API or its prompt text changes
# Wall-clock limit per trial. The step cap is the intended limit; this only
# guards hangs, with room for two lanes sharing one GPU.
COLLECT_TRIAL_TIMEOUT_S = 3600


# Collection-only instruction overrides for public-pack tasks whose manifest
# instruction disagrees with the simulator's success check. At evaluation the
# gateway still sends the manifest instruction; these only steer data collection.
INSTRUCTION_OVERRIDES = {
    # Success requires can_of_soda_113, the only blue can (r1pro_b1k.py
    # pick_up_trash_reward); the scene also has two orange cans. The CaP-X task
    # prompt (r1pro_pickup_trash.py) names the blue can, the public pack does not.
    "picking_up_trash": ("Find the can of soda on the floor and pick it up.",
                         "Find the blue can of soda on the floor and pick it up."),
}


def task_instruction(task: str, instruction: str) -> str:
    """The instruction used in the collection prompt for ``task``.

    The manifest instruction verbatim, except for INSTRUCTION_OVERRIDES. An
    override applies only while the manifest still has the text it was written
    for, so a changed public pack fails loudly instead of being silently replaced.
    """
    if task not in INSTRUCTION_OVERRIDES:
        return instruction
    expected, override = INSTRUCTION_OVERRIDES[task]
    if instruction != expected:
        raise ValueError(f"{task}: manifest instruction changed to {instruction!r}; review INSTRUCTION_OVERRIDES")
    return override


class _BlockDumper(yaml.SafeDumper):
    """Write multi-line strings as ``|`` blocks.

    geniac_baseline.data.prompts.yaml_template reads ``multi_turn_prompt: |``
    verbatim from the run config when converting trials to SFT/DMPO rows.
    """


def _str(dumper, value):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_BlockDumper.add_representer(str, _str)


def dump(data) -> str:
    return yaml.dump(data, Dumper=_BlockDumper, sort_keys=False, allow_unicode=True, width=10**6)


def main() -> None:
    split = yaml.safe_load((ROOT / "evaluation/split.yaml").read_text())
    development = split["development_task_ids"]["behavior"]
    holdout = split["validation_task_ids"]["behavior"]
    assert len(set(development)) == len(development) == 40
    assert not set(development).intersection(holdout)
    assert split["initial_state_policy"]["behavior"]["development_instance_ids"] == list(range(20))
    pack = yaml.safe_load((ROOT / split["source"]["taskpack"]).read_text())
    public = {t["task_id"]: t for t in pack["tasks"] if t["simulator"] == "behavior"}
    base_cfg = yaml.safe_load((BASELINE / "configs/capx/geniac_prize/robosuite/cube_lifting.yaml").read_text())
    tasks_dir = EXP / "configs/behavior_tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    controllers_dir = EXP / "configs/behavior_controllers"
    controllers_dir.mkdir(parents=True, exist_ok=True)
    metadata = yaml.safe_load((EXP / "runtime/data/2026-challenge-task-instances/metadata/available_tasks.yaml").read_text())
    tasks = []
    for task in development:
        t = public[task]
        cfg = copy.deepcopy(base_cfg)
        low = {"_target_": "capx.envs.simulators.r1pro_b1k.R1ProBehaviourLowLevel",
               "controller_cfg": "r1pro_primitives.yaml", **t.get("env_kwargs", {})}
        activity = "turning_on_radio" if task == "picking_up_radio" else task
        controller = yaml.safe_load((CAPX / "capx/envs/simulators/configs" / low["controller_cfg"]).read_text())
        controller["scene"]["scene_model"] = metadata[activity][0]["scene_model"]
        controller["task"]["activity_name"] = activity
        controller_path = controllers_dir / f"{task}.yaml"
        controller_path.write_text(yaml.safe_dump(controller, sort_keys=False))
        low["controller_cfg"] = str(controller_path)
        high = "r1pro_behavior.R1ProBehaviorCodeEnv"
        if task == "picking_up_radio":
            high = "r1pro_pickup_radio.R1ProRadioCodeEnv"
        elif task == "picking_up_trash":
            high = "r1pro_pickup_trash.R1ProTrashCodeEnv"
        cfg["env"] = {"_target_": f"capx.envs.tasks.r1pro.{high}", "cfg": {
            "_target_": "capx.envs.tasks.base.CodeExecEnvConfig", "low_level": low,
            "privileged": False, "apis": ["R1ProControlApi"],
            "prompt": "You are controlling a R1Pro robot with the API described below.\n"
                      f"Goal: {task_instruction(task, t['instruction'])}\n"
                      "Write executable Python code only. Import libraries explicitly when needed.\n",
            "multi_turn_prompt": base_cfg["env"]["cfg"]["multi_turn_prompt"],
        }}
        cfg["api_servers"] = copy.deepcopy(base_cfg["api_servers"][:2])
        cfg.update(trials=19, resume_idx=0, num_workers=1, use_oracle_code=False,
                   output_dir=f"./outputs/{task}", save_multiturn_prompts=True)
        (tasks_dir / f"{task}.yaml").write_text(dump(cfg))
        tasks.append({"task": task, "config": f"behavior_tasks/{task}.yaml", "seeds": ["0-19"]})
    spec = yaml.safe_load((EXP / "configs/collect_libero.yaml").read_text())
    spec.update(sim_python=str(EXP / "runtime/venv/bin/python"), num_workers=1,
                pythonpath=f"{CAPX}:{EXP / 'src'}", qwen_gpu_util=0.45, molmo_gpu_util=0.18,
                port_offset=200, vllm_port_offset=200)
    spec["trial_timeout_s"] = COLLECT_TRIAL_TIMEOUT_S
    spec["extra_env"] = {"VLLM_USE_FLASHINFER_SAMPLER": "0", "OMNI_KIT_ACCEPT_EULA": "YES",
                         "OMNIGIBSON_HEADLESS": "1", "OMNIGIBSON_DATA_PATH": str(EXP / "runtime/data"),
                         # pick_object grasps with GraspGen-X (src/tools/graspgenx_server.py on :8390).
                         "EXP001_GRASP_PLANNER": "graspgenx",
                         "EXP001_GRASPGENX_URL": "http://127.0.0.1:8390"}

    def save(name: str, selected: list[dict], output_name: str | None = None) -> None:
        item = copy.deepcopy(spec)
        item.update(tasks=selected, output_dir=f"../runs/{output_name or name}")
        (EXP / "configs" / f"{name}.yaml").write_text(
            "# BEHAVIOR seeds are actual instance IDs; use src/behavior_collect/plan.py.\n"
            + yaml.safe_dump(item, sort_keys=False))
        print(f"[prepare] {name}: {len(selected)} tasks")

    save("collect_behavior", tasks, output_name=f"collect_behavior_{DATA_VERSION}")
    for i in range(4):
        name = f"collect_behavior_part{i + 1:02d}"
        save(name, tasks[i * 10:(i + 1) * 10], output_name=f"{name}_{DATA_VERSION}")


if __name__ == "__main__":
    main()
