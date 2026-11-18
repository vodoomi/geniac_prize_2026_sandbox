"""Generate exp001 LIBERO collection specs from the development split."""

from __future__ import annotations

if __package__ in (None, ""):
    # Run as a file: put src/ (not this package directory) first on sys.path so
    # sibling modules are imported through the package and cannot shadow others.
    import sys as _sys
    from pathlib import Path as _Path
    _here = _Path(__file__).resolve().parent
    _sys.path[:] = [p for p in _sys.path if _Path(p or ".").resolve() != _here]
    _sys.path.insert(0, str(_here.parent))

import ast
import copy
from pathlib import Path
import re

import yaml


def main() -> None:
    exp = Path(__file__).resolve().parents[2]
    root = exp.parents[1]
    baseline = root / "third_party/geniac_prize_2026_baseline"
    capx = baseline / "geniac_prize_2026_code"
    split = yaml.safe_load((root / "evaluation/split.yaml").read_text())
    development = split["development_task_ids"]["libero"]
    holdout = set(split["validation_task_ids"]["libero"])
    assert len(development) == len(set(development)) == 104
    assert not set(development) & holdout
    assert split["initial_state_policy"]["libero"]["development_seed_range"] == [301, 320]
    pack = yaml.safe_load((root / "third_party/geniac_prize_2026_code/taskpacks/original/pack.yaml").read_text())
    public = {t["task_id"]: t for t in pack["tasks"]}
    libero = capx / "capx/third_party/LIBERO-PRO/libero/libero"
    tree = ast.parse((libero / "benchmark/libero_suite_task_map.py").read_text())
    task_map = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "libero_task_map" for t in n.targets))
    configs = exp / "configs"
    tasks_dir = configs / "libero_tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    template_path = baseline / "configs/capx/geniac_prize/libero/libero_spatial_on_stove.yaml"
    template = template_path.read_text()
    tasks = []
    for task in development:
        suite, index = task.rsplit("_", 1)
        index = int(index)
        assert public[task]["simulator"] == "libero"
        assert public[task]["env_target"] == f"franka_libero_{task}_low_level"
        name = task_map[suite][index]  # All five public suites use task_order_index=0.
        for rel in [f"bddl_files/{suite}/{name}.bddl", f"init_files/{suite}/{name}.pruned_init"]:
            if not (libero / rel).is_file():
                raise FileNotFoundError(libero / rel)
        text, count = re.subn(r"(?m)^(\s+suite_name:) .*$", rf"\1 {suite}", template)
        assert count == 1
        text, count = re.subn(r"(?m)^(\s+task_id:) \d+\s*$", rf"\g<1> {index}", text)
        assert count == 1
        # The gateway fills {task_instruction} with the taskpack instruction in every
        # prompt. The in-process loop fills {libero_environment_goal} (the BDDL
        # language, worded differently) only into the first prompt, so write the
        # taskpack instruction into the template here.
        instruction = public[task]["instruction"]
        assert "\n" not in instruction and "{" not in instruction and "}" not in instruction
        text, count = re.subn(r"(?m)^(\s+Goal:) \{libero_environment_goal\}$",
                              lambda m: f"{m.group(1)} {instruction}", text)
        assert count == 1
        steplimit = public[task].get("trial_steplimit", pack["limits"]["trial_steplimit"])
        text, count = re.subn(r"(?m)^(\s+max_steps:) \d+\s*$", rf"\g<1> {steplimit}", text)
        assert count == 1
        text = f"# exp001: {task}; generated from {template_path.relative_to(root)}\n" + text
        (tasks_dir / f"{task}.yaml").write_text(text)
        tasks.append({"task": task, "config": f"libero_tasks/{task}.yaml", "layouts": 50, "seeds": ["301-320"]})

    runtime = configs / "libero_runtime"
    runtime.mkdir(exist_ok=True)
    paths = {"benchmark_root": str(libero), "assets": str(libero / "assets"),
             "bddl_files": str(libero / "bddl_files"), "init_states": str(libero / "init_files"),
             "datasets": str(libero.parent / "datasets")}
    (runtime / "config.yaml").write_text(yaml.safe_dump(paths, sort_keys=False))
    spec = yaml.safe_load((baseline / "configs/my_collect.yaml").read_text())
    spec.update({"capx_root": "../../../third_party/geniac_prize_2026_baseline/geniac_prize_2026_code",
                 "sim_python": ".venv-libero/bin/python", "num_workers": 8,
                 "chat_template": "../../../third_party/geniac_prize_2026_baseline/runs/qwen38_low_reasoning.jinja",
                 "b300_workarounds": True, "tasks": tasks,
                 # libero_collect/launch.py (success latch) imports from exp001/src.
                 "pythonpath": f"{capx}:{exp / 'src'}"})
    spec.setdefault("extra_env", {}).update({"VLLM_USE_FLASHINFER_SAMPLER": "0", "LIBERO_CONFIG_PATH": str(runtime)})
    for key in ["model_path", "molmo_path"]:
        if not spec.get(key) or not (Path(spec[key]) / "config.json").is_file():
            raise FileNotFoundError(f"{key}: {spec.get(key)}")

    def save(name: str, selected: list[dict]) -> None:
        item = copy.deepcopy(spec)
        # v2: taskpack instruction in every prompt, taskpack step limit, success
        # latch. v1 outputs (runs/collect_libero*) must not be mixed in.
        item.update(tasks=selected, output_dir=f"../runs/{name}_v2")
        path = configs / f"{name}.yaml"
        header = "# exp001 LIBERO collection; paths are relative to this YAML.\n"
        path.write_text(header + yaml.safe_dump(item, sort_keys=False, allow_unicode=True))
        print(f"{path.relative_to(root)}: {len(selected)} tasks")

    save("collect_libero", tasks)
    smoke = [copy.deepcopy(t) for t in tasks if t["task"] in {"libero_spatial_0", "libero_object_0", "libero_goal_0"}]
    assert len(smoke) == 3
    for t in smoke:
        t["seeds"] = ["301-308"]
    save("collect_libero_smoke", smoke)
    for i in range(4):
        save(f"collect_libero_part{i + 1:02d}", tasks[i * 26:(i + 1) * 26])


if __name__ == "__main__":
    main()
