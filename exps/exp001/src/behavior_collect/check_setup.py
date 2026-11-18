"""Check runtime imports, RTX renderer, and a real public instance without an LLM."""
if __package__ in (None, ""):
    # Run as a file: put src/ (not this package directory) first on sys.path so
    # sibling modules are imported through the package and cannot shadow others.
    import sys as _sys
    from pathlib import Path as _Path
    _here = _Path(__file__).resolve().parent
    _sys.path[:] = [p for p in _sys.path if _Path(p or ".").resolve() != _here]
    _sys.path.insert(0, str(_here.parent))

from pathlib import Path
import json
import os
import yaml


def check_instance_files():
    exp = Path(__file__).resolve().parents[2]
    root = exp.parents[1]
    split = yaml.safe_load((root / "evaluation/split.yaml").read_text())
    controllers = root / "third_party/geniac_prize_2026_baseline/geniac_prize_2026_code/capx/envs/simulators/configs"
    data = Path(os.environ["OMNIGIBSON_DATA_PATH"])
    records = {}
    for task in split["development_task_ids"]["behavior"]:
        cfg = yaml.safe_load((exp / f"configs/behavior_tasks/{task}.yaml").read_text())
        low = cfg["env"]["cfg"]["low_level"]
        controller = yaml.safe_load((controllers / low["controller_cfg"]).read_text())
        activity = low.get("activity_name", controller["task"]["activity_name"])
        scene = controller["scene"]["scene_model"]
        definition = controller["task"]["activity_definition_id"]
        # Mirror competition CaP-X's 2026-first directory selection exactly.
        candidates = [data / dataset / "scenes" / scene / "json" / f"{scene}_task_{activity}_instances"
                      for dataset in ("2026-challenge-task-instances", "2025-challenge-task-instances")]
        directory = next((p for p in candidates if p.is_dir()), None)
        if directory is None:
            raise FileNotFoundError(f"{task}: no public instance directory in {candidates}")
        files = [directory / f"{scene}_task_{activity}_{definition}_{i}_template-tro_state.json"
                 for i in split["initial_state_policy"]["behavior"]["development_instance_ids"]]
        for f in files:
            if not f.is_file():
                raise FileNotFoundError(f)
        records[task] = [str(f.relative_to(data)) for f in files]
    (exp / "runtime/instances.json").write_text(json.dumps(records, indent=2))
    print(f"[check] {sum(map(len, records.values()))} public instance files verified", flush=True)


def main():
    check_instance_files()
    import torch
    # Native libraries used by perception/IK can conflict with Kit startup if
    # imported first. Initialize Kit before importing CaP-X's API registry.
    from behavior_collect.launch import initialize_behavior
    og, R1ProBehaviourLowLevel = initialize_behavior()
    from sam3.model.sam3_image_processor import Sam3Processor
    from capx.integrations.r1pro.control import R1ProControlApi

    assert torch.cuda.is_available(), "CUDA is unavailable"
    assert og.__version__ == "3.9.2"
    print(f"[check] torch={torch.__version__}, OmniGibson={og.__version__}, GPU={torch.cuda.get_device_name()}", flush=True)
    env = R1ProBehaviourLowLevel()
    try:
        obs, _ = env.reset(seed=0, options={"trial": 0})
        frame = env.render()
        assert frame.ndim == 3 and frame.shape[-1] == 3 and frame.size > 0
        assert frame.std() > 0, "Renderer returned a constant image"
        from PIL import Image
        Image.fromarray(frame).save(Path(__file__).resolve().parents[2] / "runtime/check_rgb.png")
        result = {"hostname": os.uname().nodename, "slurm_job_id": os.environ["SLURM_JOB_ID"],
                  "gpu": torch.cuda.get_device_name(), "instance_id": 0,
                  "frame_shape": list(frame.shape), "observation_keys": list(obs)}
        (Path(__file__).resolve().parents[2] / "runtime/check.json").write_text(json.dumps(result, indent=2))
        print("[check] instance 0 reset + RGB render OK", flush=True)
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
