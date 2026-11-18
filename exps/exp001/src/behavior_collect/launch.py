"""Initialize Kit, adapt control APIs, then run CaP-X's collection loop."""
if __package__ in (None, ""):
    # Run as a file: put src/ (not this package directory) first on sys.path so
    # sibling modules are imported through the package and cannot shadow others.
    import sys as _sys
    from pathlib import Path as _Path
    _here = _Path(__file__).resolve().parent
    _sys.path[:] = [p for p in _sys.path if _Path(p or ".").resolve() != _here]
    _sys.path.insert(0, str(_here.parent))

import os
import sys


def initialize_behavior():
    import omnigibson as og
    from omnigibson.macros import gm

    # Match the competition simulator's macros before Kit consumes them.
    gm.ENABLE_OBJECT_STATES = True
    gm.USE_GPU_DYNAMICS = False
    gm.HEADLESS = True
    original_argv = sys.argv[:]
    try:
        sys.argv = sys.argv[:1]
        og.launch()
    finally:
        sys.argv = original_argv
    # The simulator module repeats these same assignments on import. OmniGibson
    # 3.9 locks macros after their first use, so permit the identical assignments.
    with gm.unlocked():
        from capx.envs.simulators.r1pro_b1k import R1ProBehaviourLowLevel
    from behavior_collect.compat import install
    install(R1ProBehaviourLowLevel)
    return og, R1ProBehaviourLowLevel


def main():
    # The parent never imports Kit/JAX, so native stalls cannot block its clock.
    if "EXP001_BEHAVIOR_INSTANCE" not in os.environ:
        from behavior_collect.supervisor import supervise
        supervise(sys.argv[1:])
        return
    from behavior_collect.supervisor import bind_worker_to_supervisor, report_phase
    import faulthandler
    import signal
    stop_group = bind_worker_to_supervisor()
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    report_phase("startup")
    og, _ = initialize_behavior()
    # Kit may install its own handlers during launch; keep group termination.
    signal.signal(signal.SIGTERM, stop_group)
    from behavior_collect.supervisor import install_worker_runtime
    install_worker_runtime()
    from capx.envs.launch import LaunchArgs, main as launch
    import tyro

    try:
        launch(tyro.cli(LaunchArgs))
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
