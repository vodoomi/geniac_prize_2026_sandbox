"""Run CaP-X's LIBERO collection with gateway-equivalent scoring.

Workers are started with the ``spawn`` method, which re-imports this file as
``__mp_main__`` in every worker; the patch is therefore applied at import time.
"""
if __package__ in (None, ""):
    # Run as a file: put src/ (not this package directory) first on sys.path so
    # sibling modules are imported through the package and cannot shadow others.
    import sys as _sys
    from pathlib import Path as _Path
    _here = _Path(__file__).resolve().parent
    _sys.path[:] = [p for p in _sys.path if _Path(p or ".").resolve() != _here]
    _sys.path.insert(0, str(_here.parent))

from capx.envs.simulators.libero import FrankaLiberoEnv

from libero_collect.parity import install

install(FrankaLiberoEnv)

if __name__ == "__main__":
    import tyro
    from capx.envs.launch import LaunchArgs, main

    print("[libero-parity] success latch enabled", flush=True)
    main(tyro.cli(LaunchArgs))
