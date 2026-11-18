"""Run CaP-X's LIBERO collection with gateway-equivalent scoring.

Workers are started with the ``spawn`` method, which re-imports this file as
``__mp_main__`` in every worker; the patch is therefore applied at import time.
"""
from capx.envs.simulators.libero import FrankaLiberoEnv

from libero_parity import install

install(FrankaLiberoEnv)

if __name__ == "__main__":
    import tyro
    from capx.envs.launch import LaunchArgs, main

    print("[libero-parity] success latch enabled", flush=True)
    main(tyro.cli(LaunchArgs))
